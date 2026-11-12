"""Retrieval evaluation: hit@k and MRR for dense, sparse, hybrid, hybrid+rerank on a golden set generated from the
documents themselves (question -> the section that answers it). Usage: python -m evaluation.retrieval
"""

from __future__ import annotations

import json
import statistics
import time

from radar.config import ARTIFACTS_DIR, DATA_DIR, settings
from radar.retrieval.store import DocumentStore

# Hand-written questions with the section that answers them (doc_id, section prefix). Written against the generator's
# policies so they are stable; contract questions are templated per supplier.
GOLDEN: list[tuple[str, str, str]] = [
    ("What PFAS content must suppliers declare?", "POL-ENV-001", "3"),
    ("How fast must Compliance issue a PFAS impact assessment after a restriction is published?", "POL-ENV-001", "4"),
    ("Which parts are in scope of the PFAS policy?", "POL-ENV-001", "2"),
    ("What is the RoHS threshold for cadmium?", "POL-ENV-002", "2"),
    ("How is exemption 7(a) for lead in solder tracked?", "POL-ENV-002", "3"),
    ("What format do RoHS material declarations use?", "POL-ENV-002", "4"),
    ("When does a new SVHC trigger parts screening and how long do we have?", "POL-ENV-003", "1"),
    ("When is Article 33 communication and SCIP notification required?", "POL-ENV-003", "2"),
    ("How often must suppliers submit a CMRT?", "POL-SRC-004", "2"),
    ("What happens if a smelter in the chain is non-conformant?", "POL-SRC-004", "3"),
    ("Which suppliers get a corrective action plan based on RBA audit score?", "POL-SR-005", "2"),
    ("What is the policy on recruitment fees and forced labour?", "POL-SR-005", "3"),
    ("How often are suppliers screened against sanctions lists?", "POL-TRD-006", "1"),
    ("Which items need ECCN classification before shipment?", "POL-TRD-006", "2"),
    ("What happens to open purchase orders when a supplier is newly listed?", "POL-TRD-006", "3"),
    ("What supplier data is needed for CSRD reporting?", "POL-ESG-007", "2"),
    ("What must cell suppliers provide under the battery regulation policy?", "POL-ENV-008", "2"),
    ("By when is EPS foam packaging phased out?", "POL-ENV-009", "1"),
    ("Which products need an SBOM under the cybersecurity policy?", "POL-SEC-010", "1"),
    ("How long do we have to complete a right-to-erasure request and what does it include?", "POL-PRV-011", "1"),
    ("How quickly must a regulatory change be logged after publication?", "POL-GOV-012", "1"),
    ("Who approves an impact brief before tasks are issued?", "POL-GOV-012", "3"),
    ("How is urgency defined for regulatory changes?", "POL-GOV-012", "4"),
    ("What are the steps when handling a regulatory change on conflict minerals?", "SOP-104", "2"),
    ("What triggers the PFAS regulatory change SOP?", "SOP-101", "1"),
]


# Paraphrased questions that avoid the documents' own vocabulary (synonyms, lay terms, indirect phrasing).
# This is the set that separates retrieval methods; the direct set above mostly checks nothing is broken.
HARD: list[tuple[str, str, str]] = [
    ("How much 'forever chemicals' content do vendors have to tell us about?", "POL-ENV-001", "3"),
    ("After regulators announce a ban on non-stick fluorinated coatings, how many days until our assessment is due?", "POL-ENV-001", "4"),
    ("Are Teflon-type seals covered by our chemical restrictions?", "POL-ENV-001", "2"),
    ("What is the maximum concentration of toxic heavy metals allowed in a homogeneous material?", "POL-ENV-002", "2"),
    ("Do we keep a register of legal carve-outs for hazardous solder and when they lapse?", "POL-ENV-002", "3"),
    ("In what file standard should vendors report what their components are made of?", "POL-ENV-002", "4"),
    ("When the European chemicals agency adds hazardous substances twice a year, what do we do?", "POL-ENV-003", "1"),
    ("Do we have to inform customers and a waste database when a product contains a hazardous chemical above a tenth of a percent?", "POL-ENV-003", "2"),
    ("Are plastic softeners tracked for each component?", "POL-ENV-003", "3"),
    ("How do we make sure metals don't come from mines that fund armed groups?", "POL-SRC-004", "1"),
    ("What yearly paperwork do vendors of tin or cobalt components owe us?", "POL-SRC-004", "2"),
    ("What is the consequence for a vendor whose refinery fails the audit programme?", "POL-SRC-004", "3"),
    ("Does the industry ethics code also bind our vendors' vendors?", "POL-SR-005", "1"),
    ("What happens to vendors that score poorly on the social responsibility audit?", "POL-SR-005", "2"),
    ("What is our stance on workers paying to get hired?", "POL-SR-005", "3"),
    ("How frequently do we check business partners against government blacklists?", "POL-TRD-006", "1"),
    ("Which chips need a trade licence check before we ship them?", "POL-TRD-006", "2"),
    ("A vendor just got blacklisted by Washington. What do we do with pending orders?", "POL-TRD-006", "3"),
    ("What greenhouse gas information do we collect from vendors for European disclosure rules?", "POL-ESG-007", "2"),
    ("What must makers of rechargeable cells give us about their carbon impact?", "POL-ENV-008", "2"),
    ("When do we stop using styrofoam in boxes?", "POL-ENV-009", "1"),
    ("Which connected devices need a software ingredients list?", "POL-SEC-010", "1"),
    ("If someone asks us to delete their information, does that include the AI system, and how fast?", "POL-PRV-011", "1"),
    ("Can an automatically written assessment be acted on without a manager signing off?", "POL-GOV-012", "3"),
    ("What makes a new rule count as top priority?", "POL-GOV-012", "4"),
]


def build_golden(hard: bool = False) -> list[dict]:
    if hard:
        return [{"q": q, "doc_id": d, "section_prefix": s} for q, d, s in HARD]
    docs = json.loads((DATA_DIR / "generated" / "documents.json").read_text(encoding="utf-8"))
    golden = [{"q": q, "doc_id": d, "section_prefix": s} for q, d, s in GOLDEN]
    # Contract questions: 10 suppliers, two clause types each
    contracts = [d for d in docs if d["doc_type"] == "contract"][:10]
    for c in contracts:
        name = c["title"].replace("Supply agreement with ", "")
        golden.append({"q": f"How many days notice must {name} give before changing a material or sub-supplier?", "doc_id": c["doc_id"], "section_prefix": "6"})
        golden.append({"q": f"What are the remedies if {name} breaches compliance clauses?", "doc_id": c["doc_id"], "section_prefix": "7"})
    return golden


def evaluate(store: DocumentStore | None = None, k: int = 5) -> dict:
    store = store or DocumentStore()
    out = {"k": k, "embedding_model": settings.embedding_model, "reranker_model": settings.reranker_model,
           "direct": _run(store, build_golden(False), k), "paraphrased": _run(store, build_golden(True), k)}
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "retrieval_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _run(store: DocumentStore, golden: list[dict], k: int) -> list[dict]:
    configs = [("dense", dict(mode="dense", rerank=False)), ("sparse (BM25)", dict(mode="sparse", rerank=False)),
               ("hybrid (RRF)", dict(mode="hybrid", rerank=False)), ("hybrid + rerank", dict(mode="hybrid", rerank=True))]
    rows = []
    for name, kw in configs:
        hits1, hitsk, rr, lat = 0, 0, [], []
        for g in golden:
            t0 = time.perf_counter()
            res = store.search(g["q"], k=k, **kw)
            lat.append(time.perf_counter() - t0)
            rank = next((i for i, h in enumerate(res) if h.doc_id == g["doc_id"] and h.section.startswith(g["section_prefix"])), None)
            if rank is not None:
                hitsk += 1
                rr.append(1 / (rank + 1))
                hits1 += rank == 0
            else:
                rr.append(0.0)
        rows.append({"config": name, "n": len(golden), "hit@1": round(hits1 / len(golden), 3), f"hit@{k}": round(hitsk / len(golden), 3),
                     "mrr": round(statistics.mean(rr), 3), "p50_latency_ms": round(statistics.median(lat) * 1000, 1)})
    return rows


def to_markdown(rows: list[dict]) -> str:
    keys = list(rows[0].keys())
    return "\n".join(["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)] + ["| " + " | ".join(str(r[c]) for c in keys) + " |" for r in rows])


if __name__ == "__main__":
    out = evaluate()
    print("Direct questions (n=%d)" % out["direct"][0]["n"])
    print(to_markdown(out["direct"]))
    print()
    print("Paraphrased questions (n=%d)" % out["paraphrased"][0]["n"])
    print(to_markdown(out["paraphrased"]))
