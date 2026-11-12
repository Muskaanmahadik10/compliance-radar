"""Brief evaluation on the 11 relevant fixture events, against ground truth written by hand below (independent of the
pipeline's own triage code).

Metrics per brief:
  impact_exact        supplier count equals the hand-specified truth (materials x country) computed from the supplier master
  policy_cited        the policy that governs the topic is among the citations
  grounded_first_try  the verifier had to remove nothing (no invented ids, no citations outside the evidence)
  actions_cited       every action carries at least one valid citation after verification
  critic_approved     the LLM critic approved (possibly after one revision)
plus cost and latency. Usage: python -m evaluation.briefs [--provider mock|env]
"""

from __future__ import annotations

import argparse
import json
import statistics

from sqlalchemy import select

from radar.config import ARTIFACTS_DIR, DATA_DIR
from radar.db.schema import events
from radar.infra import get_db_engine, get_redis
from radar.ingest.feeds import RAW_STREAM, load_replay, publish
from radar.ingest.worker import CLASSIFIED_STREAM, Deduper, process_available

# url -> (materials that are truly affected, country scope or None, policy that should be cited)
TRUTH: dict[str, tuple[list[str], str | None, str]] = {
    "https://echa.europa.eu/-/pfas-restriction-proposal-progress": (["PTFE coating", "fluoropolymer gasket", "PFAS surfactant"], None, "POL-ENV-001"),
    "https://eur-lex.europa.eu/eli/dir_del/2026/0001/oj": (["lead solder"], None, "POL-ENV-002"),
    "https://echa.europa.eu/-/candidate-list-update-2026-11": (["brominated flame retardant"], None, "POL-ENV-003"),
    "https://www.federalregister.gov/d/2026-bis-entity-list": (["high-performance FPGA", "GaN RF amplifier", "encryption module"], "China", "POL-TRD-006"),
    "https://example-news.com/supplier-fire-vietnam": (["tantalum capacitor"], "Vietnam", "POL-GOV-012"),
    "https://eur-lex.europa.eu/eli/reg/2026/battery-passport-delegated": (["lithium-ion cell"], None, "POL-ENV-008"),
    "https://www.federalregister.gov/d/2026-epa-pfas-tsca": (["PTFE coating", "fluoropolymer gasket", "PFAS surfactant"], None, "POL-ENV-001"),
    "https://example-news.com/uflpa-electronics": (["aluminium housing"], None, "POL-SR-005"),
    "https://www.moea.gov.tw/news/2026/export-control-update": (["high-performance FPGA", "GaN RF amplifier"], None, "POL-TRD-006"),
    "https://eur-lex.europa.eu/eli/reg/2026/ppwr-recycled-content": (["EPS foam packaging", "PVC blister"], None, "POL-ENV-009"),
    "https://example-news.com/cra-guidance": ([], None, "POL-SEC-010"),
}


def _truth_suppliers(materials: list[str], country: str | None) -> int:
    sup = json.loads((DATA_DIR / "generated" / "suppliers.json").read_text(encoding="utf-8"))
    return sum(1 for s in sup if set(s["materials"]) & set(materials) and (country is None or s["country"] == country))


def prepare() -> list[dict]:
    """Fresh company, index and classified events so the run is reproducible."""
    from radar.data.build import build_all
    from radar.retrieval.ingest import ingest

    build_all()
    ingest(recreate=True)
    r = get_redis()
    r.delete(RAW_STREAM, CLASSIFIED_STREAM)
    with get_db_engine().begin() as c:
        c.execute(events.delete())
    publish(load_replay())
    process_available(Deduper())
    with get_db_engine().connect() as c:
        return [dict(x) for x in c.execute(select(events).where(events.c.status == "new")).mappings()]


def run(provider: str = "mock", limit: int | None = None) -> dict:
    from radar.agent import start_brief

    evs = [e for e in prepare() if e["url"] in TRUTH][: limit or None]
    rows = []
    for e in evs:
        mats, country, policy = TRUTH[e["url"]]
        b = start_brief(e["event_id"])
        br = b["brief"]
        removed = [f for f in br.get("_auto_findings", []) if "removed" in f]
        rows.append({
            "topic": e["topic"], "title": e["title"][:70],
            "impact_exact": br["affected_counts"]["suppliers"] == _truth_suppliers(mats, country),
            "suppliers_found": br["affected_counts"]["suppliers"], "suppliers_truth": _truth_suppliers(mats, country),
            "policy_cited": any(c["doc_id"] == policy for c in br["citations"]),
            "grounded_first_try": not removed,
            "actions_cited": bool(br["actions"]) and all(a["citations"] for a in br["actions"]),
            "critic_approved": bool(br["_verdict"].get("approved")),
            "fallback": br["title"].startswith("[Automatic fallback]"),
            "n_citations": len(br["citations"]), "cost_usd": b["cost_usd"], "latency_s": b["latency_s"],
        })
        print(json.dumps(rows[-1], ensure_ascii=False))
    n = len(rows)

    def rate(k: str) -> float:
        return round(sum(bool(r[k]) for r in rows) / n, 3)

    summary = {"n": n, "impact_exact": rate("impact_exact"), "policy_cited": rate("policy_cited"), "grounded_first_try": rate("grounded_first_try"),
               "actions_cited": rate("actions_cited"), "critic_approved": rate("critic_approved"), "fallback_rate": rate("fallback"),
               "mean_cost_usd": round(statistics.mean(r["cost_usd"] for r in rows), 5), "p50_latency_s": round(statistics.median(r["latency_s"] for r in rows), 1)}
    from radar.llm import get_router

    out = {"provider": provider, "models": get_router().names, "summary": summary, "rows": rows}
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / f"brief_eval_{provider}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "env"], help="label for the output file; the provider itself comes from RADAR_LLM_PROVIDER")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(run(a.provider, a.limit)["summary"], indent=2))
