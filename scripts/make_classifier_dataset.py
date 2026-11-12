"""Build a training set for the event classifier by distillation: a strong LLM writes realistic, varied headlines for
each topic, and a small encoder is then fine-tuned on them (colab/train_classifier.ipynb). The hand-written test set in
data/fixtures/classifier_test.jsonl is never shown to the generator, so the evaluation stays honest.

Usage: python scripts/make_classifier_dataset.py --per-topic 60     (needs a real provider in .env)
Output: data/classifier/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere without installing the package

from pydantic import BaseModel, Field

from radar.config import DATA_DIR
from radar.data.schemas import Topic
from radar.llm import CostLedger, call_structured, get_router

DESCRIPTIONS = {
    Topic.PFAS: "PFAS / fluorinated 'forever chemicals' restrictions, reporting rules, bans, litigation affecting products",
    Topic.ROHS: "RoHS-type hazardous substance limits in electrical/electronic equipment (lead, cadmium, mercury, exemptions) in EU, UK, China, Taiwan and others",
    Topic.REACH_SVHC: "EU REACH: Candidate List / SVHC additions, Authorisation List, Annex XVII restrictions, SCIP notifications, phthalates, bisphenols, flame retardants",
    Topic.CONFLICT_MINERALS: "responsible minerals: tin, tantalum, tungsten, gold, cobalt; smelter audits (RMAP), due diligence rules, CMRT reporting",
    Topic.RBA_LABOUR: "labour rights in electronics supply chains: forced labour laws (UFLPA, EU regulation), RBA code, audits, recruitment fees, working hours",
    Topic.EXPORT_CONTROL: "export controls and sanctions: entity lists, chip and equipment controls, dual-use rules, OFAC/BIS/EU/Japan/Taiwan measures",
    Topic.CSRD_ESG: "sustainability reporting and climate disclosure: CSRD, ESRS, Scope 3, ISSB/IFRS S1-S2, taxonomy, assurance",
    Topic.BATTERY: "battery regulation: EU Battery Regulation, battery passport, carbon footprint, recycled content, removability, due diligence for batteries",
    Topic.PACKAGING: "packaging and packaging waste: PPWR, EPR fees, recycled content, bans on formats or materials (EPS, PVC), labelling",
    Topic.CYBERSECURITY: "product and network cybersecurity regulation: Cyber Resilience Act, NIS2, SBOM, vulnerability reporting, IoT security standards",
    Topic.DATA_PROTECTION: "personal data protection: GDPR, Taiwan PDPA, fines, erasure rights, AI and personal data, cross-border transfers",
    Topic.SUPPLIER_INCIDENT: "supply-chain disruption at suppliers: fires, explosions, earthquakes, floods, strikes, insolvency, recalls, plant shutdowns",
    Topic.OTHER: "general business, technology, finance or local news that is NOT about compliance regulation or supplier disruption (earnings, product launches, markets, politics, sport)",
}


class Headlines(BaseModel):
    headlines: list[str] = Field(min_length=5)


PROMPT = """Write {n} different, realistic news or regulator headlines about: {desc}.
Vary the source style (regulator notice, trade press, newswire), the region, the wording and the length (8 to 22 words). Use everyday synonyms as often as official terms. Do not number them. Do not repeat a headline. {avoid}"""


def main(per_topic: int, batch: int = 30) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    router, ledger = get_router(), CostLedger()
    if router.strong._llm_type == "mock":
        raise SystemExit("Set RADAR_LLM_PROVIDER to a real provider to generate training data.")
    test_titles = {json.loads(ln)["title"].lower() for ln in (DATA_DIR / "fixtures" / "classifier_test.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()}
    out_dir = DATA_DIR / "classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for topic, desc in DESCRIPTIONS.items():
        seen: set[str] = set()
        while len(seen) < per_topic:
            avoid = f"Avoid these angles already covered: {'; '.join(list(seen)[-8:])}" if seen else ""
            try:
                res = call_structured(router.strong, Headlines, [("human", PROMPT.format(n=min(batch, per_topic - len(seen) + 5), desc=desc, avoid=avoid))], config={"callbacks": [ledger]})
            except Exception as e:
                print(f"{topic.value}: generation failed ({e}); keeping {len(seen)}")
                break
            before = len(seen)
            for h in res.headlines:
                h = " ".join(h.split())
                if 20 <= len(h) <= 220 and h.lower() not in test_titles:
                    seen.add(h)
            if len(seen) == before:
                break
        rows += [{"title": h, "topic": topic.value} for h in list(seen)[:per_topic]]
        print(f"{topic.value}: {min(len(seen), per_topic)} headlines")
    Path(out_dir / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out_dir / 'train.jsonl'} · cost ${ledger.cost_usd:.4f} · {ledger.calls} calls")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-topic", type=int, default=60)
    main(ap.parse_args().per_topic)
