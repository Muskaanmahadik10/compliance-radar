"""Brief evaluation on the 11 relevant fixture events, against ground truth written by hand below (independent of the
pipeline's own triage code).

Metrics per brief (each is exactly this check, nothing more):
  impact_ids_exact    the brief's full supplier id set equals the hand-specified truth set (materials x country, from the
                      supplier master). Equal-size but different sets fail.
  policy_cited        the policy that governs the topic is among the citations (presence, not whether it supports a claim)
  first_draft_no_removals   on the FIRST draft the verifier removed nothing: no invented supplier/part/product id, no
                      citation outside the evidence, at any level. Only that: a first draft can pass this and still have
                      a wrong count, a corrected name or an uncited entry (see first_try_codes). Saved artifacts from
                      before 14 Nov call this `grounded_first_try`.
  structural_ok       the final brief passes the structural checks (every action cited, a valid citation, summary totals
                      equal to SQL)
  actions_cited       every action carries at least one valid citation after verification (presence, not entailment)
  critic_approved     the LLM critic's own verdict on the final draft (None if the critic did not run)
plus cost and latency. Deadlines, obligation direction and whether quotes are verbatim are not scored.
The summary reports how many truth events exist, how many the classifier passed on as relevant, and how many were run.
Usage: python -m evaluation.briefs [--provider mock|env] [--limit N] [--use-app-state]
  --limit takes the first N matching events. The run resets the state it uses (fresh company, cleared tombstones,
  re-index, cleared event streams), so by default it uses its own namespace (evaluation/isolation.py) and cannot undo
  the app's erasures; --use-app-state reuses and resets the app's state on purpose.
  --provider must match the configured model: env refuses the mock, mock refuses a real provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections import Counter
from datetime import UTC, datetime

from sqlalchemy import select

from radar.config import DATA_DIR, settings
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


def _truth_suppliers(materials: list[str], country: str | None) -> set[str]:
    sup = json.loads((settings.generated_dir / "suppliers.json").read_text(encoding="utf-8"))
    return {s["supplier_id"] for s in sup if set(s["materials"]) & set(materials) and (country is None or s["country"] == country)}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def _provenance() -> dict:
    """Source revision, and a hash of the uncommitted diff when the tree is dirty, so a dirty run is pinned too."""
    try:
        describe = _git("describe", "--always", "--dirty", "--abbrev=7").strip()
        out = {"commit": describe, "dirty": describe.endswith("-dirty")}
        if out["dirty"]:
            out["diff_sha256"] = hashlib.sha256(_git("diff", "HEAD", "--binary").encode("utf-8")).hexdigest()
        return out
    except Exception:
        return {"commit": "unknown", "dirty": None}


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_provider(provider: str, names: dict) -> None:
    """The output label must describe the model that actually ran."""
    is_mock = all(str(n).startswith("mock") for n in names.values())
    if provider == "env" and is_mock:
        raise SystemExit("--provider env, but the configured model is the mock; set RADAR_LLM_PROVIDER")
    if provider == "mock" and not is_mock:
        raise SystemExit(f"--provider mock, but real models are configured ({names}); use --provider env")


def prepare() -> list[dict]:
    """Fresh company (erasure tombstones cleared), index and classified events so the run is reproducible."""
    from radar.data.build import build_all
    from radar.retrieval.ingest import ingest

    build_all(reset_erasures=True)
    ingest(recreate=True)
    r = get_redis()
    r.delete(RAW_STREAM, CLASSIFIED_STREAM)
    with get_db_engine().begin() as c:
        c.execute(events.delete())
    publish(load_replay())
    process_available(Deduper())
    with get_db_engine().connect() as c:
        return [dict(x) for x in c.execute(select(events).where(events.c.status == "new")).mappings()]


def run(provider: str = "mock", limit: int | None = None, use_app_state: bool = False) -> dict:
    from evaluation.isolation import use_isolated_state
    from radar.agent import start_brief
    from radar.llm import get_router

    state = {"namespace": "app state (reset by this run)"} if use_app_state else use_isolated_state("briefs")
    check_provider(provider, get_router().names)
    relevant = [e for e in prepare() if e["url"] in TRUTH]
    evs = relevant[: limit or None]
    rows = []
    for e in evs:
        mats, country, policy = TRUTH[e["url"]]
        b = start_brief(e["event_id"])
        br = b["brief"]
        v = br["_verdict"]
        attempts = br.get("_attempts") or []
        truth_ids = _truth_suppliers(mats, country)
        found_ids = set((br.get("_impact") or {}).get("supplier_ids", []))
        rows.append({
            "topic": e["topic"], "title": e["title"][:70],
            "impact_ids_exact": found_ids == truth_ids,
            "suppliers_found": len(found_ids), "suppliers_truth": len(truth_ids),
            "policy_cited": any(c["doc_id"] == policy for c in br["citations"]),
            "first_draft_no_removals": bool(attempts) and attempts[0]["removed"] == 0,
            "first_try_removals": attempts[0]["removed"] if attempts else None, "first_try_codes": attempts[0]["codes"] if attempts else None,
            "n_attempts": len(attempts),
            "structural_ok": bool(v.get("structural_ok")),
            "actions_cited": bool(br["actions"]) and all(a["citations"] for a in br["actions"]),
            "critic_ran": v.get("critic_ran", False),
            "critic_approved": v.get("critic_approved") if v.get("critic_ran") else None,
            "critic_notes": v.get("missing_citations", [])[:3],
            "final_findings": br.get("_auto_findings", [])[:6],
            "fallback": br["title"].startswith("[Automatic fallback]"), "drafted_by": br.get("_drafted_by", ""),
            "n_citations": len(br["citations"]), "cost_usd": b["cost_usd"], "latency_s": b["latency_s"],
        })
        print(json.dumps(rows[-1], ensure_ascii=False))
    n = len(rows)

    def rate(k: str) -> float | None:
        vals = [r[k] for r in rows if r[k] is not None]
        return round(sum(bool(x) for x in vals) / len(vals), 3) if vals else None

    summary = {"truth_events": len(TRUTH), "classified_relevant": len(relevant), "evaluated": n,
               "classified_by": dict(Counter((e.get("raw") or {}).get("classifier", "?") for e in relevant)),
               "sample": f"first {limit} matching events (not random)" if limit else "all matching events",
               "impact_ids_exact": rate("impact_ids_exact"), "policy_cited": rate("policy_cited"), "first_draft_no_removals": rate("first_draft_no_removals"),
               "structural_ok": rate("structural_ok"), "actions_cited": rate("actions_cited"), "critic_ran": rate("critic_ran"), "critic_approved": rate("critic_approved"),
               "fallback_rate": rate("fallback"),
               "mean_cost_usd_estimated": round(statistics.mean(r["cost_usd"] for r in rows), 5), "p50_latency_s": round(statistics.median(r["latency_s"] for r in rows), 1)}
    out = {"provider": provider, "models": get_router().names, **_provenance(), "state": state,
           "inputs_sha256": {"events_replay.jsonl": _sha256(DATA_DIR / "fixtures" / "events_replay.jsonl")},
           "run_at": datetime.now(UTC).isoformat(timespec="seconds"), "summary": summary, "rows": rows}
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    # A partial run is saved separately so it can never overwrite the full evaluation.
    name = f"brief_eval_{provider}" + (f"_sample{n}" if limit else "")
    (settings.artifacts_dir / f"{name}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "env"], help="must match RADAR_LLM_PROVIDER (checked); also names the output file")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--use-app-state", action="store_true", help="run on (and reset) the app's own state instead of an isolated namespace")
    a = ap.parse_args()
    print(json.dumps(run(a.provider, a.limit, a.use_app_state)["summary"], indent=2))
