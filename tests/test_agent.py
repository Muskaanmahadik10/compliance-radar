import json

import pytest
from sqlalchemy import select

from radar.config import DATA_DIR
from radar.data.schemas import MATERIAL_TOPIC, ImpactBrief
from radar.db.schema import audit_log, events
from radar.infra import get_db_engine, get_redis
from radar.ingest.feeds import RAW_STREAM, load_replay, publish
from radar.ingest.worker import CLASSIFIED_STREAM, Deduper, process_available


@pytest.fixture(scope="module")
def company():
    """Full offline stack: generated company in the DB, documents indexed, fixture events classified."""
    from radar.data.build import build_all
    from radar.retrieval.ingest import ingest

    build_all(reset_erasures=True)  # explicit demo reset: tombstones from other test modules are cleared
    ingest(recreate=True)
    r = get_redis()
    r.delete(RAW_STREAM, CLASSIFIED_STREAM)
    with get_db_engine().begin() as c:
        c.execute(events.delete())
    publish(load_replay())
    process_available(Deduper())
    with get_db_engine().connect() as c:
        return {row.topic: row.event_id for row in c.execute(select(events.c.topic, events.c.event_id)) if row.topic != "other"}


def test_pfas_event_produces_grounded_brief_and_pauses_for_review(company):
    from radar.agent import decide, get_brief, start_brief
    from radar.agent.service import graph

    b = start_brief(company["pfas"])
    assert b["status"] == "pending_review"
    brief = ImpactBrief(**{k: v for k, v in b["brief"].items() if not k.startswith("_")})
    assert brief.requires_approval and brief.topic.value == "pfas"
    # counts are the SQL truth, independent of the model
    sup = json.loads((DATA_DIR / "generated" / "suppliers.json").read_text(encoding="utf-8"))
    pfas_mats = {m for m, t in MATERIAL_TOPIC.items() if t.value == "pfas"}
    truth = sum(1 for s in sup if set(s["materials"]) & pfas_mats)
    assert brief.affected_counts["suppliers"] == truth > 0
    # every citation points at a real retrieved section; the PFAS policy is among them
    assert brief.citations and any(c.doc_id == "POL-ENV-001" for c in brief.citations)
    assert all(a.citations for a in brief.actions)
    assert b["tasks"] == []  # nothing issued before approval
    assert graph().get_state({"configurable": {"thread_id": b["brief_id"]}}).next == ("human_review",)

    done = decide(b["brief_id"], approve=True, by="compliance_manager", comment="ok")
    assert done["status"] == "approved" and done["decided_by"] == "compliance_manager"
    assert len(done["tasks"]) == len(brief.actions) and all(t["status"] == "open" for t in done["tasks"])
    with get_db_engine().connect() as c:
        acts = [r.action for r in c.execute(select(audit_log.c.action).where(audit_log.c.subject == b["brief_id"]))]
    assert "brief_approved" in acts
    assert get_brief(b["brief_id"])["status"] == "approved"
    with pytest.raises(ValueError):
        decide(b["brief_id"], approve=True, by="x")  # cannot decide twice


def test_incident_event_is_scoped_by_country_and_material(company):
    from radar.agent import decide, start_brief

    b = start_brief(company["supplier_incident"])
    counts = b["brief"]["affected_counts"]
    sup = json.loads((DATA_DIR / "generated" / "suppliers.json").read_text(encoding="utf-8"))
    truth = sum(1 for s in sup if s["country"] == "Vietnam" and "tantalum capacitor" in s["materials"])
    assert counts["suppliers"] == truth
    rej = decide(b["brief_id"], approve=False, by="compliance_manager", comment="not relevant")
    assert rej["status"] == "rejected" and rej["tasks"] == []


def test_verify_removes_invented_ids_and_citations(company):
    from radar.agent.graph import _RUNS, verify

    cfg = {"configurable": {"thread_id": "t-verify"}}
    state = {
        "evidence": [{"doc_id": "POL-ENV-001", "section": "3 Requirements", "citation": "POL-ENV-001 §3 Requirements", "text": "x", "doc_type": "policy"}],
        "suppliers": [{"supplier_id": "S0001"}], "parts": [{"part_id": "P00001"}],
        "draft": {"citations": [{"doc_id": "POL-ENV-001", "section": "3 Requirements", "quote": ""}, {"doc_id": "POL-FAKE-999", "section": "1 X", "quote": ""}],
                  "affected": [{"kind": "supplier", "id": "S0001", "name": "a", "why": "w", "citations": []}, {"kind": "supplier", "id": "S9999", "name": "ghost", "why": "w", "citations": []}],
                  "actions": [{"action": "Request declarations from suppliers", "owner_role": "Procurement", "deadline": "30 days", "citations": [{"doc_id": "POL-FAKE-999", "section": "1 X", "quote": ""}]}]},
    }
    out = verify(state, cfg)
    assert [c["doc_id"] for c in out["draft"]["citations"]] == ["POL-ENV-001"]
    assert [a["id"] for a in out["draft"]["affected"]] == ["S0001"]
    assert any("S9999" in f for f in out["auto_findings"]) and any("without a valid citation" in f for f in out["auto_findings"])
    _RUNS.pop("t-verify", None)


def test_verify_rejects_another_suppliers_contract_as_evidence(company):
    from radar.agent.graph import _RUNS, verify

    cfg = {"configurable": {"thread_id": "t-verify2"}}
    ev = [{"doc_id": "CTR-S0111", "section": "6 Change notification", "citation": "CTR-S0111 §6 Change notification", "text": "x", "doc_type": "contract"}]
    cite = {"doc_id": "CTR-S0111", "section": "6 Change notification", "quote": ""}
    state = {"evidence": ev, "suppliers": [{"supplier_id": "S0111"}, {"supplier_id": "S0135"}], "parts": [],
             "draft": {"citations": [dict(cite)], "actions": [{"action": "Obtain change notification", "owner_role": "Procurement", "deadline": "within 90 days", "citations": [dict(cite)]}],
                       "affected": [{"kind": "supplier", "id": "S0111", "name": "a", "why": "w", "citations": [dict(cite)]},
                                    {"kind": "supplier", "id": "S0135", "name": "b", "why": "w", "citations": [dict(cite)]},
                                    {"kind": "contract_clause", "id": "CTR-S0038", "name": "c", "why": "w", "citations": []}]}}
    out = verify(state, cfg)
    by_id = {a["id"]: a for a in out["draft"]["affected"]}
    assert by_id["S0111"]["citations"] and not by_id["S0135"]["citations"]
    assert any("another supplier's contract" in f for f in out["auto_findings"])
    assert any("CTR-S0038 has no citation" in f for f in out["auto_findings"])
    _RUNS.pop("t-verify2", None)
