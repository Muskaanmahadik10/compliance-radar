"""Regression tests for the independent audit of b91c1dc (14 Nov 2026): CP-01..CP-03, CP-08, CD-01..CD-05."""

from __future__ import annotations

import json

import pytest

from radar.data.schemas import BriefDraft
from radar.privacy import redact_pii
from tests import test_reaudit_fixes as _base
from tests.test_reaudit_fixes import CapturingModel, _cfg, _draft_answer, _state

fake_router = _base.fake_router  # shared fixture: a capturing fake model behind the router


# ---------------- CP-01: a shortened action is not a complete review ----------------
def test_one_very_long_action_reaches_the_critic_in_full_or_blocks_approval(fake_router):
    from radar.agent.graph import verify

    cite = {"doc_id": "POL-ENV-001", "section": "3 Requirements", "quote": ""}
    action = "Request PFAS declarations from all suppliers. " + "Details. " * 760 + "TAIL-MARKER: stop all shipments within 1 day."
    d = _draft_answer(actions=[{"action": action, "owner_role": "Procurement", "deadline": "within 30 days", "citations": [cite]}])
    assert len(d.actions[0].action) > 6000  # schema-valid
    out = verify(_state(draft=d.model_dump(mode="json")), _cfg("cp01"))
    v = out["verdict"]
    if "TAIL-MARKER" in fake_router.sent_text():
        assert v["review_complete"] is True
    else:
        assert v["review_complete"] is False and v["approved"] is False and v["critic_approved"] is True
    assert out["draft"]["actions"][0]["action"].endswith("within 1 day.")  # the stored brief keeps the full action


def test_medium_action_text_is_never_shortened_for_the_critic(fake_router):
    from radar.agent.graph import critic_view

    cite = {"doc_id": "POL-ENV-001", "section": "3 Requirements", "quote": ""}
    acts = [{"action": f"A{i} " + "x" * 600 + f" END{i}", "owner_role": "P", "deadline": "d", "citations": [cite]} for i in range(6)]
    d = _draft_answer(actions=acts).model_dump(mode="json")
    text, _, gaps = critic_view(d, {"suppliers": 1}, _state()["evidence"])
    assert all(f"END{i}" in text for i in range(6)) or gaps


# ---------------- CD-01: masking on compound and edge-case inputs ----------------
@pytest.mark.parametrize("text,masked,kept", [
    ("jane@example.test +886 912 345 678 2026-11-14 2023/1542", ["jane@", "912 345 678"], ["2026-11-14", "2023/1542"]),
    ("Call 123456789 on 2026-11-14", ["123456789"], ["2026-11-14"]),
    ("電話0912345678，截止 2026-12-01", ["0912345678"], ["2026-12-01"]),
    ("+886 912\n345 678 then REACH 1907/2006", ["912\n345"], ["1907/2006"]),
    ("Regulation 2023/1542, CAS 117-81-7, by 13/11/2026; 200 suppliers", [], ["2023/1542", "117-81-7", "13/11/2026", "200 suppliers"]),
])
def test_phone_masking_keeps_adjacent_dates_and_regulation_numbers(text, masked, kept):
    out, _ = redact_pii(text)
    assert all(m not in out for m in masked), out
    assert all(k in out for k in kept), out


# ---------------- CD-02: tuple shorthand goes through the same boundary ----------------
def test_tuple_messages_are_masked_at_the_call_boundary():
    from radar.data.schemas import SubQuestions
    from radar.llm.router import call_structured

    model = CapturingModel({SubQuestions: SubQuestions(sub_questions=["q"])})
    call_structured(model, SubQuestions, [("system", "s"), ("human", "jane@example.test +886 912 345 678 by 2026-12-01")])
    sent = json.dumps(model.seen[0], ensure_ascii=False)
    assert "jane@" not in sent and "912 345 678" not in sent and "2026-12-01" in sent


# ---------------- CD-03: MCP results are masked ----------------
def test_mcp_results_are_masked():
    from radar.mcp_server import _j

    out = _j({"hits": [{"section": "9 Contacts jane@example.test", "text": "Call +886 912 345 678 by 2026-12-01"}]})
    assert "jane@" not in out and "912 345 678" not in out and "2026-12-01" in out


# ---------------- CD-04: certificate wording matches its contents ----------------
def test_certificate_scope_note_does_not_claim_a_task_or_checkpoint_inventory():
    import inspect

    from radar import forget

    src = inspect.getsource(forget.forget_document)
    assert "(listed above)" not in src and "not inventoried" in src


# ---------------- CD-05: malformed citation ids are reported ----------------
def test_lowercase_or_malformed_citation_ids_are_reported_invalid():
    from radar.chat import _finish
    from radar.llm import CostLedger
    from radar.retrieval.store import SearchHit

    hits = [SearchHit(chunk_id="a", doc_id="POL-ENV-001", section="3 Records", text="records text", score=1.0, meta={})]
    res = _finish("Keep records [POL-ENV-001 §3 Records]. Also [pol-fake-001 §9 No source].", hits, "q", 0, CostLedger())
    assert res.invalid_citations == ["pol-fake-001 §9 No source"] and res.support == "partly_cited"


# ---------------- CP-02: feed timestamps sort chronologically ----------------
def test_rss_dates_are_normalised_so_newest_first_is_chronological():
    from radar.ingest.feeds import normalize_timestamp

    stamps = ["Mon, 02 Nov 2026 09:00:00 GMT", "Fri, 13 Nov 2026 09:00:00 GMT", "2026-11-10T09:00:00Z", "Thu, 12 Nov 2026 23:30:00 -0500"]
    norm = [normalize_timestamp(s)[0] for s in stamps]
    assert sorted(norm, reverse=True) == ["2026-11-13T09:00:00+00:00", "2026-11-13T04:30:00+00:00", "2026-11-10T09:00:00+00:00", "2026-11-02T09:00:00+00:00"]
    assert normalize_timestamp("not a date")[1] is False


def test_worker_stores_normalised_timestamps_and_triage_status():
    from sqlalchemy import select

    from radar.db import create_all
    from radar.db.schema import events
    from radar.infra import get_db_engine, get_redis
    from radar.ingest.feeds import RAW_STREAM, publish
    from radar.ingest.worker import CLASSIFIED_STREAM, Deduper, process_available

    create_all()
    get_redis().delete(RAW_STREAM, CLASSIFIED_STREAM)
    with get_db_engine().begin() as c:
        c.execute(events.delete())
    publish([{"event_id": "rss-old", "source": "t", "url": "https://x.test/old", "title": "ECHA adds PFAS substances to the REACH candidate list", "summary": "", "published": "Mon, 02 Nov 2026 09:00:00 GMT"},
             {"event_id": "rss-new", "source": "t", "url": "https://x.test/new", "title": "Commission amends RoHS exemption for lead in solder", "summary": "", "published": "Fri, 13 Nov 2026 09:00:00 GMT"},
             {"event_id": "vague", "source": "t", "url": "https://x.test/v", "title": "Regional retailer posts quarterly earnings", "summary": "", "published": "2026-11-12"}])
    process_available(Deduper(), llm=None)
    with get_db_engine().connect() as c:
        rows = c.execute(select(events.c.event_id, events.c.status, events.c.published).order_by(events.c.published.desc())).all()
    assert [r.event_id for r in rows] == ["rss-new", "vague", "rss-old"]
    assert {r.event_id: r.status for r in rows}["vague"] == "triage"


# ---------------- CP-03: an incomplete impact traversal blocks approval ----------------
def test_incomplete_impact_is_a_structural_failure(fake_router):
    from radar.agent.graph import verify

    out = verify(_state(draft=_draft_answer().model_dump(mode="json"), impact_complete=False), _cfg("cp03"))
    v = out["verdict"]
    assert v["critic_approved"] is True and v["approved"] is False and any("lower bounds" in f for f in v["structural_failures"])


def test_schema_still_accepts_long_actions():
    BriefDraft.model_validate(_draft_answer().model_dump() | {"actions": [{"action": "x" * 7000, "owner_role": "P", "deadline": "d", "citations": []}]})
