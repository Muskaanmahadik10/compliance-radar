"""Regression tests for the fixes made after the 13 Nov 2026 Codex audit of af9dd6b (finding ids in each docstring)."""

import json
from pathlib import Path

import pytest
from sqlalchemy import select

from radar.config import DATA_DIR, ROOT, settings
from radar.data.schemas import Topic
from radar.db.schema import documents, events, tasks
from radar.infra import get_db_engine, get_redis
from radar.ingest.feeds import RAW_STREAM, publish
from radar.ingest.worker import CLASSIFIED_STREAM, Deduper, process_available
from radar.privacy import redact_pii


@pytest.fixture(scope="module")
def company():
    from radar.data.build import build_all
    from radar.retrieval.ingest import ingest

    build_all(reset_erasures=True)
    ingest(recreate=True)


# ---------------- CLS-01: the LLM tier is wired into live ingestion ----------------
def test_live_ingestion_uses_the_llm_tier(monkeypatch, company):
    """process_available() with no arguments (as the worker, API and UI call it) must use the configured LLM tier."""
    import radar.ingest.worker as worker

    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "battery"

    fake_llm.model_name = "fake-cheap"
    monkeypatch.setattr(worker, "llm_callback", lambda: fake_llm)
    r = get_redis()
    r.delete(RAW_STREAM, CLASSIFIED_STREAM)
    with get_db_engine().begin() as c:
        c.execute(events.delete())
    vague = {"event_id": "cls01-vague", "source": "test", "url": "https://example.test/cls01", "title": "Lawmakers confirm new targets for rechargeable cells in consumer devices",
             "summary": "", "published": "2026-11-14T00:00:00+00:00"}
    publish([vague])
    stats = process_available(Deduper())
    assert stats["stored"] == 1 and len(calls) == 1
    with get_db_engine().connect() as c:
        row = c.execute(select(events).where(events.c.event_id == "cls01-vague")).mappings().one()
    assert row["topic"] == Topic.BATTERY.value and row["raw"]["classifier"] == "llm" and row["raw"]["classifier_llm"] == "fake-cheap"


def test_llm_callback_is_off_under_the_mock_and_uses_the_cheap_model_otherwise():
    from radar.ingest.classify import llm_callback
    from radar.llm import get_router

    assert llm_callback(get_router()) is None  # tests run on the mock: no fake "LLM" classification

    class Msg:
        content = "pfas"

    class Cheap:
        _llm_type = "fake"
        model_name = "cheap-model"

        def __init__(self):
            self.n = 0

        def invoke(self, prompt, config=None):
            self.n += 1
            return Msg()

    class Router:
        cheap = Cheap()

    cb = llm_callback(Router())
    assert cb("Classify this regulatory item") == "pfas" and Router.cheap.n == 1 and cb.model_name == "cheap-model"


# ---------------- RET-01/02: erased documents stay erased ----------------
def test_erased_document_is_not_restored_by_reingest_or_rebuild(company):
    from radar.data.build import build_all
    from radar.forget import forget_document
    from radar.retrieval import DocumentStore
    from radar.retrieval.ingest import ingest

    doc = "POL-ENV-008"
    cert = forget_document(doc, requested_by="dpo@company.example", reason="RET-01 regression")
    assert cert["verified"], cert["checks"]
    assert cert["checks"]["tombstone_blocks_reingestion"] and cert["checks"]["no_local_source_copy"]
    assert "documents.json" in cert["source_copies_removed"]
    docs_json = json.loads((DATA_DIR / "generated" / "documents.json").read_text(encoding="utf-8"))
    assert all(d["doc_id"] != doc for d in docs_json)
    # ordinary ingestion, then an ordinary rebuild (no reset): the document must not come back
    ingest()
    assert DocumentStore().count_document(doc) == 0
    build_all()  # regenerates the synthetic files from the seed, then keeps tombstoned documents out of them
    assert all(d["doc_id"] != doc for d in json.loads((DATA_DIR / "generated" / "documents.json").read_text(encoding="utf-8")))
    ingest()
    assert DocumentStore().count_document(doc) == 0
    with get_db_engine().connect() as c:
        assert c.execute(select(documents.c.status).where(documents.c.doc_id == doc)).scalar() == "deleted"
    assert not (DATA_DIR / "generated" / "docs" / f"{doc}.md").exists()
    # only the explicit demo reset restores it
    build_all(reset_erasures=True)
    ingest()
    assert DocumentStore().count_document(doc) > 0


# ---------------- RET-03 / RET-06: redaction ----------------
def test_retrieved_passages_are_redacted_before_the_model_call():
    from radar import chat
    from radar.llm import CostLedger
    from radar.retrieval.store import SearchHit

    class FakeStore:
        def search(self, q, k=6, **kw):
            return [SearchHit(chunk_id="c1", doc_id="CTR-S0001", section="9 Contacts", text="Escalate to ming.chen@supplier.example or +886 2 2737 6123 by 2026-11-30.",
                        score=1.0, meta={"doc_type": "contract", "title": "t"})]

    messages, hits, _, n = chat._prepare("Who do I call? I am amy@corp.example", [{"role": "user", "content": "my cell is 0912 345 678"}], FakeStore(), CostLedger(), 6, False)
    sent = "\n".join(str(m.content) for m in messages)
    assert "ming.chen" not in sent and "2737" not in sent and "amy@" not in sent and "0912" not in sent
    assert "2026-11-30" in sent and n == 4
    assert hits[0].text.startswith("Escalate to ming.chen")  # the original is kept for display


def test_brief_evidence_block_is_redacted():
    from radar.agent.graph import _fmt_evidence

    block = _fmt_evidence([{"citation": "CTR-S0001 §9", "doc_type": "contract", "text": "Contact ops@supplier.example, +886 912 345 678, before 2026-12-01."}])
    assert "ops@" not in block and "912" not in block and "2026-12-01" in block


def test_dates_and_identifiers_are_not_masked_as_phones():
    assert redact_pii("Is the deadline 2026-11-13 or 2026-12-01?") == ("Is the deadline 2026-11-13 or 2026-12-01?", 0)
    assert redact_pii("By 13/11/2026 under Regulation (EU) 2023/1542, CAS 117-81-7.")[1] == 0
    text, n = redact_pii("Call +886 912 345 678 before 2026-11-13")
    assert n == 1 and "2026-11-13" in text and "912" not in text


# ---------------- RET-04: signing key ----------------
def test_demo_signature_is_labelled_and_not_authentic(monkeypatch):
    from radar import forget

    body = {"doc_id": "X", "checks": {"a": True}}
    monkeypatch.setattr(settings, "signing_key", "")
    key, kid = forget._active_key()
    cert = dict(body, signature_key_id=kid)
    cert["signature"] = forget.sign(cert, key)
    assert kid == "demo-public" and forget.verify_certificate(cert) and not forget.certificate_authentic(cert)

    monkeypatch.setattr(settings, "signing_key", "a-private-test-key")
    key, kid = forget._active_key()
    real = dict(body, signature_key_id=kid)
    real["signature"] = forget.sign(real, key)
    assert kid.startswith("configured-") and forget.certificate_authentic(real)
    assert not forget.verify_certificate(dict(real, checks={"a": False}))  # tampering breaks it
    # a forger who only knows the public demo key cannot produce an authentic certificate
    forged = dict(body, signature_key_id=kid)
    forged["signature"] = forget.sign(forged, forget.DEMO_SIGNING_KEY)
    assert not forget.verify_certificate(forged) and not forget.certificate_authentic(forged)
    monkeypatch.setattr(settings, "signing_key", "another-key")
    assert not forget.verify_certificate(real)  # signed with a key this process does not hold


# ---------------- AP-15: long deadlines survive publication ----------------
def test_publish_keeps_a_long_prompted_deadline(company):
    from radar.agent.graph import publish

    long_deadline = "within 10 working days of publication"  # the prompt's own example, 37 characters
    longer = "within 30 calendar days of the supplier's written notification under clause 6.2 of the supply agreement"
    assert len(long_deadline) == 37
    state = {"brief_id": "BRF-ap15", "decision": {"approve": True, "by": "reviewer", "comment": ""},
             "draft": {"actions": [{"action": "Notify customers", "owner_role": "Compliance", "deadline": long_deadline},
                                   {"action": "Collect declarations", "owner_role": "Procurement and Supplier Quality Engineering", "deadline": longer}]}}
    publish(state, {"configurable": {"thread_id": "t-ap15"}})
    with get_db_engine().connect() as c:
        got = sorted(r.deadline for r in c.execute(select(tasks.c.deadline).where(tasks.c.brief_id == "BRF-ap15")))
    assert got == sorted([long_deadline, longer])


# ---------------- AP-05: mandatory research queries ----------------
@pytest.mark.parametrize("n_model", [0, 1, 4, 5])
def test_mandatory_research_queries_are_never_dropped(n_model):
    from radar.agent.graph import MAX_RESEARCH_QUERIES, research_queries

    model_qs = [f"q{i}" for i in range(n_model)] + ["q0"]  # with a duplicate
    qs = research_queries(model_qs, "pfas")
    assert len(qs) <= MAX_RESEARCH_QUERIES and len(set(qs)) == len(qs)
    assert "pfas policy requirements and deadlines after a regulatory change" in qs
    assert "supplier contract notification and material declaration duties pfas" in qs


# ---------------- OPS-01 / OPS-05 ----------------
def test_dockerignore_keeps_secrets_and_local_state_out():
    lines = {ln.strip() for ln in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()}
    assert {".env", ".env.*", ".venv", ".local", ".local-test", "data/generated", ".git"} <= lines and "!.env.example" in lines


def test_missing_key_fails_instead_of_becoming_the_mock(monkeypatch):
    from radar.llm import router

    monkeypatch.setattr(settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "llm_base_url", "https://api.groq.com/openai/v1")
    with pytest.raises(router.MissingCredentials):
        router._make("openai/gpt-oss-20b")
    monkeypatch.setattr(settings, "llm_base_url", "http://localhost:11434/v1")  # keyless local server is allowed
    assert router._make("llama3")._llm_type != "mock"
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    with pytest.raises(router.MissingCredentials):
        router._make("claude-haiku-5-5")
    monkeypatch.setattr(settings, "llm_provider", "mock")
    assert router._make("x")._llm_type == "mock"


# ---------------- AP-01 / AP-02 / AP-04 / AP-10: verifier ----------------
def _verify_state(**draft_overrides):
    ev = [{"doc_id": "POL-ENV-001", "section": "3 Requirements", "citation": "POL-ENV-001 §3 Requirements", "text": "Suppliers declare PFAS within 30 days.", "doc_type": "policy"},
          {"doc_id": "POL-GOV-012", "section": "2 Impact assessment", "citation": "POL-GOV-012 §2 Impact assessment", "text": "x", "doc_type": "policy"}]
    good = {"doc_id": "POL-ENV-001", "section": "3 Requirements", "quote": ""}
    draft = {"title": "t", "summary": "PFAS rules affect 23 suppliers and 11 parts.", "citations": [dict(good)],
             "affected": [{"kind": "supplier", "id": "S0001", "name": "Wrong Name Ltd", "why": "w", "citations": []}],
             "actions": [{"action": "Request declarations", "owner_role": "Procurement", "deadline": "within 30 days", "citations": [dict(good)]}],
             "open_questions": []}
    draft.update(draft_overrides)
    return {"event": {"title": "PFAS", "summary": "", "jurisdiction": "EU", "topic": "pfas", "published": "2026-11-01"}, "evidence": ev,
            "suppliers": [{"supplier_id": "S0001", "name": "Acme Fluoro"}],
            "parts": [{"part_id": "P00001", "description": "PTFE gasket", "products": [{"product_id": "PSU-1200", "name": "1200 W PSU"}]}],
            "counts": {"suppliers": 23, "parts": 11, "products": 4, "tier1_suppliers": 5}, "draft": draft}


def _run_verify(state, tid):
    from radar.agent.graph import _RUNS, verify

    try:
        return verify(state, {"configurable": {"thread_id": tid}})
    finally:
        _RUNS.pop(tid, None)


def test_verifier_checks_product_ids_names_and_policy_citations():
    """AP-01: invented product ids are removed, names come from the record, a policy entry cannot borrow another policy's citation."""
    cite_gov = {"doc_id": "POL-GOV-012", "section": "2 Impact assessment", "quote": ""}
    st = _verify_state(affected=[{"kind": "product", "id": "GHOST-PRODUCT", "name": "g", "why": "w", "citations": []},
                                 {"kind": "product", "id": "PSU-1200", "name": "made-up name", "why": "w", "citations": []},
                                 {"kind": "supplier", "id": "S0001", "name": "Wrong Name Ltd", "why": "w", "citations": []},
                                 {"kind": "policy", "id": "GHOST-POLICY", "name": "p", "why": "w", "citations": [cite_gov]}])
    out = _run_verify(st, "t-ap01")
    by_id = {a["id"]: a for a in out["draft"]["affected"]}
    assert "GHOST-PRODUCT" not in by_id
    assert by_id["PSU-1200"]["name"] == "1200 W PSU" and by_id["S0001"]["name"] == "Acme Fluoro"
    assert by_id["GHOST-POLICY"]["citations"] == []
    assert any("GHOST-PRODUCT" in f for f in out["auto_findings"]) and any("cited a different document" in f for f in out["auto_findings"])


def test_summary_totals_must_match_sql():
    """AP-02: a contradictory total fails; subsets and qualitative wording do not."""
    from radar.agent.graph import count_claims

    counts = {"suppliers": 23, "parts": 11, "products": 4, "tier1_suppliers": 5}
    assert count_claims("PFAS rules affect 23 suppliers, 11 parts and 4 products; 5 tier-1 suppliers.", counts) == []
    assert count_claims("The top 8 suppliers are listed; 8 of the 23 suppliers have contracts. Several suppliers...", counts) == []
    assert len(count_claims("999 suppliers are affected", counts)) == 1
    out = _run_verify(_verify_state(summary="999 suppliers are affected."), "t-ap02")
    assert out["verdict"]["approved"] is False and not out["verdict"]["structural_ok"]
    assert any("999 suppliers" in f for f in out["verdict"]["structural_failures"])


def test_critic_approval_cannot_override_an_uncited_action():
    """AP-04: the mock critic always approves; an action without a valid citation must still block automatic approval."""
    bad = {"doc_id": "POL-FAKE-1", "section": "1 X", "quote": ""}
    out = _run_verify(_verify_state(actions=[{"action": "Do something", "owner_role": "Ops", "deadline": "not specified in policy", "citations": [bad]}]), "t-ap04")
    v = out["verdict"]
    assert v["critic_ran"] and v["critic_approved"] is True
    assert v["approved"] is False and any("without a valid citation" in f for f in v["structural_failures"])


def test_first_attempt_records_nested_removals():
    """AP-10: removing an invalid citation inside an action counts as a first-try removal."""
    good = {"doc_id": "POL-ENV-001", "section": "3 Requirements", "quote": ""}
    fake = {"doc_id": "POL-FAKE-1", "section": "1 X", "quote": ""}
    out = _run_verify(_verify_state(actions=[{"action": "Request declarations", "owner_role": "Procurement", "deadline": "within 30 days", "citations": [dict(good), fake]}]), "t-ap10")
    att = out["attempts"][0]
    assert att["attempt"] == 0 and att["removed"] == 1 and "removed_citation" in att["codes"] and att["structural_ok"]


# ---------------- OPS-07: cost budget ----------------
def test_spent_budget_stops_model_calls(monkeypatch):
    from radar.agent.graph import _RUNS, draft, run_context

    cfg = {"configurable": {"thread_id": "t-ops07"}}
    monkeypatch.setattr(settings, "max_cost_per_brief_usd", 0.01)
    run_context(cfg).ledger.cost_usd = 0.02
    st = _verify_state()
    st |= {"materials": ["PTFE coating"], "draft": None}
    st["event"]["event_id"] = "E1"
    st["suppliers"][0] |= {"country": "TW", "tier": 1, "materials": ["PTFE coating"], "rba_audit_score": 150, "risk_flags": []}
    st["parts"][0] |= {"material": "PTFE coating", "supplier_id": "S0001"}
    try:
        out = draft(st, cfg)
        assert out["draft"]["title"].startswith("[Automatic fallback]") and "budget" in out["draft"]["summary"]
        assert run_context(cfg).ledger.calls == 0
    finally:
        _RUNS.pop("t-ops07", None)


def test_superseded_results_are_kept_with_an_explanation():
    folder = Path(ROOT / "artifacts" / "superseded_af9dd6b")
    assert (folder / "README.md").exists() and (folder / "brief_eval_env.json").exists()
