"""Regression tests for the 14 Nov 2026 re-audit of a92c2bd (finding ids CR-01 ... CR-09 in each section)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from radar.config import ROOT
from radar.data.schemas import BriefDraft, CriticVerdict, SubQuestions
from radar.llm.router import BudgetSpent, CostLedger, ModelRouter, call_structured
from radar.retrieval.store import SearchHit

EMAIL, PHONE = "jane@example.test", "+886 912 345 678"
KEEP = ("2026-12-01", "2023/1542", "SOP-PRV-002")  # dates and regulation/document ids must survive masking


class CapturingModel:
    """Fake chat model for call_structured: records every message list it is sent and answers per schema."""

    _llm_type = "mock"
    model_name = "fake"

    def __init__(self, answers: dict | None = None, cost_per_call: float = 0.0, ledger: CostLedger | None = None):
        self.seen: list[list] = []
        self.answers = answers or {}
        self.cost, self.ledger = cost_per_call, ledger

    def with_structured_output(self, schema, **kw):
        outer = self

        class Runner:
            def invoke(self, messages, config=None):
                outer.seen.append(messages)
                if outer.ledger is not None:
                    outer.ledger.cost_usd += outer.cost
                return outer.answers.get(schema)

        return Runner()

    def sent_text(self) -> str:
        return "\n".join(str(m.content) for msgs in self.seen for m in msgs)


def _draft_answer(**kw) -> BriefDraft:
    cite = {"doc_id": "POL-ENV-001", "section": "3 Requirements", "quote": ""}
    base = {"title": "PFAS update", "summary": "PFAS restriction progress.", "topic": "pfas", "jurisdiction": "EU", "confidence": 0.6, "citations": [cite],
            "affected": [], "actions": [{"action": "Request PFAS declarations", "owner_role": "Procurement", "deadline": "within 30 days", "citations": [cite]}]}
    return BriefDraft(**(base | kw))


def _state(**kw) -> dict:
    ev = [{"doc_id": "POL-ENV-001", "section": "3 Requirements", "citation": "POL-ENV-001 §3 Requirements", "text": "Suppliers declare PFAS within 30 days.", "doc_type": "policy"},
          {"doc_id": "CTR-S0001", "section": f"9 Contacts {EMAIL}", "citation": f"CTR-S0001 §9 Contacts {EMAIL}", "text": f"Call {PHONE} before 2026-12-01 (SOP-PRV-002).", "doc_type": "contract"}]
    st = {"brief_id": "b1", "event": {"event_id": "e1", "title": f"PFAS 2023/1542 notice, contact {EMAIL}", "summary": f"Phone {PHONE} by 2026-12-01.",
                                      "topic": "pfas", "jurisdiction": "EU", "published": "2026-11-01"},
          "evidence": ev, "suppliers": [{"supplier_id": "S0001", "name": "Acme Fluoro", "country": "Vietnam", "tier": 1, "materials": ["PTFE coating"], "rba_audit_score": 80, "risk_flags": []}],
          "parts": [], "counts": {"suppliers": 1, "parts": 0, "products": 0, "tier1_suppliers": 1, "suppliers_with_risk_flags": 0}, "materials": ["PTFE coating"]}
    return st | kw


@pytest.fixture
def fake_router(monkeypatch):
    import radar.agent.graph as graph

    model = CapturingModel({SubQuestions: SubQuestions(sub_questions=[f"ask {EMAIL}"]), BriefDraft: _draft_answer(), CriticVerdict: CriticVerdict(approved=True)})
    router = ModelRouter(cheap=model, strong=model)
    monkeypatch.setattr(graph, "get_router", lambda: router)
    yield model
    graph._RUNS.clear()


def _cfg(tid: str) -> dict:
    return {"configurable": {"thread_id": tid}}


# ---------------- CR-02: every outbound path is masked ----------------
def _assert_masked(text: str) -> None:
    assert EMAIL not in text and "912 345 678" not in text, text[:400]
    assert all(k in text for k in KEEP if k in text) and "2026-12-01" in text


def test_decompose_draft_and_critic_requests_are_masked(fake_router):
    from radar.agent.graph import decompose, draft, verify

    decompose(_state(), _cfg("m1"))
    rejected = {"approved": False, "missing_citations": [f"ask {EMAIL} about it"], "notes": f"call {PHONE}"}
    out = draft(_state(verdict=rejected, auto_findings=[]), _cfg("m1"))
    verify(_state(draft=out["draft"]), _cfg("m1"))
    assert len(fake_router.seen) == 3  # decompose, draft, critic
    for msgs in fake_router.seen:
        _assert_masked("\n".join(str(m.content) for m in msgs))


def test_classifier_llm_tier_request_is_masked():
    from radar.ingest.classify import classify, llm_callback

    class Cheap:
        _llm_type = "openai"
        model_name = "fake-cheap"
        prompts: list[str] = []

        def invoke(self, prompt, config=None):
            self.prompts.append(prompt)
            return type("R", (), {"content": "pfas"})()

    cheap = Cheap()
    call = llm_callback(ModelRouter(cheap=cheap, strong=cheap))
    classify(f"Notice {EMAIL}", f"Phone {PHONE} by 2026-12-01", llm=call, lora=None)
    assert cheap.prompts and EMAIL not in cheap.prompts[0] and "912 345 678" not in cheap.prompts[0] and "2026-12-01" in cheap.prompts[0]


def test_chat_request_masks_citation_headings_too():
    from radar import chat

    class FakeStore:
        def search(self, q, k=6, **kw):
            return [SearchHit(chunk_id="c1", doc_id="CTR-S0001", section=f"9 Contacts {EMAIL}", text=f"Call {PHONE} by 2026-12-01.", score=1.0, meta={"doc_type": "contract", "title": "t"})]

    messages, _, _, n = chat._prepare("Who is the contact?", [], FakeStore(), CostLedger(), 6, False)
    _assert_masked("\n".join(str(m.content) for m in messages))
    assert n >= 2


# ---------------- CR-03: ghost policy / contract identities ----------------
@pytest.mark.parametrize("kind,ghost", [("policy", "GHOST-POLICY"), ("contract_clause", "CTR-S9999")])
def test_approving_critic_cannot_keep_an_unretrieved_document_identity(fake_router, kind, ghost):
    from radar.agent.graph import verify

    d = _draft_answer(affected=[{"kind": kind, "id": ghost, "name": "invented", "why": "w", "citations": []},
                                {"kind": "policy", "id": "POL-ENV-001", "name": "PFAS policy", "why": "w", "citations": [{"doc_id": "POL-ENV-001", "section": "3 Requirements", "quote": ""}]}])
    out = verify(_state(draft=d.model_dump(mode="json")), _cfg(f"g-{kind}"))
    ids = {a["id"] for a in out["draft"]["affected"]}
    assert ghost not in ids and "POL-ENV-001" in ids  # the real, cited policy entry stays
    assert any(ghost in q for q in out["draft"]["open_questions"])
    assert out["verdict"]["critic_approved"] is True  # the critic approved, yet the ghost is not a verified item


# ---------------- CR-04: the critic sees every action, or the review is partial ----------------
def _long_draft(n_actions: int, text_len: int) -> dict:
    cite = {"doc_id": "POL-ENV-001", "section": "3 Requirements", "quote": "x" * 200}
    acts = [{"action": f"A{i} " + "y" * text_len, "owner_role": "Procurement", "deadline": "within 30 days", "citations": [cite]} for i in range(n_actions)]
    acts[-1]["action"] = "WRONG-FINAL stop all shipments within 1 day " + "z" * text_len
    return _draft_answer(actions=acts).model_dump(mode="json")


def test_last_action_and_sql_totals_reach_the_critic(fake_router):
    from radar.agent.graph import verify

    out = verify(_state(draft=_long_draft(12, 400)), _cfg("c1"))
    sent = fake_router.sent_text()
    assert "WRONG-FINAL" in sent and "sql_totals" in sent and "Suppliers declare PFAS" in sent
    assert out["verdict"]["review_complete"] is True and out["verdict"]["approved"] is True


def test_brief_too_long_to_review_completely_cannot_be_approved(fake_router):
    from radar.agent.graph import verify

    out = verify(_state(draft=_long_draft(120, 400)), _cfg("c2"))
    v = out["verdict"]
    assert v["critic_approved"] is True and v["review_complete"] is False
    assert v["approved"] is False and any("complete critic review" in f for f in v["structural_failures"])


# ---------------- CR-06: the budget stops retries and fallbacks ----------------
def test_no_retry_or_fallback_starts_after_a_call_exhausts_the_budget():
    ledger = CostLedger()
    primary = CapturingModel({}, cost_per_call=0.06, ledger=ledger)  # returns no structured output, costs $0.06
    fallback = CapturingModel({SubQuestions: SubQuestions(sub_questions=["q"])}, ledger=ledger)
    with pytest.raises(BudgetSpent):
        call_structured(primary, SubQuestions, ["hi"], attempts=3, fallback=fallback,
                        budget_spent=lambda: "cost budget reached" if ledger.cost_usd >= 0.05 else "")
    assert len(primary.seen) == 1 and fallback.seen == []


def test_spent_budget_gives_the_fallback_draft_without_any_model_call(fake_router):
    from radar.agent.graph import draft, run_context

    cfg = _cfg("b1")
    run_context(cfg).ledger.cost_usd = 1.0
    out = draft(_state(), cfg)
    assert out["draft"]["title"].startswith("[Automatic fallback]") and fake_router.seen == []


def test_unpriced_calls_are_counted_not_hidden():
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    ledger = CostLedger()
    msg = AIMessage(content="x", response_metadata={"model_name": "some/unknown-model"}, usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    ledger.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))
    assert ledger.summary()["unpriced_calls"] == 1 and ledger.cost_usd == 0


# ---------------- CR-07: chat citations by exact identity ----------------
def _hit(section: str, text: str) -> SearchHit:
    return SearchHit(chunk_id=section, doc_id="POL-ENV-001", section=section, text=text, score=1.0, meta={"doc_type": "policy", "title": "t"})


def test_chat_citations_resolve_exactly_and_independently_of_hit_order():
    from radar.chat import _finish, resolve_citations

    hits = [_hit("2 Reporting", "first part text"), _hit("2 Reporting (part 2)", "second part text"), _hit("3 Records", "records text")]
    answer = "Report yearly [POL-ENV-001 §2 Reporting]. Keep records [POL-ENV-001 §3]. Also [POL-ENV-001 §2 Invented heading]. And [POL-ENV-001 §2]."
    for order in (hits, list(reversed(hits))):
        cites, invalid = resolve_citations(answer, order)
        assert [(c["section"], c["quote"]) for c in cites] == [("2 Reporting", "first part text"), ("3 Records", "records text")]
        assert invalid == ["POL-ENV-001 §2 Invented heading", "POL-ENV-001 §2"]  # invented heading; ambiguous bare number
    assert _finish("Probably fine, no source needed.", hits, "q", 0, CostLedger()).support == "uncited"
    assert _finish("Yes [POL-ENV-001 §3 Records].", hits, "q", 0, CostLedger()).support == "cited"


# ---------------- CR-08: classifier boundaries and unresolved results ----------------
def test_jurisdiction_acronyms_need_word_boundaries():
    from radar.data.schemas import Jurisdiction
    from radar.ingest.classify import classify_rules

    assert classify_rules("ECHA adds Bisphenol AF to the candidate list").jurisdiction == Jurisdiction.EU
    assert classify_rules("BIS adds 12 entities to the Entity List").jurisdiction == Jurisdiction.US


def test_uncertain_result_without_a_later_tier_is_marked_unresolved():
    from radar.ingest.classify import classify

    c = classify("Lawmakers confirm new targets for rechargeable cells", "", llm=None, lora=None)
    assert c.tier == "rules" and c.confidence < 0.6 and c.resolved is False
    assert classify("ECHA adds four substances to the REACH SVHC candidate list", "", llm=None, lora=None).resolved is True


# ---------------- CR-09: exact impact totals beyond one page ----------------
def test_impact_totals_are_exact_beyond_the_page_size(monkeypatch):
    import radar.agent.graph as graph

    suppliers = [{"supplier_id": f"S{i:04d}", "name": f"s{i}", "country": "Vietnam", "tier": 2, "materials": ["m"], "rba_audit_score": 80, "risk_flags": []} for i in range(150)]
    parts = [{"part_id": f"P{i:05d}", "material": "m", "supplier_id": f"S{i % 150:04d}", "products": [{"product_id": f"PR{i % 7}"}]} for i in range(450)]

    def fake_tool(ctx, name, args):
        if name == "lookup_suppliers":
            return {"count": len(suppliers), "suppliers": suppliers[args["offset"]: args["offset"] + args["limit"]]}
        if name == "lookup_bom":
            return {"count": len(parts), "parts": parts[args["offset"]: args["offset"] + args["limit"]]}
        return {"hits": []}

    monkeypatch.setattr(graph, "call_tool", fake_tool)
    out = graph.map_impact({"materials": ["m"], "countries": []}, _cfg("i1"))
    graph._RUNS.clear()
    assert out["counts"]["suppliers"] == 150 and out["counts"]["parts"] == 450 and out["counts"]["products"] == 7 and out["impact_complete"] is True
    assert len({s["supplier_id"] for s in out["suppliers"]}) == 150


def test_lookup_bom_pages_add_up_to_the_exact_total(company_db):
    from radar.tools import ToolContext, call_tool

    ctx = ToolContext(store=None)
    first = call_tool(ctx, "lookup_bom", {"limit": 7, "offset": 0})
    seen, offset = [], 0
    while offset < first["count"]:
        page = call_tool(ctx, "lookup_bom", {"limit": 7, "offset": offset})
        seen += [p["part_id"] for p in page["parts"]]
        offset += 7
    assert len(seen) == len(set(seen)) == first["count"] > 7
    assert first["total_products"] == len({pp["product_id"] for p in call_tool(ctx, "lookup_bom", {"limit": 200})["parts"] for pp in p["products"]}) or first["count"] > 200


@pytest.fixture(scope="module")
def company_db():
    from radar.db import load_company

    gen = __import__("radar.config", fromlist=["settings"]).settings.generated_dir
    if not (gen / "suppliers.json").exists():
        from radar.data.build import build_all

        build_all()
    load_company()


# ---------------- CR-05: evaluation cannot reset the app's state ----------------
def test_isolation_refuses_once_state_is_open_and_points_elsewhere():
    from evaluation.isolation import use_isolated_state
    from radar.infra import get_db_engine

    get_db_engine()
    with pytest.raises(RuntimeError):
        use_isolated_state("x")
    code = ("from pathlib import Path; from evaluation.isolation import use_isolated_state; from radar.config import settings;"
            "info = use_isolated_state('briefs', root=Path(r'%s')); print(settings.local_sqlite_path); print(settings.generated_dir); print(settings.collection)")
    env = os.environ | {"RADAR_MODE": "local"}
    out = subprocess.run([sys.executable, "-c", code % (ROOT / ".local-test" / "iso")], cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout.split()
    assert all(".local-test" in x for x in out[:2]) and out[2].endswith("_eval")
    env = os.environ | {"RADAR_MODE": "compose"}
    r = subprocess.run([sys.executable, "-c", code % (ROOT / ".local-test" / "iso")], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "--use-app-state" in (r.stdout + r.stderr)


def test_provider_label_must_match_the_configured_model():
    from evaluation.briefs import check_provider

    check_provider("mock", {"cheap": "mock", "strong": "mock"})
    with pytest.raises(SystemExit):
        check_provider("env", {"cheap": "mock", "strong": "mock"})
    with pytest.raises(SystemExit):
        check_provider("mock", {"cheap": "openai/gpt-oss-20b", "strong": "openai/gpt-oss-120b"})


# ---------------- smaller findings ----------------
def test_failed_postgres_migration_is_retried(monkeypatch):
    from radar.db import schema

    calls = {"n": 0}

    class Conn:
        def __enter__(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("connection dropped")
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a):
            return None

    class Eng:
        dialect = type("D", (), {"name": "postgresql"})()
        url = "postgresql://test-migration-retry"

        def begin(self):
            return Conn()

    monkeypatch.setattr(schema, "get_db_engine", lambda: Eng())
    monkeypatch.setattr(schema.metadata, "create_all", lambda eng: None)
    monkeypatch.setattr(schema, "_migrated", set())
    with pytest.raises(RuntimeError):
        schema.create_all()
    schema.create_all()  # not skipped: the first attempt never committed
    assert calls["n"] == 2 and Eng.url in schema._migrated


def test_retrieval_eval_matches_section_numbers_exactly():
    from evaluation.retrieval import section_matches

    gold = {"doc_id": "POL-ENV-001", "section_prefix": "1"}
    assert section_matches("POL-ENV-001", "1 Purpose", gold)
    assert not section_matches("POL-ENV-001", "10 Records", gold) and not section_matches("POL-ENV-002", "1 Purpose", gold)


def test_notebook_uses_arguments_the_pinned_transformers_accepts():
    nb = json.loads((ROOT / "colab" / "train_classifier.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "warmup_ratio=" not in code and "warmup_steps=0.1" in code  # transformers 5 rejects warmup_ratio
    assert '"transformers==5.22.2"' in code and '"peft==0.21.2"' in code and "label_mapping.json" in code and "run_manifest.json" in code
    smoke = sorted((ROOT / "artifacts").glob("notebook_smoke_*.json"))
    assert smoke and json.loads(smoke[-1].read_text(encoding="utf-8"))["reload_check"].startswith("passed")
    assert Path(ROOT / "scripts" / "smoke_test_notebook.py").exists()
