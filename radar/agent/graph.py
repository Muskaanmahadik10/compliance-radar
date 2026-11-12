"""The brief pipeline as a LangGraph state machine.

    triage ─► decompose ─► research ×N (parallel, Send API) ─► map_impact ─► draft ─► verify ─┬─► stage ─► human_review ─► publish
                                                                              ▲               │
                                                                              └── revise ◄────┘ (at most one loop)

Design points worth knowing:
- Counts and affected lists come from SQL (`map_impact`), not from the model. The model describes and prioritises.
- `verify` runs deterministic grounding checks first (do cited sections exist in the evidence? are named suppliers in the
  SQL result?) and only then an LLM critic. Invented ids are removed before a human sees the draft.
- `human_review` uses LangGraph's `interrupt`: the run is checkpointed and resumes when a person decides.
- Every model call goes through the router and a cost ledger; the cost is stored with the brief.
"""

from __future__ import annotations

import json
import operator
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from sqlalchemy import insert, update

from radar.agent.prompts import CRITIC, DECOMPOSE, DRAFT, SYSTEM
from radar.config import settings
from radar.data.schemas import MATERIAL_ALIASES, MATERIAL_TOPIC, BriefDraft, CriticVerdict, ImpactBrief, SubQuestions
from radar.db.schema import audit_log, briefs, tasks
from radar.db.schema import events as events_t
from radar.infra import get_db_engine
from radar.llm import CostLedger, call_structured, get_router
from radar.tools import ToolContext, call_tool

MAX_REVISIONS = 1
COUNTRY_WORDS = {"china": "China", "chinese": "China", "vietnam": "Vietnam", "vietnamese": "Vietnam", "taiwan": "Taiwan", "taiwanese": "Taiwan", "malaysia": "Malaysia",
                 "japan": "Japan", "japanese": "Japan", "korea": "South Korea", "thailand": "Thailand", "germany": "Germany", "mexico": "Mexico", "india": "India", "philippines": "Philippines"}
# Events about a specific place (incident, sanctions listing) are scoped to suppliers in that country.
COUNTRY_SCOPED_TOPICS = {"supplier_incident", "export_control"}


class BriefState(TypedDict, total=False):
    brief_id: str
    event: dict
    materials: list[str]
    countries: list[str]
    sub_questions: list[str]
    evidence: Annotated[list[dict], operator.add]
    suppliers: list[dict]
    parts: list[dict]
    counts: dict
    draft: dict
    auto_findings: list[str]
    verdict: dict
    revisions: int
    decision: dict
    status: str
    started: float
    cost: dict
    injection_findings: list[dict]


@dataclass
class RunContext:
    ledger: CostLedger = field(default_factory=CostLedger)
    tools: ToolContext = field(default_factory=ToolContext)
    drafted_by: str = ""


_RUNS: dict[str, RunContext] = {}


def run_context(config: RunnableConfig) -> RunContext:
    return _RUNS.setdefault(config["configurable"]["thread_id"], RunContext())


def _callbacks(config: RunnableConfig) -> list:
    cbs: list[Any] = [run_context(config).ledger]
    if settings.langfuse_enabled:  # optional tracing; never required for the pipeline to run
        try:
            from langfuse.langchain import CallbackHandler

            cbs.append(CallbackHandler())
        except Exception:
            pass
    return cbs


def _fmt_evidence(evidence: list[dict], limit: int = 12, max_chars: int = 480) -> str:
    seen, lines = set(), []
    for h in evidence:
        if h["citation"] in seen:
            continue
        seen.add(h["citation"])
        lines.append(f"[{h['citation']}] ({h.get('doc_type')}) {h['text'][:max_chars]}")
        if len(lines) >= limit:
            break
    return "\n".join(lines) or "(no passages retrieved)"


# ---------------- nodes ----------------
def triage(state: BriefState, config: RunnableConfig) -> dict:
    ev = state["event"]
    text = f"{ev['title']} {ev.get('summary', '')}".lower()
    mats = [m for m, t in MATERIAL_TOPIC.items() if t.value == ev["topic"]]
    mentioned = [m for m in MATERIAL_TOPIC if m.lower() in text or any(re.search(pat, text) for pat in MATERIAL_ALIASES.get(m, []))]
    # A named material narrows a broad topic ("tantalum capacitor plant fire"); otherwise use the topic's materials.
    materials = mentioned or mats
    countries = sorted({c for w, c in COUNTRY_WORDS.items() if re.search(rf"\b{w}\b", text)}) if ev["topic"] in COUNTRY_SCOPED_TOPICS else []
    # The topic's own policy is always evidence, whatever the decomposition later asks for.
    own = call_tool(run_context(config).tools, "search_documents", {"query": f"{ev['topic'].replace('_', ' ')} requirements response deadlines", "k": 4, "doc_type": "policy", "topics": [ev["topic"]]}).get("hits", [])
    return {"materials": materials, "countries": countries, "revisions": 0, "started": state.get("started") or time.time(), "evidence": own}


def decompose(state: BriefState, config: RunnableConfig) -> dict:
    try:
        out = call_structured(get_router().for_task("decompose"), SubQuestions, [SystemMessage(SYSTEM), HumanMessage(DECOMPOSE.format(event=json.dumps(state["event"], ensure_ascii=False)))],
                              config={"callbacks": _callbacks(config)}, attempts=2)
        qs = out.sub_questions
    except Exception:
        qs = []
    topic = state["event"]["topic"].replace("_", " ")
    # Always include two deterministic queries so a weak decomposition cannot starve retrieval.
    qs = list(dict.fromkeys([*qs, f"{topic} policy requirements and deadlines after a regulatory change", f"supplier contract notification and material declaration duties {topic}"]))[:5]
    return {"sub_questions": qs}


def fan_out(state: BriefState) -> list[Send]:
    return [Send("research", {"question": q, "event": state["event"], "brief_id": state["brief_id"]}) for q in state["sub_questions"]]


def research(payload: dict, config: RunnableConfig) -> dict:
    """One sub-question: policies/SOPs scoped to the topic first, then unscoped as a fallback, plus contract clauses."""
    ctx = run_context(config).tools
    q, topic = payload["question"], payload["event"]["topic"]
    hits: list[dict] = []
    r = call_tool(ctx, "search_documents", {"query": q, "k": 4, "topics": [topic, "other"]})
    hits += r.get("hits", [])
    if len(hits) < 2:
        hits += call_tool(ctx, "search_documents", {"query": q, "k": 4}).get("hits", [])
    hits += call_tool(ctx, "search_documents", {"query": q, "k": 2, "doc_type": "contract"}).get("hits", [])
    return {"evidence": hits}


def map_impact(state: BriefState, config: RunnableConfig) -> dict:
    ctx = run_context(config).tools
    mats, countries = state["materials"], state.get("countries") or []
    sup: list[dict] = []
    if mats:
        if countries:
            for c in countries:
                sup += call_tool(ctx, "lookup_suppliers", {"materials": mats, "country": c, "limit": 100})["suppliers"]
        else:
            sup = call_tool(ctx, "lookup_suppliers", {"materials": mats, "limit": 100})["suppliers"]
    # Priority: tier 1 first, then risk flags, then lowest RBA audit score.
    sup.sort(key=lambda s: (s["tier"], -len(s["risk_flags"]), s["rba_audit_score"]))
    sup_ids = {s["supplier_id"] for s in sup}
    parts: list[dict] = []
    for m in mats:
        parts += call_tool(ctx, "lookup_bom", {"material": m, "limit": 200})["parts"]
    parts = [p for p in parts if p["supplier_id"] in sup_ids] if countries else parts
    products = sorted({pp["product_id"] for p in parts for pp in p["products"]})
    # Contract clauses for the top suppliers that have a contract on file.
    extra: list[dict] = []
    for s in sup[:5]:
        extra += call_tool(ctx, "search_documents", {"query": "notification of regulatory change and material declarations", "k": 2, "supplier_id": s["supplier_id"]}).get("hits", [])
    counts = {"suppliers": len(sup), "parts": len(parts), "products": len(products), "tier1_suppliers": sum(1 for s in sup if s["tier"] == 1),
              "suppliers_with_risk_flags": sum(1 for s in sup if s["risk_flags"])}
    return {"suppliers": sup, "parts": parts, "counts": counts, "evidence": extra}


def draft(state: BriefState, config: RunnableConfig) -> dict:
    fb = ""
    if state.get("verdict") and not state["verdict"].get("approved", True):
        fb = "A reviewer rejected the previous draft. Fix these issues:\n- " + "\n- ".join(state["verdict"].get("missing_citations", []) + state.get("auto_findings", []) + [state["verdict"].get("notes", "")]) + "\n"
    top_sup = [{"supplier_id": s["supplier_id"], "name": s["name"], "country": s["country"], "tier": s["tier"], "materials": s["materials"], "rba_audit_score": s["rba_audit_score"], "risk_flags": s["risk_flags"]} for s in state["suppliers"][:8]]
    top_parts = [{"part_id": p["part_id"], "material": p["material"], "supplier_id": p["supplier_id"], "products": [pp["product_id"] for pp in p["products"]][:4]} for p in state["parts"][:8]]
    prompt = DRAFT.format(event=json.dumps(state["event"], ensure_ascii=False), counts=json.dumps(state["counts"]), suppliers=json.dumps(top_sup, ensure_ascii=False),
                          parts=json.dumps(top_parts, ensure_ascii=False), evidence=_fmt_evidence(state["evidence"]), feedback=fb)
    try:
        written: BriefDraft = call_structured(get_router().for_task("draft_brief"), BriefDraft, [SystemMessage(SYSTEM), HumanMessage(prompt)], config={"callbacks": _callbacks(config)},
                                              fallback=get_router().cheap)
        body = written.model_dump()
        run_context(config).drafted_by = getattr(written, "_model_used", "")
    except Exception as e:
        body = _fallback_draft(state, str(e))
    # The pipeline, not the model, sets the event id, the approval flag and the SQL-derived counts.
    brief = ImpactBrief(**body, event_id=state["event"]["event_id"], requires_approval=True, affected_counts=state["counts"])
    return {"draft": brief.model_dump(mode="json")}


def _fallback_draft(state: BriefState, error: str) -> dict:
    """If the model cannot produce a valid draft, the reviewer still gets the SQL impact and the retrieved evidence,
    clearly labelled as an automatic fallback. The pipeline never fails silently and never invents content."""
    ev = state["event"]
    seen, cites = set(), []
    for h in state["evidence"]:
        if h["citation"] not in seen and len(cites) < 6:
            seen.add(h["citation"])
            cites.append({"doc_id": h["doc_id"], "section": h["section"], "quote": h["text"][:280]})
    affected = [{"kind": "supplier", "id": s["supplier_id"], "name": s["name"], "why": f"supplies {', '.join(m for m in s['materials'] if m in state['materials']) or 'affected materials'}", "citations": []} for s in state["suppliers"][:8]]
    return {"title": f"[Automatic fallback] {ev['title'][:150]}", "topic": ev["topic"], "jurisdiction": ev["jurisdiction"],
            "summary": f"The drafting model did not return a valid brief ({error[:120]}). This fallback lists the impact from the supplier master and BOMs and the most relevant policy and contract passages for manual assessment.",
            "affected": affected, "actions": [{"action": "Assess this event manually using the listed evidence", "owner_role": "Compliance", "deadline": "not specified in policy", "citations": cites[:1]}],
            "confidence": 0.2, "open_questions": ["Automatic drafting failed; analyst review required."], "citations": cites or [{"doc_id": "POL-GOV-012", "section": "2 Impact assessment", "quote": ""}]}


def verify(state: BriefState, config: RunnableConfig) -> dict:
    """Deterministic grounding checks, then an LLM critic. Invented ids are removed, not just reported."""
    d = dict(state["draft"])
    ev_cites = {(h["doc_id"], h["section"]) for h in state["evidence"]}
    ev_docs = {h["doc_id"] for h in state["evidence"]}
    sup_ids = {s["supplier_id"] for s in state["suppliers"]}
    part_ids = {p["part_id"] for p in state["parts"]}
    findings: list[str] = []

    def ok_cite(c: dict) -> bool:
        return (c["doc_id"], c["section"]) in ev_cites or (c["doc_id"] in ev_docs and any(c["section"].split(" ")[0] == s.split(" ")[0] for dd, s in ev_cites if dd == c["doc_id"]))

    full_section = {}
    for dd, sec in ev_cites:
        full_section[(dd, sec.split(" ")[0])] = sec

    def norm(c: dict) -> dict:
        """'§4' -> '§4 Response to regulatory change' when the evidence has that section."""
        key = (c["doc_id"], c["section"].split(" ")[0])
        if (c["doc_id"], c["section"]) not in ev_cites and key in full_section:
            c["section"] = full_section[key]
        return c

    for c in d.get("citations", []):
        norm(c)
    for item in [*d.get("affected", []), *d.get("actions", [])]:
        for c in item.get("citations", []):
            norm(c)
    bad = [c for c in d.get("citations", []) if not ok_cite(c)]
    if bad:
        findings.append(f"{len(bad)} citation(s) not found in the evidence were removed: " + ", ".join(f"{c['doc_id']} §{c['section']}" for c in bad[:4]))
        d["citations"] = [c for c in d["citations"] if ok_cite(c)]
    kept = []
    for a in d.get("affected", []):
        if a["kind"] == "supplier" and a["id"] not in sup_ids:
            findings.append(f"supplier {a['id']} is not in the SQL impact result and was removed")
            continue
        if a["kind"] == "part" and a["id"] not in part_ids:
            findings.append(f"part {a['id']} is not in the SQL impact result and was removed")
            continue
        a["citations"] = [c for c in a.get("citations", []) if ok_cite(c)]
        if a["kind"] == "supplier":
            # A supply agreement binds only its own supplier: CTR-S0111 cannot support a claim about S0135.
            foreign = [c for c in a["citations"] if c["doc_id"].startswith("CTR-") and c["doc_id"] != f"CTR-{a['id']}"]
            if foreign:
                findings.append(f"supplier {a['id']} cited another supplier's contract ({', '.join(c['doc_id'] for c in foreign)}); citation removed")
                a["citations"] = [c for c in a["citations"] if c not in foreign]
        if a["kind"] in ("contract_clause", "policy") and not a["citations"]:
            findings.append(f"{a['kind']} entry {a['id']} has no citation to the section it refers to")
        kept.append(a)
    d["affected"] = kept
    for act in d.get("actions", []):
        act["citations"] = [c for c in act.get("citations", []) if ok_cite(c)]
        if not act["citations"]:
            findings.append(f"action without a valid citation: {act['action'][:80]}")
    if not d.get("citations"):
        findings.append("brief has no valid citations")
    if run_context(config).tools.injection_findings:
        findings.append(f"{len(run_context(config).tools.injection_findings)} instruction-like passage(s) were redacted from retrieved content")

    critic_input = {k: d.get(k) for k in ("title", "summary", "affected", "actions", "citations", "open_questions")}
    verdict, last_err = None, ""
    try:
        v: CriticVerdict = call_structured(get_router().for_task("critic"), CriticVerdict,
                                           [SystemMessage(SYSTEM), HumanMessage(CRITIC.format(event=json.dumps({k: state["event"].get(k) for k in ("title", "summary", "jurisdiction", "topic", "published")}, ensure_ascii=False) if state.get("event") else "(not provided)",
                                                                                         draft=json.dumps(critic_input, ensure_ascii=False)[:6000], evidence=_fmt_evidence(state["evidence"]), auto_findings=findings or "none"))],
                                           config={"callbacks": _callbacks(config)}, attempts=2, fallback=get_router().cheap)
        verdict = v.model_dump()
        verdict["critic_model"] = getattr(v, "_model_used", "")
    except Exception as e:
        last_err = str(e)[:160]
    if verdict is None:
        # The critic did not run. Record that plainly: automatic checks decide whether a revision is needed, and the
        # reviewer is told the draft had no second-model review.
        verdict = {"approved": not any("without a valid citation" in f or "no valid citations" in f for f in findings), "missing_citations": [],
                   "notes": f"Critic did not run ({last_err}). Automatic grounding checks only.", "critic_ran": False}
    else:
        verdict["critic_ran"] = True
    hard_fail = any("no valid citations" in f for f in findings)
    if hard_fail:
        verdict["approved"] = False
    return {"draft": d, "auto_findings": findings, "verdict": verdict, "revisions": state.get("revisions", 0) + (0 if verdict["approved"] else 1),
            "injection_findings": list(run_context(config).tools.injection_findings)}


def after_verify(state: BriefState) -> str:
    if not state["verdict"]["approved"] and state.get("revisions", 0) <= MAX_REVISIONS:
        return "draft"
    return "stage"


def stage(state: BriefState, config: RunnableConfig) -> dict:
    """Persist the draft as 'pending_review' before pausing, so the UI can show it and a restart loses nothing."""
    rc = run_context(config)
    cost = rc.ledger.summary()
    latency = round(time.time() - state["started"], 2)
    row = {"brief_id": state["brief_id"], "event_id": state["event"]["event_id"], "status": "pending_review",
           "brief": state["draft"] | {"_verdict": state["verdict"], "_auto_findings": state.get("auto_findings", []), "_injection_findings": state.get("injection_findings", []),
                                      "_models": get_router().names, "_drafted_by": rc.drafted_by or "fallback (no model)", "_n_evidence": len({h['citation'] for h in state['evidence']})},
           "cost_usd": cost["cost_usd"], "latency_s": latency, "created_at": datetime.now(UTC).isoformat()}
    eng = get_db_engine()
    with eng.begin() as c:
        c.execute(briefs.delete().where(briefs.c.brief_id == state["brief_id"]))
        c.execute(insert(briefs), [row])
        c.execute(update(events_t).where(events_t.c.event_id == state["event"]["event_id"]).values(status="briefed"))
    return {"status": "pending_review", "cost": cost | {"latency_s": latency}}


def human_review(state: BriefState, config: RunnableConfig) -> dict:
    decision = interrupt({"brief_id": state["brief_id"], "title": state["draft"]["title"], "verdict": state["verdict"], "question": "Approve this impact brief and issue its tasks?"})
    return {"decision": decision}


def publish(state: BriefState, config: RunnableConfig) -> dict:
    dec = state["decision"]
    approved = bool(dec.get("approve"))
    now = datetime.now(UTC).isoformat()
    eng = get_db_engine()
    with eng.begin() as c:
        c.execute(update(briefs).where(briefs.c.brief_id == state["brief_id"]).values(status="approved" if approved else "rejected", decided_by=dec.get("by", "unknown"), decided_at=now))
        if approved:
            c.execute(insert(tasks), [{"task_id": f"TASK-{uuid.uuid4().hex[:8]}", "brief_id": state["brief_id"], "action": a["action"], "owner_role": a["owner_role"], "deadline": a["deadline"], "status": "open"} for a in state["draft"].get("actions", [])] or [{"task_id": f"TASK-{uuid.uuid4().hex[:8]}", "brief_id": state["brief_id"], "action": "Review brief", "owner_role": "Compliance", "deadline": "", "status": "open"}])
        c.execute(insert(audit_log), [{"ts": now, "actor": dec.get("by", "unknown"), "action": "brief_approved" if approved else "brief_rejected", "subject": state["brief_id"], "detail": {"comment": dec.get("comment", "")}}])
    _RUNS.pop(config["configurable"]["thread_id"], None)
    return {"status": "approved" if approved else "rejected"}


def build_graph(checkpointer=None):
    g = StateGraph(BriefState)
    for name, fn in (("triage", triage), ("decompose", decompose), ("research", research), ("map_impact", map_impact), ("draft", draft),
                     ("verify", verify), ("stage", stage), ("human_review", human_review), ("publish", publish)):
        g.add_node(name, fn)
    g.add_edge(START, "triage")
    g.add_edge("triage", "decompose")
    g.add_conditional_edges("decompose", fan_out, ["research"])
    g.add_edge("research", "map_impact")
    g.add_edge("map_impact", "draft")
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", after_verify, {"draft": "draft", "stage": "stage"})
    g.add_edge("stage", "human_review")
    g.add_edge("human_review", "publish")
    g.add_edge("publish", END)
    return g.compile(checkpointer=checkpointer)
