"""The brief pipeline as a LangGraph state machine.

    triage ─► decompose ─► research ×N (parallel, Send API) ─► map_impact ─► draft ─► verify ─┬─► stage ─► human_review ─► publish
                                                                              ▲               │
                                                                              └── revise ◄────┘ (at most one loop)

Design points worth knowing:
- Counts and affected lists come from SQL (`map_impact`, paginated, so totals are exact), not from the model. The model
  describes and prioritises. The full supplier/part/product id lists are stored with the brief (`_impact`). They are
  candidate impact from keyword-matched materials and countries, not checked legal applicability.
- `verify` runs deterministic checks first, then an LLM critic. The deterministic checks are exactly these:
    * a cited (document, section) must be among the retrieved passages (section matched by its leading number);
    * supplier, part and product ids must be in the SQL impact result (otherwise removed), and their names are replaced
      with the names on record;
    * a policy/contract entry must be a document retrieved in this run and cite one of its own sections; otherwise it
      is removed from the affected items and listed under open questions as an unverified reference;
    * a supplier may only cite its own supply agreement;
    * every action needs at least one valid citation;
    * an explicit "N suppliers/parts/products" in the summary must equal the SQL total;
    * the critic must have been shown every action and affected item and the passage of every cited section.
  The structural ones (uncited action, no valid citation, wrong count in the summary, partial critic review) make
  automatic approval false whatever the critic says. Not checked deterministically: whether a quote is verbatim, whether
  a cited passage really supports the claim, its direction or its deadline. That is left to the LLM critic and the
  human reviewer.
- `human_review` uses LangGraph's `interrupt`: the run is checkpointed and resumes when a person decides.
- Every model call goes through `call_structured`, which masks e-mail/phone in the outbound messages and checks the
  brief's cost and time budget before every invocation, retry and fallback. Cost is estimated from reported tokens.
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
    impact_complete: bool
    draft: dict
    auto_findings: list[str]
    attempts: Annotated[list[dict], operator.add]
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
    started: float = field(default_factory=time.time)


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


def _budget_spent(config: RunnableConfig) -> str:
    """Why no further model call may start for this brief, or "" if one may. Checked by call_structured before every
    invocation, retry and fallback. Cost: RADAR_MAX_COST_PER_BRIEF_USD (estimated; calls with unknown prices count $0
    and are reported as `unpriced_calls`). Time: RADAR_MAX_SECONDS_PER_BRIEF. <= 0 disables either."""
    rc = run_context(config)
    cap = settings.max_cost_per_brief_usd
    if cap > 0 and rc.ledger.cost_usd >= cap:
        return f"cost budget of ${cap} per brief reached (estimated ${rc.ledger.cost_usd:.4f})"
    limit = settings.max_seconds_per_brief
    if limit > 0 and time.time() - rc.started >= limit:
        return f"time budget of {limit:.0f}s per brief reached"
    return ""


def _guard(config: RunnableConfig):
    return lambda: _budget_spent(config)


def _fmt_evidence(evidence: list[dict], limit: int = 12, max_chars: int = 480, first: set[str] | None = None) -> tuple[str, set[str]]:
    """Evidence block for a model prompt: up to `limit` unique passages, the citations in `first` before any others.
    Returns the text and the set of citations included. (E-mail/phone masking happens at the call boundary.)"""
    unique: dict[str, dict] = {}
    for h in evidence:
        unique.setdefault(h["citation"], h)
    order = [c for c in unique if c in (first or set())] + [c for c in unique if c not in (first or set())]
    shown = order[:limit]
    lines = [f"[{c}] ({unique[c].get('doc_type')}) {unique[c]['text'][:max_chars]}" + (" [...]" if len(unique[c]["text"]) > max_chars else "") for c in shown]
    return "\n".join(lines) or "(no passages retrieved)", set(shown)


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


MAX_RESEARCH_QUERIES = 5


def research_queries(model_questions: list[str], topic: str) -> list[str]:
    """Two deterministic queries always run, so a weak decomposition cannot starve retrieval. They are reserved first;
    the model's unique questions fill the remaining slots (at most MAX_RESEARCH_QUERIES in total)."""
    mandatory = [f"{topic} policy requirements and deadlines after a regulatory change", f"supplier contract notification and material declaration duties {topic}"]
    extra = [q for q in dict.fromkeys(q.strip() for q in model_questions) if q and q not in mandatory]
    return extra[: MAX_RESEARCH_QUERIES - len(mandatory)] + mandatory


def decompose(state: BriefState, config: RunnableConfig) -> dict:
    if _budget_spent(config):
        return {"sub_questions": research_queries([], state["event"]["topic"].replace("_", " "))}
    try:
        out = call_structured(get_router().for_task("decompose"), SubQuestions, [SystemMessage(SYSTEM), HumanMessage(DECOMPOSE.format(event=json.dumps(state["event"], ensure_ascii=False)))],
                              config={"callbacks": _callbacks(config)}, attempts=2, budget_spent=_guard(config))
        qs = out.sub_questions
    except Exception:
        qs = []
    return {"sub_questions": research_queries(qs, state["event"]["topic"].replace("_", " "))}


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


MAX_PAGES = 50  # safety bound on pagination; hitting it marks the impact incomplete instead of looping forever


def _all_pages(ctx, tool: str, args: dict, key: str, page: int) -> tuple[list[dict], int, bool]:
    """Every match of a paginated lookup. Returns (items, exact total, complete)."""
    items: list[dict] = []
    total = 0
    for _ in range(MAX_PAGES):
        r = call_tool(ctx, tool, args | {"limit": page, "offset": len(items)})
        batch, total = r.get(key, []), r.get("count", 0)
        items += batch
        if not batch or len(items) >= total:
            return items, total, len(items) >= total
    return items, total, False


def map_impact(state: BriefState, config: RunnableConfig) -> dict:
    """Candidate impact from SQL: every supplier and part matching the event's materials (and country, for
    location-scoped topics), paginated so the totals are exact rather than capped at one page. Materials and countries
    come from keyword matching (triage), so this is candidate impact for review, not legal applicability."""
    ctx = run_context(config).tools
    mats, countries = state["materials"], state.get("countries") or []
    sup: list[dict] = []
    complete = True
    if mats:
        for c in (countries or [None]):
            got, _, ok = _all_pages(ctx, "lookup_suppliers", {"materials": mats} | ({"country": c} if c else {}), "suppliers", 100)
            sup += got
            complete &= ok
    # Priority: tier 1 first, then risk flags, then lowest RBA audit score.
    sup.sort(key=lambda s: (s["tier"], -len(s["risk_flags"]), s["rba_audit_score"]))
    sup_ids = {s["supplier_id"] for s in sup}
    parts: list[dict] = []
    for m in mats:
        got, _, ok = _all_pages(ctx, "lookup_bom", {"material": m}, "parts", 200)
        parts += got
        complete &= ok
    parts = [p for p in parts if p["supplier_id"] in sup_ids] if countries else parts
    products = sorted({pp["product_id"] for p in parts for pp in p["products"]})
    # Contract clauses for the top suppliers that have a contract on file.
    extra: list[dict] = []
    for s in sup[:5]:
        extra += call_tool(ctx, "search_documents", {"query": "notification of regulatory change and material declarations", "k": 2, "supplier_id": s["supplier_id"]}).get("hits", [])
    counts = {"suppliers": len(sup), "parts": len(parts), "products": len(products), "tier1_suppliers": sum(1 for s in sup if s["tier"] == 1),
              "suppliers_with_risk_flags": sum(1 for s in sup if s["risk_flags"])}
    return {"suppliers": sup, "parts": parts, "counts": counts, "impact_complete": complete, "evidence": extra}


def draft(state: BriefState, config: RunnableConfig) -> dict:
    fb = ""
    if state.get("verdict") and not state["verdict"].get("approved", True):
        fb = "A reviewer rejected the previous draft. Fix these issues:\n- " + "\n- ".join(state["verdict"].get("missing_citations", []) + state.get("auto_findings", []) + [state["verdict"].get("notes", "")]) + "\n"
    top_sup = [{"supplier_id": s["supplier_id"], "name": s["name"], "country": s["country"], "tier": s["tier"], "materials": s["materials"], "rba_audit_score": s["rba_audit_score"], "risk_flags": s["risk_flags"]} for s in state["suppliers"][:8]]
    top_parts = [{"part_id": p["part_id"], "material": p["material"], "supplier_id": p["supplier_id"], "products": [pp["product_id"] for pp in p["products"]][:4]} for p in state["parts"][:8]]
    prompt = DRAFT.format(event=json.dumps(state["event"], ensure_ascii=False), counts=json.dumps(state["counts"]), suppliers=json.dumps(top_sup, ensure_ascii=False),
                          parts=json.dumps(top_parts, ensure_ascii=False), evidence=_fmt_evidence(state["evidence"])[0], feedback=fb)
    try:
        written: BriefDraft = call_structured(get_router().for_task("draft_brief"), BriefDraft, [SystemMessage(SYSTEM), HumanMessage(prompt)], config={"callbacks": _callbacks(config)},
                                              fallback=get_router().cheap, budget_spent=_guard(config))
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


_HYPHENS = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")  # models write "tier-1" with a non-breaking hyphen
_COUNT_CLAIM = re.compile(r"(?<![\w.-])(\d[\d,]*)\s+(affected\s+|impacted\s+|tier[- ]?1\s+|direct\s+)?(suppliers?|parts?|products?)\b", re.I)
_COUNT_KEY = {"supplier": "suppliers", "part": "parts", "product": "products"}
# Codes whose presence makes automatic approval false, whatever the critic says.
STRUCTURAL = {"uncited_action", "no_valid_citation", "count_mismatch", "partial_review", "impact_incomplete"}
CRITIC_DRAFT_CHARS = 6000  # with the evidence block this keeps a critic request inside an 8k-token provider limit
CRITIC_PASSAGES = 12


def critic_view(d: dict, counts: dict, evidence: list[dict]) -> tuple[str, str, list[str]]:
    """What the critic reviews: (draft JSON, evidence block, gaps). Every action and affected item is included with
    its citation ids; long free text is shortened before anything is dropped. Model-written quotes are left out: the
    critic compares claims with the retrieved passages themselves, cited passages first. `gaps` lists anything that
    still did not fit; a non-empty list means the review is partial."""
    def ids(cs: list[dict] | None) -> list[str]:
        return [f"{c['doc_id']} §{c['section']}" for c in cs or []]

    view = {"sql_totals": counts, "title": d.get("title"), "summary": d.get("summary"),
            "actions": [{"n": i + 1, "action": a["action"], "owner_role": a["owner_role"], "deadline": a["deadline"], "citations": ids(a.get("citations"))}
                        for i, a in enumerate(d.get("actions", []))],
            "affected": [{"kind": a["kind"], "id": a["id"], "name": a["name"], "why": a["why"], "citations": ids(a.get("citations"))} for a in d.get("affected", [])],
            "citations": ids(d.get("citations")), "open_questions": d.get("open_questions", [])}
    text = json.dumps(view, ensure_ascii=False)
    for cut in (200, 120, 60):  # shorten explanations, never an action (the obligations under review) and never drop an item
        if len(text) <= CRITIC_DRAFT_CHARS:
            break
        for a in view["affected"]:
            a["why"] = a["why"][:cut]
        view["open_questions"] = [q[:cut] for q in view["open_questions"]]
        text = json.dumps(view, ensure_ascii=False)
    gaps = []
    if len(text) > CRITIC_DRAFT_CHARS:
        gaps.append(f"draft too long for a complete critic review ({len(text)} > {CRITIC_DRAFT_CHARS} characters with every action in full); not every action was reviewed")
        text = text[:CRITIC_DRAFT_CHARS] + " ...[CUT: review incomplete]"
    cited = {c for a in [*d.get("actions", []), *d.get("affected", [])] for c in ids(a.get("citations"))} | set(ids(d.get("citations")))
    evidence_text, shown = _fmt_evidence(evidence, limit=CRITIC_PASSAGES, first=cited)
    missing = sorted(c for c in cited if c not in shown)
    if missing:
        gaps.append(f"{len(missing)} cited passage(s) did not fit in the critic's evidence: {', '.join(missing[:4])}")
    return text, evidence_text, gaps


def count_claims(summary: str, counts: dict) -> list[str]:
    """Explicit totals in the prose ("23 suppliers", "11 affected parts", "5 tier-1 suppliers") that differ from SQL.
    Phrases about a subset ("top 8 suppliers", "8 of the 23 suppliers", "the first 3 parts") and qualitative wording
    ("several suppliers") are not treated as totals."""
    wrong = []
    summary = (summary or "").translate(_HYPHENS)
    for m in _COUNT_CLAIM.finditer(summary):
        before = summary[max(0, m.start() - 12): m.start()].lower()
        after = summary[m.end(): m.end() + 4].lower()
        if any(w in before for w in ("top ", "first ", "highest", "priority", "listed")) or after.startswith(" of"):
            continue
        n = int(m.group(1).replace(",", ""))
        noun = m.group(3).lower().rstrip("s")
        key = "tier1_suppliers" if (m.group(2) or "").lower().startswith("tier") else _COUNT_KEY[noun]
        if key in counts and n != counts[key]:
            wrong.append(f"summary says {m.group(0).strip()!r} but the SQL total is {counts[key]}")
    return wrong


def verify(state: BriefState, config: RunnableConfig) -> dict:
    """Deterministic checks (see the module docstring for their exact scope), then an LLM critic. Invented ids and
    citations outside the evidence are removed, not just reported; every check result is recorded with a code."""
    d = json.loads(json.dumps(state["draft"]))  # deep copy: the stored draft of an earlier attempt is not mutated
    ev_cites = {(h["doc_id"], h["section"]) for h in state["evidence"]}
    ev_docs = {h["doc_id"] for h in state["evidence"]}
    sup_by_id = {s["supplier_id"]: s for s in state["suppliers"]}
    part_by_id = {p["part_id"]: p for p in state["parts"]}
    prod_by_id = {pp["product_id"]: pp for p in state["parts"] for pp in p.get("products", [])}
    checks: list[dict] = []

    def note(code: str, detail: str, removed: int = 0) -> None:
        checks.append({"code": code, "detail": detail, "removed": removed, "structural": code in STRUCTURAL})

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

    def keep_valid(cites: list[dict], where: str) -> list[dict]:
        good = [c for c in cites if ok_cite(c)]
        if len(good) < len(cites):
            bad = [c for c in cites if not ok_cite(c)]
            note("removed_citation", f"{len(bad)} citation(s) not found in the evidence were removed from {where}: " + ", ".join(f"{c['doc_id']} §{c['section']}" for c in bad[:4]), len(bad))
        return good

    for c in d.get("citations", []):
        norm(c)
    for item in [*d.get("affected", []), *d.get("actions", [])]:
        for c in item.get("citations", []):
            norm(c)
    d["citations"] = keep_valid(d.get("citations", []), "the brief")
    kept, renamed, unresolved = [], [], []
    for a in d.get("affected", []):
        record = {"supplier": sup_by_id, "part": part_by_id, "product": prod_by_id}.get(a["kind"])
        if record is not None:
            if a["id"] not in record:
                note(f"removed_{a['kind']}", f"{a['kind']} {a['id']} is not in the SQL impact result and was removed", 1)
                continue
            true_name = record[a["id"]].get("name") or record[a["id"]].get("description") or a["name"]
            if a["name"] != true_name:
                renamed.append(f"{a['id']} ({a['name'][:30]!r} -> {true_name!r})")
                a["name"] = true_name
        a["citations"] = keep_valid(a.get("citations", []), f"{a['kind']} {a['id']}")
        if a["kind"] == "supplier":
            # A supply agreement binds only its own supplier: CTR-S0111 cannot support a claim about S0135.
            foreign = [c for c in a["citations"] if c["doc_id"].startswith("CTR-") and c["doc_id"] != f"CTR-{a['id']}"]
            if foreign:
                note("foreign_contract", f"supplier {a['id']} cited another supplier's contract ({', '.join(c['doc_id'] for c in foreign)}); citation removed", len(foreign))
                a["citations"] = [c for c in a["citations"] if c not in foreign]
        if a["kind"] in ("contract_clause", "policy"):
            # The entry is about one document: it may only cite that document.
            other = [c for c in a["citations"] if c["doc_id"] != a["id"]]
            if other:
                note("policy_id_mismatch", f"{a['kind']} entry {a['id']} cited a different document ({', '.join(c['doc_id'] for c in other)}); citation removed", len(other))
                a["citations"] = [c for c in a["citations"] if c["doc_id"] == a["id"]]
            # Its identity must be a document this run actually retrieved, and it must cite one of that document's
            # sections. Otherwise it is not a verified affected item: it moves to the open questions for the reviewer.
            why = "is not among the retrieved documents" if a["id"] not in ev_docs else "" if a["citations"] else "cites none of its own sections"
            if why:
                note("removed_unverified_document", f"{a['kind']} entry {a['id']} {why}; moved to open questions", 1)
                unresolved.append(f"Unverified {a['kind'].replace('_', ' ')} reference {a['id']} ({a['name'][:60]}): {why}. Check manually.")
                continue
        kept.append(a)
    d["affected"] = kept
    if unresolved:
        d["open_questions"] = list(d.get("open_questions") or []) + unresolved
    if renamed:
        note("name_corrected", f"{len(renamed)} name(s) replaced with the names on record: " + ", ".join(renamed[:3]) + (" ..." if len(renamed) > 3 else ""))
    for act in d.get("actions", []):
        act["citations"] = keep_valid(act.get("citations", []), f"action '{act['action'][:40]}'")
        if not act["citations"]:
            note("uncited_action", f"action without a valid citation: {act['action'][:80]}")
    if not d.get("citations"):
        note("no_valid_citation", "brief has no valid citations")
    if state.get("impact_complete") is False:
        note("impact_incomplete", "the SQL impact lookup stopped at its page limit: the totals are lower bounds, not exact; review the full impact manually")
    for w in count_claims(d.get("summary", ""), state.get("counts") or {}):
        note("count_mismatch", w)
    if run_context(config).tools.injection_findings:
        note("injection_redacted", f"{len(run_context(config).tools.injection_findings)} instruction-like passage(s) were redacted from retrieved content")
    # The critic gets a complete, bounded view: every action and affected item with its citations and the SQL totals,
    # plus the passage of every cited section. If that does not fit, the review is recorded as partial, which blocks
    # automatic approval like any other structural failure.
    draft_json, evidence_text, gaps = critic_view(d, state.get("counts") or {}, state["evidence"])
    for g in gaps:
        note("partial_review", g)
    findings = [c["detail"] for c in checks]
    structural = [c["detail"] for c in checks if c["structural"]]
    verdict, last_err = None, ""
    try:
        v: CriticVerdict = call_structured(get_router().for_task("critic"), CriticVerdict,
                                           [SystemMessage(SYSTEM), HumanMessage(CRITIC.format(event=json.dumps({k: state["event"].get(k) for k in ("title", "summary", "jurisdiction", "topic", "published")}, ensure_ascii=False) if state.get("event") else "(not provided)",
                                                                                         draft=draft_json, evidence=evidence_text, auto_findings=findings or "none"))],
                                           config={"callbacks": _callbacks(config)}, attempts=2, fallback=get_router().cheap, budget_spent=_guard(config))
        verdict = v.model_dump()
        verdict["critic_model"] = getattr(v, "_model_used", "")
        verdict["critic_ran"] = True
        verdict["critic_approved"] = verdict["approved"]
    except Exception as e:
        last_err = str(e)[:160]
    if verdict is None:
        # The critic did not run. Record that plainly; the reviewer is told the draft had no second-model review.
        verdict = {"approved": True, "critic_approved": None, "missing_citations": [], "notes": f"Critic did not run ({last_err}). Automatic checks only.", "critic_ran": False}
    # Structural failures are not the critic's to wave through.
    verdict["structural_ok"] = not structural
    verdict["structural_failures"] = structural
    verdict["review_complete"] = not gaps  # every action and item in full, and every cited passage (as an excerpt)
    verdict["evidence_excerpt_chars"] = 480  # cited passages reach the critic as their first 480 characters
    verdict["approved"] = bool(verdict["approved"]) and not structural
    attempt = {"attempt": len(state.get("attempts") or []), "removed": sum(c["removed"] for c in checks), "codes": sorted({c["code"] for c in checks}),
               "structural_ok": not structural, "critic_ran": verdict["critic_ran"], "critic_approved": verdict.get("critic_approved"),
               "findings": [c["detail"][:200] for c in checks if c["code"] != "name_corrected"], "critic_notes": verdict.get("missing_citations", [])[:4]}
    return {"draft": d, "auto_findings": findings, "verdict": verdict, "attempts": [attempt], "revisions": state.get("revisions", 0) + (0 if verdict["approved"] else 1),
            "injection_findings": list(run_context(config).tools.injection_findings)}


def after_verify(state: BriefState, config: RunnableConfig) -> str:
    if not state["verdict"]["approved"] and state.get("revisions", 0) <= MAX_REVISIONS and not _budget_spent(config):
        return "draft"
    return "stage"


def stage(state: BriefState, config: RunnableConfig) -> dict:
    """Persist the draft as 'pending_review' before pausing, so the UI can show it. The resumable state itself lives in
    the graph checkpointer (see service._checkpointer for which one is in use)."""
    rc = run_context(config)
    cost = rc.ledger.summary()
    latency = round(time.time() - state["started"], 2)
    row = {"brief_id": state["brief_id"], "event_id": state["event"]["event_id"], "status": "pending_review",
           "brief": state["draft"] | {"_verdict": state["verdict"], "_auto_findings": state.get("auto_findings", []), "_attempts": state.get("attempts", []),
                                      "_injection_findings": state.get("injection_findings", []), "_models": get_router().names,
                                      "_drafted_by": rc.drafted_by or "fallback (no model)", "_n_evidence": len({h['citation'] for h in state['evidence']}),
                                      "_impact": {"supplier_ids": [x["supplier_id"] for x in state["suppliers"]], "part_ids": [x["part_id"] for x in state["parts"]],
                                                  "product_ids": sorted({pp["product_id"] for x in state["parts"] for pp in x["products"]}),
                                                  "materials": state.get("materials", []), "countries": state.get("countries", []),
                                                  "complete": state.get("impact_complete", True), "scope": "candidate impact from keyword-matched materials/countries; applicability not checked"},
                                      "_checkpointer": _checkpointer_kind()},
           "cost_usd": cost["cost_usd"], "latency_s": latency, "created_at": datetime.now(UTC).isoformat()}
    eng = get_db_engine()
    with eng.begin() as c:
        c.execute(briefs.delete().where(briefs.c.brief_id == state["brief_id"]))
        c.execute(insert(briefs), [row])
        c.execute(update(events_t).where(events_t.c.event_id == state["event"]["event_id"]).values(status="briefed"))
    return {"status": "pending_review", "cost": cost | {"latency_s": latency}}


def _checkpointer_kind() -> str:
    try:
        from radar.agent.service import checkpointer_kind

        return checkpointer_kind()
    except Exception:
        return "unknown"


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
