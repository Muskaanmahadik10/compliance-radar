"""Question answering over the company knowledge base: optional query rewriting, hybrid retrieval with reranking,
streamed answers with citations, a citation check, and e-mail/phone masking (radar/privacy.py) applied to the final
outbound messages (question, history, passages and their headings).

Citation check: a citation must name a retrieved passage by its exact document id and full section heading
(whitespace/case-insensitive). A bare section number ("§2") is accepted only when exactly one retrieved heading of that
document starts with it. Anything else is listed as invalid. The quote shown for a citation is the retrieved source
text, never model-written text. An answer with no valid citation is marked `support="uncited"`, so the UI can show it
as unsupported. Whether a cited passage actually says what the answer claims is not checked.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from radar.agent.prompts import CHAT, SYSTEM
from radar.llm import CostLedger, get_router
from radar.privacy import redact_messages, redact_pii
from radar.retrieval import DocumentStore
from radar.tools.registry import _INJ

_CITE = re.compile(r"\[([A-Z]+-[A-Z0-9-]+) §([^\]]+)\]")

REWRITE = """Rewrite the question as one short search query for a corporate compliance knowledge base. Replace everyday words with the standard terms a policy would use (for example PFAS, RoHS, REACH SVHC, CMRT, RMAP smelter, RBA audit, Entity List, ECCN, SBOM, material declaration). Reply with the query only.
Question: {q}"""


def _answer_model():
    """Strong model for answers, falling back to the cheap one if the provider rejects the call (rate limit, outage)."""
    r = get_router()
    return r.strong if r.strong._llm_type == "mock" else r.strong.with_fallbacks([r.cheap])


def rewrite_query(question: str, ledger: CostLedger | None = None) -> str:
    """Cheap-model rewrite into policy vocabulary. Falls back to the original question on any failure."""
    llm = get_router().for_task("rewrite")
    if llm._llm_type == "mock":
        return question
    try:
        out = llm.invoke([HumanMessage(REWRITE.format(q=question))], config={"callbacks": [ledger] if ledger else []})
        q = str(out.content).strip().strip('"').splitlines()[0]
        return q if 3 <= len(q) <= 300 else question
    except Exception:
        return question


@dataclass
class ChatResult:
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    hits: list[dict] = field(default_factory=list)
    search_query: str = ""
    pii_redacted: int = 0
    cost: dict = field(default_factory=dict)
    support: str = "uncited"  # cited | partly_cited (some citations invalid) | uncited (no valid citation)


def _prepare(question: str, history: list[dict], store: DocumentStore, ledger: CostLedger, k: int, rewrite: bool):
    safe_q, n_pii = redact_pii(question)
    search_q = rewrite_query(safe_q, ledger) if rewrite else safe_q
    hits = store.search(f"{search_q} {safe_q}" if search_q != safe_q else safe_q, k=k)
    lines = []
    for h in hits:  # retrieved passages are provider-bound too, so they get the same masking as the question
        body, n = redact_pii(_INJ.sub("[REDACTED instruction-like text]", h.text))
        n_pii += n
        lines.append(f"[{h.citation}] {body}")
    evidence = "\n".join(lines) or "(nothing retrieved)"
    hist_lines = []
    for m in history[-6:]:
        body, n = redact_pii(str(m.get("content", "")))
        n_pii += n
        hist_lines.append(f"{m.get('role', 'user')}: {body}")
    hist = "\n".join(hist_lines) or "(none)"
    # Final pass over the complete outbound messages: also covers section headings and anything formatted above.
    messages, n = redact_messages([SystemMessage(SYSTEM), HumanMessage(CHAT.format(evidence=evidence, history=hist, question=safe_q))])
    return messages, hits, search_q, n_pii + n


def normalise(text: str) -> str:
    """Models vary their punctuation: map lenticular brackets and non-breaking hyphens so citations parse uniformly."""
    return text.replace("【", "[").replace("】", "]").replace("‑", "-").replace("‐", "-")


def _heading(section: str) -> str:
    return " ".join(normalise(section).split()).lower()


def resolve_citations(text: str, hits) -> tuple[list[dict], list[str]]:
    """Map [DOC §section] references to retrieved passages by exact identity. Deterministic whatever the hit order."""
    exact: dict[tuple[str, str], list] = defaultdict(list)
    by_number: dict[tuple[str, str], set[str]] = defaultdict(set)
    for h in hits:
        exact[(h.doc_id, _heading(h.section))].append(h)
        by_number[(h.doc_id, h.section.split(" ")[0])].add(_heading(h.section))
    cites, invalid = [], []
    for doc_id, section in dict.fromkeys(_CITE.findall(text)):
        key = (doc_id, _heading(section))
        if key not in exact and " " not in section.strip():  # bare "§2": only if exactly one heading of the doc starts with it
            only = by_number.get((doc_id, section.strip()), set())
            key = (doc_id, next(iter(only))) if len(only) == 1 else key
        found = exact.get(key)
        if not found:
            invalid.append(f"{doc_id} §{section}")
            continue
        h = min(found, key=lambda x: x.text)  # identical headings (rare): pick deterministically, not by rank
        cites.append({"doc_id": doc_id, "section": h.section, "title": h.meta.get("title"), "quote": h.text[:240], "quote_source": "retrieved passage"})
    return cites, invalid


def _finish(text: str, hits, search_q: str, n_pii: int, ledger: CostLedger) -> ChatResult:
    text = normalise(text)
    cites, invalid = resolve_citations(text, hits)
    support = "uncited" if not cites else "partly_cited" if invalid else "cited"
    return ChatResult(text, cites, invalid, [h.to_dict() for h in hits], search_q, n_pii, ledger.summary(), support)


def answer(question: str, history: list[dict] | None = None, store: DocumentStore | None = None, k: int = 6, rewrite: bool = True) -> ChatResult:
    store, ledger = store or DocumentStore(), CostLedger()
    messages, hits, search_q, n_pii = _prepare(question, history or [], store, ledger, k, rewrite)
    out = _answer_model().invoke(messages, config={"callbacks": [ledger]})
    return _finish(str(out.content), hits, search_q, n_pii, ledger)


def stream_answer(question: str, history: list[dict] | None = None, store: DocumentStore | None = None, k: int = 6, rewrite: bool = True) -> Iterator[dict]:
    """Yields {'type': 'sources'|'token'|'done', ...} events; the API forwards them as server-sent events."""
    store, ledger = store or DocumentStore(), CostLedger()
    messages, hits, search_q, n_pii = _prepare(question, history or [], store, ledger, k, rewrite)
    yield {"type": "sources", "search_query": search_q, "hits": [h.to_dict() for h in hits]}
    parts: list[str] = []
    for chunk in _answer_model().stream(messages, config={"callbacks": [ledger]}):
        tok = str(chunk.content)
        if tok:
            parts.append(tok)
            yield {"type": "token", "text": tok}
    res = _finish("".join(parts), hits, search_q, n_pii, ledger)
    yield {"type": "done", "answer": res.answer, "citations": res.citations, "invalid_citations": res.invalid_citations, "support": res.support, "pii_redacted": res.pii_redacted, "cost": res.cost}
