"""Question answering over the company knowledge base: optional query rewriting, hybrid retrieval with reranking,
streamed answers with citations, a grounding check on the citations, and PII redaction before any external model call.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from radar.agent.prompts import CHAT, SYSTEM
from radar.llm import CostLedger, get_router
from radar.retrieval import DocumentStore
from radar.tools.registry import _INJ

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{8,}\d)(?!\w)")
_CITE = re.compile(r"\[([A-Z]+-[A-Z0-9-]+) §([^\]]+)\]")

REWRITE = """Rewrite the question as one short search query for a corporate compliance knowledge base. Replace everyday words with the standard terms a policy would use (for example PFAS, RoHS, REACH SVHC, CMRT, RMAP smelter, RBA audit, Entity List, ECCN, SBOM, material declaration). Reply with the query only.
Question: {q}"""


def redact_pii(text: str) -> tuple[str, int]:
    """Mask e-mail addresses and phone numbers before text leaves the process. Returns (text, number_redacted)."""
    n = len(_EMAIL.findall(text)) + len(_PHONE.findall(text))
    return _PHONE.sub("[phone]", _EMAIL.sub("[email]", text)), n


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


def _prepare(question: str, history: list[dict], store: DocumentStore, ledger: CostLedger, k: int, rewrite: bool):
    safe_q, n_pii = redact_pii(question)
    search_q = rewrite_query(safe_q, ledger) if rewrite else safe_q
    hits = store.search(f"{search_q} {safe_q}" if search_q != safe_q else safe_q, k=k)
    evidence = "\n".join(f"[{h.citation}] {_INJ.sub('[REDACTED instruction-like text]', h.text)}" for h in hits) or "(nothing retrieved)"
    hist = "\n".join(f"{m['role']}: {redact_pii(m['content'])[0]}" for m in history[-6:]) or "(none)"
    messages = [SystemMessage(SYSTEM), HumanMessage(CHAT.format(evidence=evidence, history=hist, question=safe_q))]
    return messages, hits, search_q, n_pii


def _finish(text: str, hits, search_q: str, n_pii: int, ledger: CostLedger) -> ChatResult:
    valid = {(h.doc_id, h.section.split(" ")[0]): h for h in hits}
    cites, invalid = [], []
    for doc_id, section in dict.fromkeys(_CITE.findall(text)):
        h = valid.get((doc_id, section.strip().split(" ")[0]))
        if h:
            cites.append({"doc_id": doc_id, "section": h.section, "title": h.meta.get("title"), "quote": h.text[:240]})
        else:
            invalid.append(f"{doc_id} §{section}")
    return ChatResult(text, cites, invalid, [h.to_dict() for h in hits], search_q, n_pii, ledger.summary())


def answer(question: str, history: list[dict] | None = None, store: DocumentStore | None = None, k: int = 6, rewrite: bool = True) -> ChatResult:
    store, ledger = store or DocumentStore(), CostLedger()
    messages, hits, search_q, n_pii = _prepare(question, history or [], store, ledger, k, rewrite)
    out = get_router().for_task("chat_answer").invoke(messages, config={"callbacks": [ledger]})
    return _finish(str(out.content), hits, search_q, n_pii, ledger)


def stream_answer(question: str, history: list[dict] | None = None, store: DocumentStore | None = None, k: int = 6, rewrite: bool = True) -> Iterator[dict]:
    """Yields {'type': 'sources'|'token'|'done', ...} events; the API forwards them as server-sent events."""
    store, ledger = store or DocumentStore(), CostLedger()
    messages, hits, search_q, n_pii = _prepare(question, history or [], store, ledger, k, rewrite)
    yield {"type": "sources", "search_query": search_q, "hits": [h.to_dict() for h in hits]}
    parts: list[str] = []
    for chunk in get_router().for_task("chat_answer").stream(messages, config={"callbacks": [ledger]}):
        tok = str(chunk.content)
        if tok:
            parts.append(tok)
            yield {"type": "token", "text": tok}
    res = _finish("".join(parts), hits, search_q, n_pii, ledger)
    yield {"type": "done", "citations": res.citations, "invalid_citations": res.invalid_citations, "pii_redacted": res.pii_redacted, "cost": res.cost}
