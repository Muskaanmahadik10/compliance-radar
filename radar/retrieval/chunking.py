"""Chunking. Documents are markdown with '## section' headings; one chunk per section keeps citations precise
(doc id + section). Long sections are split on sentence boundaries with overlap."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

MAX_CHARS = 1200
OVERLAP_SENTENCES = 1
_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section: str
    text: str
    meta: dict

    @property
    def citation(self) -> str:
        return f"{self.doc_id} §{self.section}"


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    sents = _SENT.split(text)
    out, buf = [], []
    for s in sents:
        if sum(len(x) for x in buf) + len(s) > MAX_CHARS and buf:
            out.append(" ".join(buf))
            buf = buf[-OVERLAP_SENTENCES:]
        buf.append(s)
    if buf:
        out.append(" ".join(buf))
    return out


def chunk_document(doc: dict) -> list[Chunk]:
    """doc: {doc_id, doc_type, title, topics, jurisdiction, supplier_id, effective_date, sections:[(heading, text)]}"""
    chunks = []
    for heading, text in doc["sections"]:
        for i, piece in enumerate(_split_long(text)):
            section = heading if i == 0 else f"{heading} (part {i + 1})"
            cid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc['doc_id']}::{section}"))
            meta = {k: doc.get(k) for k in ("doc_type", "title", "topics", "jurisdiction", "supplier_id", "effective_date")}
            meta["language"] = doc.get("language", "en")
            chunks.append(Chunk(cid, doc["doc_id"], section, f"{doc['title']}. {heading}. {piece}", meta))
    return chunks


def chunk_markdown(doc_id: str, markdown: str, **meta) -> list[Chunk]:
    """For uploaded markdown files: first '# ' line is the title, '## ' lines are sections."""
    lines = markdown.splitlines()
    title = next((ln.lstrip("# ").strip() for ln in lines if ln.startswith("# ")), doc_id)
    sections, cur, buf = [], None, []
    for ln in lines:
        if ln.startswith("## "):
            if cur is not None:
                sections.append((cur, " ".join(buf).strip()))
            cur, buf = ln[3:].strip(), []
        elif not ln.startswith("# "):
            buf.append(ln.strip())
    if cur is not None:
        sections.append((cur, " ".join(buf).strip()))
    if not sections:
        sections = [("1 Body", " ".join(buf).strip())]
    return chunk_document({"doc_id": doc_id, "title": title, "sections": sections, **meta})
