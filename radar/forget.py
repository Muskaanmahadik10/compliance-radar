"""Right to be forgotten for a RAG system: delete a document everywhere it lives, then *prove* it is gone.

Deleting a row is easy; showing that the document can no longer influence an answer is the part auditors ask about.
The procedure:
  1. Baseline  - build probe questions from the document's own sections and confirm the system can currently retrieve it.
  2. Delete    - vector store chunks, relational record (status -> deleted), source file, cached answers.
  3. Verify    - (a) zero chunks remain; (b) no probe retrieves it in dense, sparse or hybrid mode; (c) answers to the
                 probes no longer cite it; (d) a scan of the whole collection finds no payload with that doc id.
  4. Certify   - a signed JSON certificate (HMAC-SHA256) with the content hash, the checks and their results, stored in
                 the audit log and on disk.

Scope note (kept honest): the LLM was never trained on these documents, so nothing has to be removed from model weights.
If a model *is* fine-tuned on company documents, deletion must also trigger retraining or an unlearning method; the
certificate records which models were in use so that case is visible.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import UTC, datetime

from qdrant_client import models
from sqlalchemy import insert, select, update

from radar import chat
from radar.config import DATA_DIR, settings
from radar.db import create_all
from radar.db.schema import audit_log, briefs, documents
from radar.infra import get_db_engine, get_redis
from radar.llm import get_router
from radar.retrieval import DocumentStore

SIGNING_KEY = os.environ.get("RADAR_SIGNING_KEY", "dev-signing-key-change-me").encode()
MODES = ("dense", "sparse", "hybrid")


def _doc_chunks(store: DocumentStore, doc_id: str) -> list[dict]:
    flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
    pts, _ = store.client.scroll(store.collection, scroll_filter=flt, limit=500, with_payload=True)
    return [p.payload for p in pts]


def _probes(chunks: list[dict], n: int = 10) -> list[str]:
    """Probe questions made from the document's own wording: the hardest case for 'it is gone'."""
    out = []
    for c in chunks[:n]:
        body = c["text"].split(". ", 2)[-1]
        out.append(body[:220])
    return out


def _retrieved_by(store: DocumentStore, probes: list[str], doc_id: str, k: int = 10) -> dict:
    hits = {m: 0 for m in MODES}
    for q in probes:
        for m in MODES:
            if any(h.doc_id == doc_id for h in store.search(q, k=k, mode=m, rerank=False)):
                hits[m] += 1
    return hits


def sign(payload: dict) -> str:
    return hmac.new(SIGNING_KEY, json.dumps(payload, sort_keys=True, ensure_ascii=False).encode(), hashlib.sha256).hexdigest()


def verify_certificate(cert: dict) -> bool:
    body = {k: v for k, v in cert.items() if k != "signature"}
    return hmac.compare_digest(sign(body), cert.get("signature", ""))


def forget_document(doc_id: str, requested_by: str, reason: str = "erasure request", store: DocumentStore | None = None) -> dict:
    create_all()
    store = store or DocumentStore()
    eng = get_db_engine()
    chunks = _doc_chunks(store, doc_id)
    if not chunks:
        raise KeyError(f"document {doc_id} has no indexed content")
    content_hash = hashlib.sha256("\n".join(sorted(c["text"] for c in chunks)).encode()).hexdigest()
    probes = _probes(chunks)

    # 1. baseline: the system can find it now (otherwise 'not found afterwards' proves nothing)
    before = _retrieved_by(store, probes, doc_id)

    # 2. delete everywhere
    removed = store.delete_document(doc_id)
    with eng.begin() as c:
        c.execute(update(documents).where(documents.c.doc_id == doc_id).values(status="deleted", n_chunks=0))
        citing = [r.brief_id for r in c.execute(select(briefs.c.brief_id, briefs.c.brief)) if any(ci.get("doc_id") == doc_id for ci in (r.brief or {}).get("citations", []))]
    src = DATA_DIR / "generated" / "docs" / f"{doc_id}.md"
    file_removed = src.exists()
    if file_removed:
        src.unlink()
    r = get_redis()
    cache_keys = [k for k in r.scan_iter("chatcache:*")]
    if cache_keys:
        r.delete(*cache_keys)

    # 3. verify
    remaining = store.count_document(doc_id)
    after = _retrieved_by(store, probes, doc_id)
    scan_left = len(_doc_chunks(store, doc_id))
    cited_after = 0
    for q in probes:
        res = chat.answer(q, store=store, rewrite=False)
        if any(ci["doc_id"] == doc_id for ci in res.citations) or doc_id in res.answer:
            cited_after += 1
    checks = {
        "baseline_document_was_retrievable": sum(before.values()) > 0,
        "no_chunks_remaining": remaining == 0,
        "no_probe_retrieves_document": sum(after.values()) == 0,
        "no_answer_cites_document": cited_after == 0,
        "collection_scan_clean": scan_left == 0,
    }
    cert = {
        "certificate_type": "document_erasure", "doc_id": doc_id, "content_sha256": content_hash, "requested_by": requested_by, "reason": reason,
        "completed_at": datetime.now(UTC).isoformat(), "chunks_removed": removed, "source_file_removed": file_removed, "cache_entries_purged": len(cache_keys),
        "probes": len(probes), "retrieved_before": before, "retrieved_after": after, "answers_citing_after": cited_after,
        "briefs_citing_document": citing, "models_in_use": get_router().names, "checks": checks, "verified": all(checks.values()),
        "scope_note": "Retrieval-store erasure. No model in use was trained on this document; fine-tuned models would additionally require retraining or unlearning.",
    }
    cert["signature"] = sign(cert)
    with eng.begin() as c:
        c.execute(insert(audit_log), [{"ts": cert["completed_at"], "actor": requested_by, "action": "document_erased", "subject": doc_id, "detail": cert}])
    out_dir = settings.artifacts_dir / "certificates"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"erasure_{doc_id}.json").write_text(json.dumps(cert, indent=2, ensure_ascii=False), encoding="utf-8")
    return cert
