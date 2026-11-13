"""Document erasure for a RAG system: remove a document from retrieval, keep it out, and record what was checked.

The procedure:
  0. Tombstone - record the erasure in the `erasures` table first. Every ingestion and rebuild path skips tombstoned ids,
                 so ordinary re-indexing cannot bring the document back (only an explicit demo reset clears tombstones).
  1. Baseline  - build probe questions from the document's own sections and confirm the system can currently retrieve it.
  2. Delete    - vector-store chunks; the SQL document row is marked deleted (its title and metadata stay, as a record);
                 the generated source copies in this process's filesystem (documents.json entry and docs/<id>.md).
  3. Check     - (a) zero chunks remain; (b) no probe retrieves it in dense, sparse or hybrid mode; (c) answers to the
                 probes do not cite its id; (d) a filtered scroll finds no payload with that doc id; (e) the tombstone is
                 stored; (f) no source copy is left in this process's generated-data folder.
  4. Record    - a JSON certificate with the content hash, the checks and their results, HMAC-SHA256 signed, stored in
                 the audit log and on disk.

What this is not: the checks are a record of what was observed when the erasure ran, not a proof that the content can
no longer influence anything. Briefs, tasks and graph checkpoints that quoted the document are listed, not redacted;
browser chat history, backups and copies in other containers' filesystems are out of scope. Equivalent wording in
another document can still produce the same answer. The signature shows the certificate was not altered by someone
without the key; with the public demo key (no RADAR_SIGNING_KEY) it shows nothing about who issued it, and the
certificate says so. No model in this repository was trained on the company documents (the LoRA classifier was
trained on generated headlines), so no weights need retraining or unlearning.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime

from qdrant_client import models
from sqlalchemy import insert, select, update

from radar import chat
from radar.config import DATA_DIR, settings
from radar.db import create_all
from radar.db.schema import audit_log, briefs, documents, erasures
from radar.infra import get_db_engine
from radar.llm import get_router
from radar.retrieval import DocumentStore

DEMO_SIGNING_KEY = "dev-signing-key-change-me"  # public: it is in this repository
DEMO_KEY_ID = "demo-public"
MODES = ("dense", "sparse", "hybrid")


def _key_id(key: str) -> str:
    return DEMO_KEY_ID if key == DEMO_SIGNING_KEY else "configured-" + hashlib.sha256(key.encode()).hexdigest()[:8]


def _active_key() -> tuple[str, str]:
    key = settings.signing_key or DEMO_SIGNING_KEY
    return key, _key_id(key)


def _doc_chunks(store: DocumentStore, doc_id: str) -> list[dict]:
    """Every chunk of the document (paginated scroll)."""
    flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
    out, offset = [], None
    while True:
        pts, offset = store.client.scroll(store.collection, scroll_filter=flt, limit=500, with_payload=True, offset=offset)
        out += [p.payload for p in pts]
        if offset is None or not pts:
            return out


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


def sign(payload: dict, key: str | None = None) -> str:
    key = key or _active_key()[0]
    return hmac.new(key.encode(), json.dumps(payload, sort_keys=True, ensure_ascii=False).encode(), hashlib.sha256).hexdigest()


def verify_certificate(cert: dict) -> bool:
    """Integrity: the signature matches the body under the key the certificate names (demo key or the configured key)."""
    body = {k: v for k, v in cert.items() if k != "signature"}
    kid = cert.get("signature_key_id", DEMO_KEY_ID)  # certificates from before key ids were demo-signed
    if kid == DEMO_KEY_ID:
        key = DEMO_SIGNING_KEY
    elif settings.signing_key and _key_id(settings.signing_key) == kid:
        key = settings.signing_key
    else:
        return False  # signed with a key this process does not hold
    return hmac.compare_digest(sign(body, key), cert.get("signature", ""))


def certificate_authentic(cert: dict) -> bool:
    """Authenticity needs a private key: a valid signature under the public demo key does not count."""
    return cert.get("signature_key_id", DEMO_KEY_ID) != DEMO_KEY_ID and verify_certificate(cert)


def _source_copies(doc_id: str) -> list[str]:
    gen = DATA_DIR / "generated"
    found = []
    if (gen / "docs" / f"{doc_id}.md").exists():
        found.append(f"docs/{doc_id}.md")
    p = gen / "documents.json"
    if p.exists() and any(d.get("doc_id") == doc_id for d in json.loads(p.read_text(encoding="utf-8"))):
        found.append("documents.json")
    return found


def forget_document(doc_id: str, requested_by: str, reason: str = "erasure request", store: DocumentStore | None = None) -> dict:
    from radar.data.build import purge_erased_sources

    create_all()
    store = store or DocumentStore()
    eng = get_db_engine()
    chunks = _doc_chunks(store, doc_id)
    if not chunks:
        raise KeyError(f"document {doc_id} has no indexed content")
    content_hash = hashlib.sha256("\n".join(sorted(c["text"] for c in chunks)).encode()).hexdigest()
    probes = _probes(chunks)

    # 0. tombstone first, so a crash part-way or a concurrent re-index cannot bring the document back
    started = datetime.now(UTC).isoformat()
    with eng.begin() as c:
        c.execute(erasures.delete().where(erasures.c.doc_id == doc_id))
        c.execute(insert(erasures), [{"doc_id": doc_id, "erased_at": started, "requested_by": requested_by, "reason": reason, "state": "in_progress"}])

    # 1. baseline: the system can find it now (otherwise 'not found afterwards' proves nothing)
    before = _retrieved_by(store, probes, doc_id)

    # 2. delete from retrieval and from the local source copies
    removed = store.delete_document(doc_id)
    with eng.begin() as c:
        c.execute(update(documents).where(documents.c.doc_id == doc_id).values(status="deleted", n_chunks=0))
        citing = [r.brief_id for r in c.execute(select(briefs.c.brief_id, briefs.c.brief)) if any(ci.get("doc_id") == doc_id for ci in (r.brief or {}).get("citations", []))]
    copies_before = _source_copies(doc_id)
    purge_erased_sources(DATA_DIR / "generated", {doc_id})

    # 3. verify
    remaining = store.count_document(doc_id)
    after = _retrieved_by(store, probes, doc_id)
    scan_left = len(_doc_chunks(store, doc_id))
    cited_after = 0
    for q in probes:
        res = chat.answer(q, store=store, rewrite=False)
        if any(ci["doc_id"] == doc_id for ci in res.citations) or doc_id in res.answer:
            cited_after += 1
    copies_after = _source_copies(doc_id)
    with eng.begin() as c:
        c.execute(update(erasures).where(erasures.c.doc_id == doc_id).values(state="completed"))
        tombstoned = c.execute(select(erasures.c.doc_id).where(erasures.c.doc_id == doc_id)).first() is not None
    checks = {
        "baseline_document_was_retrievable": sum(before.values()) > 0,
        "no_chunks_remaining": remaining == 0,
        "no_probe_retrieves_document": sum(after.values()) == 0,
        "no_answer_cites_document": cited_after == 0,
        "filtered_scroll_clean": scan_left == 0,
        "tombstone_blocks_reingestion": tombstoned,
        "no_local_source_copy": not copies_after,
    }
    key, key_id = _active_key()
    cert = {
        "certificate_type": "document_erasure", "doc_id": doc_id, "content_sha256": content_hash, "requested_by": requested_by, "reason": reason,
        "started_at": started, "completed_at": datetime.now(UTC).isoformat(), "chunks_removed": removed, "source_copies_removed": copies_before,
        "probes": len(probes), "retrieved_before": before, "retrieved_after": after, "answers_citing_after": cited_after,
        "briefs_citing_document": citing, "chat_models_used_for_probe_answers": get_router().names, "checks": checks, "verified": all(checks.values()),
        "scope_note": ("Retrieval erasure with recorded checks. Removed: vector chunks, local source copies; SQL row kept as 'deleted' with its title; "
                       "tombstone blocks re-ingestion. Not removed: briefs, tasks and checkpoints that quote it (listed above), browser chat history, "
                       "backups, copies in other containers. The checks record what was observed at completed_at; they are not a proof of zero influence."),
        "signature_key_id": key_id,
        "signature_meaning": ("integrity only: signed with the public demo key, so anyone can produce this signature" if key_id == DEMO_KEY_ID
                              else "signed with the deployment's private key (RADAR_SIGNING_KEY)"),
    }
    cert["signature"] = sign(cert, key)
    with eng.begin() as c:
        c.execute(insert(audit_log), [{"ts": cert["completed_at"], "actor": requested_by, "action": "document_erased", "subject": doc_id, "detail": cert}])
    out_dir = settings.artifacts_dir / "certificates"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"erasure_{doc_id}.json").write_text(json.dumps(cert, indent=2, ensure_ascii=False), encoding="utf-8")
    return cert
