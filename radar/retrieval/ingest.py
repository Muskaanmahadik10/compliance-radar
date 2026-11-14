"""Index the generated company documents into Qdrant: python -m radar.retrieval.ingest [--recreate]

Documents with an erasure tombstone (radar/forget.py) are skipped, whatever the source files contain.
"""

from __future__ import annotations

import argparse
import json
import time

from radar.config import settings
from radar.db import erased_doc_ids
from radar.retrieval.chunking import chunk_document
from radar.retrieval.store import DocumentStore


def ingest(recreate: bool = False) -> dict:
    docs = json.loads((settings.generated_dir / "documents.json").read_text(encoding="utf-8"))
    erased = erased_doc_ids()
    skipped = sorted(d["doc_id"] for d in docs if d["doc_id"] in erased)
    docs = [d for d in docs if d["doc_id"] not in erased]
    store = DocumentStore()
    store.ensure_collection(recreate=recreate)
    t0 = time.perf_counter()
    chunks = [c for d in docs for c in chunk_document(d)]
    n = store.upsert_chunks(chunks)
    return {"documents": len(docs), "chunks": n, "skipped_erased": skipped, "seconds": round(time.perf_counter() - t0, 1), "collection_count": store.count()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true")
    print(json.dumps(ingest(ap.parse_args().recreate)))
