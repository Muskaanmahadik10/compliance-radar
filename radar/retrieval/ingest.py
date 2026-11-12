"""Index the generated company documents into Qdrant: python -m radar.retrieval.ingest [--recreate]"""

from __future__ import annotations

import argparse
import json
import time

from radar.config import DATA_DIR
from radar.retrieval.chunking import chunk_document
from radar.retrieval.store import DocumentStore


def ingest(recreate: bool = False) -> dict:
    docs = json.loads((DATA_DIR / "generated" / "documents.json").read_text(encoding="utf-8"))
    store = DocumentStore()
    store.ensure_collection(recreate=recreate)
    t0 = time.perf_counter()
    chunks = [c for d in docs for c in chunk_document(d)]
    n = store.upsert_chunks(chunks)
    return {"documents": len(docs), "chunks": n, "seconds": round(time.perf_counter() - t0, 1), "collection_count": store.count()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true")
    print(json.dumps(ingest(ap.parse_args().recreate)))
