"""Generate the company and load it into the database: python -m radar.data.build [--reset-erasures]

Erased documents stay erased: after regenerating the synthetic files, every tombstoned document is marked deleted in
the database and removed again from documents.json and docs/*.md. Only --reset-erasures (a deliberate demo reset)
clears the tombstones and restores the full synthetic corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radar.config import DATA_DIR
from radar.data.generate import write_all
from radar.db import clear_erasures, erased_doc_ids, load_company


def purge_erased_sources(gen_dir: Path, doc_ids: set[str] | None = None) -> list[str]:
    """Remove erased documents from the generated source copies (documents.json and docs/<id>.md). Returns ids removed."""
    doc_ids = erased_doc_ids() if doc_ids is None else doc_ids
    if not doc_ids:
        return []
    removed: list[str] = []
    path = gen_dir / "documents.json"
    if path.exists():
        docs = json.loads(path.read_text(encoding="utf-8"))
        kept = [d for d in docs if d["doc_id"] not in doc_ids]
        removed = [d["doc_id"] for d in docs if d["doc_id"] in doc_ids]
        if removed:
            path.write_text(json.dumps(kept, indent=1), encoding="utf-8")
    for doc_id in doc_ids:
        md = gen_dir / "docs" / f"{doc_id}.md"
        if md.exists():
            md.unlink()
            if doc_id not in removed:
                removed.append(doc_id)
    return removed


def build_all(seed: int = 7, reset_erasures: bool = False) -> dict:
    if reset_erasures:
        clear_erasures()
    gen = DATA_DIR / "generated"
    meta = write_all(gen, seed)
    loaded = load_company(gen)  # marks tombstoned documents as deleted
    purged = purge_erased_sources(gen)
    print({"generated": meta, "loaded": loaded, "erased_documents_kept_out": purged})
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset-erasures", action="store_true", help="demo reset: clear erasure tombstones and restore every synthetic document")
    build_all(reset_erasures=ap.parse_args().reset_erasures)
