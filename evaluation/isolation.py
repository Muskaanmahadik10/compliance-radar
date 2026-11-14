"""Evaluation state isolation.

The brief evaluation resets what it runs on (fresh company, cleared tombstones, re-indexed documents, cleared event
streams). Run against the app's own state, that reset would undo the app's erasures and wipe its events. So by default
an evaluation runs in its own namespace: its own SQLite file, embedded vector store, collection and generated-data
folder under `.local-eval/<name>/`. Reusing (and resetting) the app's state needs the explicit `--use-app-state` flag.

In compose mode (shared Postgres/Redis/Qdrant servers) there is no separate namespace to switch to, so the evaluation
refuses to run unless `--use-app-state` is given.
"""

from __future__ import annotations

import os
from pathlib import Path

from radar.config import ROOT, settings


def use_isolated_state(name: str, root: Path | None = None) -> dict:
    """Point settings at `.local-eval/<name>/`. Must be called before any store, database or index is opened."""
    from radar.infra import clients

    opened = [f.__name__ for f in (clients.get_db_engine, clients.get_qdrant, clients.get_redis) if f.cache_info().currsize]
    if opened:
        raise RuntimeError(f"use_isolated_state() must run before state is opened (already open: {', '.join(opened)})")
    if settings.mode != "local":
        raise SystemExit(f"RADAR_MODE={settings.mode}: the evaluation would reset the shared services' state. "
                         "Re-run with --use-app-state to allow that explicitly.")
    base = (root or Path(os.environ.get("RADAR_EVAL_STATE_DIR") or ROOT / ".local-eval")) / name
    settings.local_sqlite_path = base / "radar.sqlite"
    settings.local_qdrant_path = base / "qdrant"
    settings.generated_dir = base / "generated"
    settings.collection = f"{settings.collection}_eval"
    return {"namespace": str(base.relative_to(ROOT)) if base.is_relative_to(ROOT) else str(base), "collection": settings.collection}
