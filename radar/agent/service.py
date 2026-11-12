"""Service functions around the graph: start a brief for an event (runs until the human-review interrupt), decide it
(resume), read it back. The API, the stream consumer and the tests all go through these three functions.
"""

from __future__ import annotations

import sqlite3
import uuid
from functools import lru_cache

from langgraph.types import Command
from sqlalchemy import select

from radar.config import settings
from radar.db import create_all
from radar.db.schema import briefs, events, tasks
from radar.infra import get_db_engine


@lru_cache(maxsize=1)
def _checkpointer():
    """Durable checkpoints: SQLite file locally, Postgres in compose mode, memory as the last resort."""
    try:
        if settings.mode == "local":
            from langgraph.checkpoint.sqlite import SqliteSaver

            state_dir = settings.local_sqlite_path.parent
            state_dir.mkdir(parents=True, exist_ok=True)
            return SqliteSaver(sqlite3.connect(str(state_dir / "checkpoints.sqlite"), check_same_thread=False))
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg import Connection

        conn = Connection.connect(settings.database_url.replace("postgresql+psycopg://", "postgresql://"), autocommit=True)
        saver = PostgresSaver(conn)
        saver.setup()
        return saver
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


@lru_cache(maxsize=1)
def graph():
    from radar.agent.graph import build_graph

    return build_graph(_checkpointer())


def _event(event_id: str) -> dict:
    with get_db_engine().connect() as c:
        row = c.execute(select(events).where(events.c.event_id == event_id)).mappings().first()
    if not row:
        raise KeyError(f"unknown event {event_id}")
    return {k: row[k] for k in ("event_id", "source", "url", "title", "summary", "published", "jurisdiction", "topic", "urgency", "language")}


def start_brief(event_id: str) -> dict:
    create_all()
    brief_id = f"BRF-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": brief_id}}
    graph().invoke({"brief_id": brief_id, "event": _event(event_id)}, cfg)
    return get_brief(brief_id)


def decide(brief_id: str, approve: bool, by: str, comment: str = "") -> dict:
    cfg = {"configurable": {"thread_id": brief_id}}
    snap = graph().get_state(cfg)
    if not snap.next:
        raise ValueError(f"brief {brief_id} is not waiting for review")
    graph().invoke(Command(resume={"approve": approve, "by": by, "comment": comment}), cfg)
    return get_brief(brief_id)


def get_brief(brief_id: str) -> dict:
    with get_db_engine().connect() as c:
        row = c.execute(select(briefs).where(briefs.c.brief_id == brief_id)).mappings().first()
        if not row:
            raise KeyError(f"unknown brief {brief_id}")
        t = c.execute(select(tasks).where(tasks.c.brief_id == brief_id)).mappings().all()
    return dict(row) | {"tasks": [dict(x) for x in t]}


def list_briefs(limit: int = 50) -> list[dict]:
    with get_db_engine().connect() as c:
        rows = c.execute(select(briefs.c.brief_id, briefs.c.event_id, briefs.c.status, briefs.c.cost_usd, briefs.c.latency_s, briefs.c.created_at, briefs.c.decided_by).order_by(briefs.c.created_at.desc()).limit(limit)).mappings().all()
    return [dict(r) for r in rows]
