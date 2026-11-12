from __future__ import annotations

import atexit
from functools import lru_cache

import redis
from qdrant_client import QdrantClient
from sqlalchemy import Engine, create_engine, text

from radar.config import settings


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    if settings.mode == "local":
        settings.local_qdrant_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(settings.local_qdrant_path))
        atexit.register(client.close)  # embedded Qdrant must be closed before interpreter teardown
        return client
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


@lru_cache(maxsize=1)
def get_redis() -> redis.Redis:
    if settings.mode == "local":
        import fakeredis

        return fakeredis.FakeRedis(decode_responses=True)
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


@lru_cache(maxsize=1)
def get_db_engine() -> Engine:
    if settings.mode == "local":
        settings.local_sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(f"sqlite:///{settings.local_sqlite_path}", future=True)
    return create_engine(settings.database_url, future=True, pool_pre_ping=True)


def health() -> dict:
    out = {"mode": settings.mode}
    try:
        get_qdrant().get_collections()
        out["qdrant"] = "ok"
    except Exception as e:  # pragma: no cover - depends on environment
        out["qdrant"] = f"error: {type(e).__name__}"
    try:
        out["redis"] = "ok" if get_redis().ping() else "error"
    except Exception as e:  # pragma: no cover
        out["redis"] = f"error: {type(e).__name__}"
    try:
        with get_db_engine().connect() as c:
            c.execute(text("select 1"))
        out["database"] = "ok"
    except Exception as e:  # pragma: no cover
        out["database"] = f"error: {type(e).__name__}"
    out["langfuse"] = "enabled" if settings.langfuse_enabled else "disabled"
    return out
