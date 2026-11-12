"""Infrastructure adapters. Each factory returns the same client type in both modes so no other module knows which mode it runs in."""

from radar.infra.clients import get_db_engine, get_qdrant, get_redis, health  # noqa: F401
