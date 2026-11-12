from sqlalchemy import text

from radar.infra import get_db_engine, get_qdrant, get_redis, health


def test_health_all_ok():
    h = health()
    assert h["qdrant"] == "ok" and h["redis"] == "ok" and h["database"] == "ok", h


def test_redis_streams_work_in_both_modes():
    r = get_redis()
    stream = "test:events"
    r.delete(stream)
    r.xadd(stream, {"source": "unit-test", "title": "hello"})
    try:
        r.xgroup_create(stream, "workers", id="0", mkstream=True)
    except Exception:
        pass
    msgs = r.xreadgroup("workers", "w1", {stream: ">"}, count=10)
    assert msgs and msgs[0][1][0][1]["title"] == "hello"


def test_qdrant_roundtrip():
    from qdrant_client.models import Distance, PointStruct, VectorParams

    q = get_qdrant()
    name = "test_roundtrip"
    if q.collection_exists(name):
        q.delete_collection(name)
    q.create_collection(name, vectors_config=VectorParams(size=3, distance=Distance.COSINE))
    q.upsert(name, [PointStruct(id=1, vector=[0.1, 0.2, 0.3], payload={"doc": "a"})])
    hits = q.query_points(name, query=[0.1, 0.2, 0.3], limit=1).points
    assert hits[0].payload["doc"] == "a"
    q.delete_collection(name)


def test_db_roundtrip():
    eng = get_db_engine()
    with eng.begin() as c:
        c.execute(text("create table if not exists t_probe (id integer primary key, v text)"))
        c.execute(text("delete from t_probe"))
        c.execute(text("insert into t_probe (id, v) values (1, 'x')"))
        assert c.execute(text("select v from t_probe where id = 1")).scalar() == "x"
        c.execute(text("drop table t_probe"))
