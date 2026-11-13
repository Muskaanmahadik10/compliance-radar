import json

import pytest
from fastapi.testclient import TestClient

from radar.db.schema import events
from radar.infra import get_db_engine, get_redis
from radar.ingest.feeds import RAW_STREAM
from radar.ingest.worker import CLASSIFIED_STREAM


@pytest.fixture(scope="module")
def client():
    from radar.data.build import build_all
    from radar.retrieval.ingest import ingest

    build_all(reset_erasures=True)  # explicit demo reset: tombstones from other test modules are cleared
    ingest(recreate=True)
    r = get_redis()
    r.delete(RAW_STREAM, CLASSIFIED_STREAM)
    with get_db_engine().begin() as c:
        c.execute(events.delete())
    from radar import api

    with TestClient(api.app) as c:
        yield c


def test_full_flow_over_http(client):
    assert client.get("/health").json()["qdrant"] == "ok"
    p = client.post("/ingest/poll", json={"replay": True, "network": False}).json()
    assert p["published"] == 13 and p["stored"] == 12 and p["relevant"] == 11
    evs = client.get("/events", params={"status": "new"}).json()
    assert len(evs) == 11
    rohs = next(e for e in evs if e["topic"] == "rohs")
    b = client.post("/briefs", json={"event_id": rohs["event_id"]}).json()
    assert b["status"] == "pending_review" and b["brief"]["affected_counts"]["suppliers"] > 0
    assert client.post("/briefs", json={"event_id": "nope"}).status_code == 404
    d = client.post(f"/briefs/{b['brief_id']}/decision", json={"approve": True, "by": "manager"}).json()
    assert d["status"] == "approved" and d["tasks"]
    assert client.post(f"/briefs/{b['brief_id']}/decision", json={"approve": True, "by": "manager"}).status_code == 409
    assert any(x["brief_id"] == b["brief_id"] for x in client.get("/briefs").json())


def test_chat_stream_and_search_and_erasure(client):
    with client.stream("POST", "/chat", json={"question": "What is the RoHS threshold for cadmium?"}) as r:
        body = "".join(r.iter_text())
    kinds = [ln.split(": ", 1)[1] for ln in body.splitlines() if ln.startswith("event: ")]
    assert kinds[0] == "sources" and kinds[-1] == "done" and "token" in kinds
    done = json.loads([ln for ln in body.splitlines() if ln.startswith("data: ")][-1][6:])
    assert done["citations"] and done["citations"][0]["doc_id"] == "POL-ENV-002"
    hits = client.get("/search", params={"q": "battery passport carbon footprint", "k": 3}).json()
    assert hits[0]["doc_id"] == "POL-ENV-008"
    cert = client.request("DELETE", "/documents/POL-SEC-010", json={"requested_by": "dpo"}).json()
    assert cert["verified"] is True
    docs = {d["doc_id"]: d["status"] for d in client.get("/documents").json()}
    assert docs["POL-SEC-010"] == "deleted"
    assert client.request("DELETE", "/documents/POL-SEC-010", json={"requested_by": "dpo"}).status_code == 404
    m = client.get("/metrics").json()
    assert m["totals"]["events"] == 12
