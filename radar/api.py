"""FastAPI service.

  GET  /health                         infrastructure status
  POST /ingest/poll                    publish feed items to the stream (replay fixture and/or live feeds) and process them
                                       (user-triggered: there is no scheduled poller)
  GET  /events                         classified events (newest first)
  POST /briefs                         start the brief pipeline for an event; returns when it is waiting for review
  GET  /briefs, /briefs/{id}           list / read briefs
  POST /briefs/{id}/decision           approve or reject (resumes the graph)
  POST /chat                           server-sent events: sources, tokens, final citations and cost
  GET  /search                         hybrid search (debugging and demos)
  GET  /documents                      knowledge-base documents
  DELETE /documents/{id}               retrieval erasure with recorded checks and a tombstone; returns the certificate
  GET  /metrics                        evaluation results and running totals

No authentication: reviewer and requester names are plain request fields. Run it only on a trusted local machine.

Run: uvicorn radar.api:app --reload
"""

from __future__ import annotations

import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sse_starlette.sse import EventSourceResponse

from radar import __version__, chat
from radar.agent import service
from radar.config import ARTIFACTS_DIR, settings
from radar.db import create_all
from radar.db.schema import briefs, documents, events
from radar.forget import forget_document
from radar.infra import get_db_engine, health
from radar.ingest import feeds, worker
from radar.ingest.classify import active_tiers
from radar.llm import get_router
from radar.retrieval import DocumentStore

app = FastAPI(title="ComplianceRadar", version=__version__)


class PollIn(BaseModel):
    replay: bool = True
    network: bool = False


class BriefIn(BaseModel):
    event_id: str


class DecisionIn(BaseModel):
    approve: bool
    by: str = Field(min_length=2, max_length=64)
    comment: str = ""


class ChatIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    history: list[dict] = []
    rewrite: bool = True


class ForgetIn(BaseModel):
    requested_by: str = Field(min_length=2, max_length=64)
    reason: str = "erasure request"


@app.on_event("startup")
def _startup() -> None:
    create_all()


@app.get("/health")
def get_health():
    try:
        models = get_router().names
    except Exception as e:  # e.g. a real provider selected without its key: report it instead of failing silently
        models = {"error": str(e)}
    return health() | {"version": __version__, "llm_provider": settings.llm_provider, "models": models, "classifier_tiers": active_tiers(),
                       "signing_key": "configured" if settings.signing_key else "demo (public default)"}


@app.post("/ingest/poll")
def ingest_poll(body: PollIn):
    published = feeds.poll_once(use_network=body.network, replay=body.replay)
    return {"published": published} | worker.process_available()


@app.get("/events")
def list_events(status: str | None = None, limit: int = 100):
    q = select(events).order_by(events.c.published.desc()).limit(limit)
    if status:
        q = q.where(events.c.status == status)
    with get_db_engine().connect() as c:
        return [dict(r) for r in c.execute(q).mappings()]


@app.post("/briefs")
def create_brief(body: BriefIn):
    try:
        return service.start_brief(body.event_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.get("/briefs")
def get_briefs():
    return service.list_briefs()


@app.get("/briefs/{brief_id}")
def get_brief(brief_id: str):
    try:
        return service.get_brief(brief_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/briefs/{brief_id}/decision")
def decide_brief(brief_id: str, body: DecisionIn):
    try:
        return service.decide(brief_id, body.approve, body.by, body.comment)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@app.post("/chat")
def chat_stream(body: ChatIn):
    def gen():
        for ev in chat.stream_answer(body.question, body.history, rewrite=body.rewrite):
            yield {"event": ev["type"], "data": json.dumps(ev, ensure_ascii=False)}

    return EventSourceResponse(gen())


@app.get("/search")
def search(q: str, k: int = 5, mode: Literal["dense", "sparse", "hybrid"] = "hybrid", rerank: bool = True, doc_type: str | None = None):
    return [h.to_dict() for h in DocumentStore().search(q, k=k, mode=mode, rerank=rerank, doc_type=doc_type)]


@app.get("/documents")
def list_documents():
    with get_db_engine().connect() as c:
        return [dict(r) for r in c.execute(select(documents).order_by(documents.c.doc_id)).mappings()]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, body: ForgetIn):
    try:
        return forget_document(doc_id, body.requested_by, body.reason)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.get("/metrics")
def metrics():
    out: dict = {}
    for p in sorted(ARTIFACTS_DIR.glob("*_eval*.json")):  # current results only; superseded ones live in a subfolder
        out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    with get_db_engine().connect() as c:
        out["totals"] = {
            "events": c.execute(select(func.count()).select_from(events)).scalar(),
            "briefs": c.execute(select(func.count()).select_from(briefs)).scalar(),
            "brief_cost_usd": float(c.execute(select(func.coalesce(func.sum(briefs.c.cost_usd), 0.0))).scalar() or 0.0),
            "documents_active": c.execute(select(func.count()).select_from(documents).where(documents.c.status == "active")).scalar(),
        }
    return out
