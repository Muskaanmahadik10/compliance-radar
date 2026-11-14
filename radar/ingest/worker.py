"""Stream worker: consumes `events:raw` with a consumer group, deduplicates (exact id + MinHash near-duplicate on title),
classifies (rules -> LoRA -> LLM, see classify.py), stores to the events table, and publishes relevant event ids to
`events:classified`.

Nothing in this repository consumes `events:classified` yet: a brief is started by a person (UI button or POST /briefs).
Feeds are published when someone presses Poll in the UI or calls POST /ingest/poll; there is no scheduled poller.
Known prototype limits: pending (delivered but unacknowledged) messages are not reclaimed after a crash, and the SQL
insert and the classified-stream publish are not atomic (no outbox).

Run: python -m radar.ingest.worker        (loops)      |  process_available() is used by tests, the API and the UI.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime

from datasketch import MinHash, MinHashLSH
from sqlalchemy import insert, select

from radar.data.schemas import Topic
from radar.db import create_all
from radar.db.schema import events
from radar.infra import get_db_engine, get_redis
from radar.ingest.classify import classify, llm_callback
from radar.ingest.feeds import RAW_STREAM

log = logging.getLogger(__name__)
CLASSIFIED_STREAM = "events:classified"
GROUP, CONSUMER = "workers", "worker-1"
RELEVANT = {t for t in Topic if t != Topic.OTHER}


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _shingles(text: str, k: int = 2) -> set[str]:
    """Word bigrams on punctuation-stripped lowercase text; CJK text (no spaces) falls back to character bigrams."""
    norm = _PUNCT.sub(" ", text.lower())
    toks = norm.split()
    if len(toks) < 4:
        compact = "".join(toks)
        return {compact[i: i + 2] for i in range(max(1, len(compact) - 1))}
    return {" ".join(toks[i: i + k]) for i in range(max(1, len(toks) - k + 1))}


class Deduper:
    """Near-duplicate titles (same regulation reported by two feeds) collapse into one event."""

    def __init__(self, threshold: float = 0.7):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=64)
        self.seen_ids: set[str] = set()

    def _mh(self, text: str) -> MinHash:
        m = MinHash(num_perm=64)
        for s in _shingles(text):
            m.update(s.encode("utf-8"))
        return m

    def is_duplicate(self, event_id: str, title: str) -> bool:
        if event_id in self.seen_ids:
            return True
        m = self._mh(title)
        if self.lsh.query(m):
            return True
        self.seen_ids.add(event_id)
        self.lsh.insert(event_id, m)
        return False

    def seed(self, event_id: str, title: str) -> None:
        """Register an already-stored event (id and title) so near-duplicates are still caught after a restart."""
        if event_id in self.seen_ids:
            return
        self.seen_ids.add(event_id)
        self.lsh.insert(event_id, self._mh(title))


def _ensure_group(r) -> None:
    try:
        r.xgroup_create(RAW_STREAM, GROUP, id="0", mkstream=True)
    except Exception:
        pass  # group exists


_AUTO = object()


def process_available(deduper: Deduper | None = None, llm=_AUTO, max_items: int = 500) -> dict:
    """Process everything currently in the raw stream once. Returns counters.

    `llm` defaults to the configured LLM tier (classify.llm_callback: the cheap model, or None under the offline mock),
    so the worker, the API and the UI all run the same rules -> LoRA -> LLM chain. Pass llm=None to switch the tier off.
    """
    create_all()
    r = get_redis()
    _ensure_group(r)
    if llm is _AUTO:
        llm = llm_callback()
    llm_name = getattr(llm, "model_name", None) if llm is not None else None
    deduper = deduper or Deduper()
    eng = get_db_engine()
    stats = {"read": 0, "duplicates": 0, "stored": 0, "relevant": 0}
    with eng.connect() as c:
        known = c.execute(select(events.c.event_id, events.c.title)).all()
    for eid, title in known:
        deduper.seed(eid, title)
    msgs = r.xreadgroup(GROUP, CONSUMER, {RAW_STREAM: ">"}, count=max_items)
    for _stream, entries in msgs or []:
        for msg_id, fields in entries:
            stats["read"] += 1
            item = json.loads(fields["json"])
            if deduper.is_duplicate(item["event_id"], item["title"]):
                stats["duplicates"] += 1
                r.xack(RAW_STREAM, GROUP, msg_id)
                continue
            cl = classify(item["title"], item.get("summary", ""), llm=llm)
            row = {"event_id": item["event_id"], "source": item["source"], "url": item.get("url", ""), "title": item["title"], "summary": item.get("summary", ""),
                   "published": str(item.get("published", datetime.now(UTC).isoformat())), "jurisdiction": cl.jurisdiction.value, "topic": cl.topic.value,
                   "urgency": cl.urgency, "language": cl.language, "status": "new" if cl.topic in RELEVANT else "ignored",
                   "raw": {"classifier": cl.tier, "confidence": cl.confidence, "resolved": cl.resolved, "classifier_llm": llm_name if cl.tier == "llm" else None}}
            with eng.begin() as c:
                c.execute(insert(events), [row])
            stats["stored"] += 1
            if cl.topic in RELEVANT:
                stats["relevant"] += 1
                r.xadd(CLASSIFIED_STREAM, {"event_id": item["event_id"], "topic": cl.topic.value, "urgency": cl.urgency}, maxlen=10000, approximate=True)
            r.xack(RAW_STREAM, GROUP, msg_id)
    return stats


def run_forever(poll_seconds: float = 5.0) -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    d = Deduper()
    while True:
        s = process_available(d)
        if s["read"]:
            log.info("%s", s)
        time.sleep(poll_seconds)


if __name__ == "__main__":  # pragma: no cover
    run_forever()
