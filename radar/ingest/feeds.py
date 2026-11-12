"""Feed sources and the poller that publishes raw items to the Redis Stream `events:raw`.

Sources are public and official where possible. Each returns normalised dicts; nothing is interpreted here.
A 'replay' source reads a local JSONL fixture so demos, tests and CI are deterministic and need no network.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import feedparser
import httpx

from radar.config import DATA_DIR, settings
from radar.infra import get_redis

log = logging.getLogger(__name__)
RAW_STREAM = "events:raw"

RSS_SOURCES = {
    # Official Journal of the EU, L series (legislation)
    "eurlex_oj": "https://eur-lex.europa.eu/EN/display-feed.rss?myRssId=aa2ebd7c-98ee-4d71-9b1b-6f7f2b1c0c3b",
    # ECHA news (SVHC, restrictions)
    "echa_news": "https://echa.europa.eu/rss/news",
    # US Federal Register: rules from EPA and BIS (export control)
    "federal_register_epa": "https://www.federalregister.gov/api/v1/documents.rss?conditions%5Bagencies%5D%5B%5D=environmental-protection-agency&conditions%5Btype%5D%5B%5D=RULE",
    "federal_register_bis": "https://www.federalregister.gov/api/v1/documents.rss?conditions%5Bagencies%5D%5B%5D=industry-and-security-bureau",
}


def _event_id(source: str, url: str, title: str) -> str:
    return hashlib.sha1(f"{source}|{url or title}".encode()).hexdigest()[:16]


def fetch_rss(source: str, url: str, timeout: float = 20.0) -> list[dict]:
    try:
        r = httpx.get(url, timeout=timeout, headers={"User-Agent": "ComplianceRadar/0.1 (research prototype)"}, follow_redirects=True)
        r.raise_for_status()
    except httpx.HTTPError as e:
        log.warning("feed %s failed: %s", source, e)
        return []
    feed = feedparser.parse(r.text)
    items = []
    for e in feed.entries:
        published = e.get("published") or e.get("updated") or datetime.now(UTC).isoformat()
        items.append({"source": source, "url": e.get("link", ""), "title": e.get("title", "").strip(), "summary": (e.get("summary") or "")[:2000],
                      "published": published, "event_id": _event_id(source, e.get("link", ""), e.get("title", ""))})
    return items


def load_replay(path: Path | None = None) -> list[dict]:
    path = path or DATA_DIR / "fixtures" / "events_replay.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            d.setdefault("event_id", _event_id(d["source"], d.get("url", ""), d["title"]))
            out.append(d)
    return out


def publish(items: list[dict]) -> int:
    """Append to the raw stream. Idempotency is handled downstream (dedupe) so publishing twice is safe."""
    r = get_redis()
    n = 0
    for it in items:
        r.xadd(RAW_STREAM, {"json": json.dumps(it, ensure_ascii=False)}, maxlen=10000, approximate=True)
        n += 1
    return n


def poll_once(use_network: bool = True, replay: bool = True) -> int:
    items: list[dict] = []
    if replay:
        items += load_replay()
    if use_network:
        for name, url in RSS_SOURCES.items():
            items += fetch_rss(name, url)
    return publish(items)


def poll_forever() -> None:  # pragma: no cover
    while True:
        n = poll_once()
        log.info("published %d items", n)
        time.sleep(settings.feed_poll_seconds)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print(poll_once())
