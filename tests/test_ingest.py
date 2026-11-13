from sqlalchemy import select

from radar.data.schemas import Jurisdiction, Topic
from radar.db.schema import events
from radar.infra import get_db_engine, get_redis
from radar.ingest.classify import classify_rules
from radar.ingest.feeds import RAW_STREAM, load_replay, publish
from radar.ingest.worker import CLASSIFIED_STREAM, Deduper, process_available


def test_rule_classifier_on_fixture_events():
    items = load_replay()
    assert len(items) == 13
    got = {i["url"]: classify_rules(i["title"], i["summary"]) for i in items}
    assert got["https://echa.europa.eu/-/pfas-restriction-proposal-progress"].topic == Topic.PFAS
    assert got["https://eur-lex.europa.eu/eli/dir_del/2026/0001/oj"].topic == Topic.ROHS
    assert got["https://www.federalregister.gov/d/2026-bis-entity-list"].topic == Topic.EXPORT_CONTROL
    assert got["https://www.federalregister.gov/d/2026-bis-entity-list"].urgency == "high"
    assert got["https://example-news.com/supplier-fire-vietnam"].topic == Topic.SUPPLIER_INCIDENT
    assert got["https://example-news.com/quarterly-earnings"].topic == Topic.OTHER
    tw = got["https://www.moea.gov.tw/news/2026/export-control-update"]
    assert tw.jurisdiction == Jurisdiction.TW and tw.language.startswith("zh")


def test_deduper_collapses_near_duplicate_titles():
    d = Deduper()
    assert not d.is_duplicate("a", "ECHA committees advance universal PFAS restriction: fluoropolymers scoped with transition periods")
    assert d.is_duplicate("b", "ECHA committees advance universal PFAS restriction - fluoropolymers scoped with transition periods")
    assert not d.is_duplicate("c", "BIS adds 12 entities to the Entity List")


def test_worker_end_to_end_in_streams():
    r = get_redis()
    r.delete(RAW_STREAM, CLASSIFIED_STREAM)
    eng = get_db_engine()
    from radar.db import create_all

    create_all()
    with eng.begin() as c:
        c.execute(events.delete())
    n = publish(load_replay())
    assert n == 13
    stats = process_available(Deduper())
    assert stats["read"] == 13 and stats["duplicates"] == 1 and stats["stored"] == 12
    assert stats["relevant"] == 11  # all but the earnings item
    with eng.connect() as c:
        rows = c.execute(select(events.c.topic, events.c.status)).all()
    assert len(rows) == 12 and sum(1 for _, s in rows if s == "ignored") == 1
    assert r.xlen(CLASSIFIED_STREAM) == 11
    # re-publishing the same items is idempotent
    publish(load_replay())
    again = process_available(Deduper())
    assert again["stored"] == 0 and again["duplicates"] == 13


def test_tiered_classifier_uses_lora_then_llm():
    """Chain logic with stand-ins: no torch or provider needed."""
    from radar.ingest.classify import classify

    class FakeLora:
        def __init__(self, label, conf):
            self.label, self.conf, self.calls = label, conf, 0

        def predict(self, titles):
            self.calls += 1
            return [(self.label, self.conf)]

    llm_calls = []

    def llm(prompt):
        llm_calls.append(prompt)
        return "battery"

    vague = "Lawmakers confirm new targets for rechargeable cells in consumer devices"  # no keyword: rules are unsure
    assert classify_rules(vague).confidence < 0.6
    # confident LoRA answers and the LLM is never called
    sure = FakeLora("battery", 0.95)
    c = classify(vague, llm=llm, lora=sure, lora_min_confidence=0.8)
    assert (c.topic, c.tier) == (Topic.BATTERY, "lora") and not llm_calls and sure.calls == 1
    # unsure LoRA defers to the LLM
    c = classify(vague, llm=llm, lora=FakeLora("packaging", 0.4), lora_min_confidence=0.8)
    assert (c.topic, c.tier) == (Topic.BATTERY, "llm") and len(llm_calls) == 1
    # unsure LoRA and no LLM: its guess still beats 'other'
    c = classify(vague, llm=None, lora=FakeLora("packaging", 0.4), lora_min_confidence=0.8)
    assert (c.topic, c.tier) == (Topic.PACKAGING, "lora")
    # a clear keyword never reaches the later tiers
    sure2 = FakeLora("other", 0.99)
    c = classify("BIS adds entities to the Entity List under export control rules", llm=llm, lora=sure2)
    assert c.tier == "rules" and sure2.calls == 0
