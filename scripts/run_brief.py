"""Run one brief for a topic's latest event and print it: python scripts/run_brief.py pfas"""

import sys

from sqlalchemy import select

from radar.agent import start_brief
from radar.db.schema import events
from radar.infra import get_db_engine

sys.stdout.reconfigure(encoding="utf-8")
topic = sys.argv[1] if len(sys.argv) > 1 else "pfas"
with get_db_engine().connect() as c:
    ev = {r.topic: r.event_id for r in c.execute(select(events.c.topic, events.c.event_id))}
b = start_brief(ev[topic])
br = b["brief"]
print("status", b["status"], "| cost", b["cost_usd"], "| latency", b["latency_s"], "| models", br["_models"])
print("title:", br["title"])
print("summary:", br["summary"])
print("counts:", br["affected_counts"], "| confidence", br["confidence"])
print("affected:", [(a["kind"], a["id"]) for a in br["affected"]])
for a in br["actions"]:
    print("ACTION:", a["action"][:110], "|", a["owner_role"], "|", a["deadline"], "|", [c["doc_id"] + " §" + c["section"] for c in a["citations"]])
print("auto:", br["_auto_findings"])
print("verdict:", br["_verdict"])
print("open:", br["open_questions"])
