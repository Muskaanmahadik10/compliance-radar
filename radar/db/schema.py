from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import JSON, Column, Float, Integer, MetaData, String, Table, Text, insert

from radar.config import DATA_DIR
from radar.infra import get_db_engine

metadata = MetaData()

suppliers = Table("suppliers", metadata,
                  Column("supplier_id", String(16), primary_key=True), Column("name", String(200)), Column("country", String(64)),
                  Column("tier", Integer), Column("certifications", JSON), Column("materials", JSON), Column("rba_audit_score", Integer),
                  Column("risk_flags", JSON), Column("contact_email", String(200)))
parts = Table("parts", metadata,
              Column("part_id", String(16), primary_key=True), Column("description", String(200)), Column("material", String(100)),
              Column("supplier_id", String(16), index=True), Column("unit_cost_usd", Float))
products = Table("products", metadata,
                 Column("product_id", String(32), primary_key=True), Column("name", String(200)), Column("markets", JSON))
bom_lines = Table("bom_lines", metadata,
                  Column("id", Integer, primary_key=True, autoincrement=True), Column("product_id", String(32), index=True),
                  Column("part_id", String(16), index=True), Column("qty", Integer))
documents = Table("documents", metadata,
                  Column("doc_id", String(64), primary_key=True), Column("doc_type", String(16)), Column("title", String(300)),
                  Column("topics", JSON), Column("jurisdiction", String(16)), Column("supplier_id", String(16), nullable=True),
                  Column("effective_date", String(16)), Column("n_chunks", Integer, default=0), Column("status", String(16), default="active"))
events = Table("events", metadata,
               Column("event_id", String(64), primary_key=True), Column("source", String(64)), Column("url", Text), Column("title", Text),
               Column("summary", Text), Column("published", String(32)), Column("jurisdiction", String(16)), Column("topic", String(32)),
               Column("urgency", String(8)), Column("language", String(8)), Column("status", String(16), default="new"), Column("raw", JSON))
briefs = Table("briefs", metadata,
               Column("brief_id", String(64), primary_key=True), Column("event_id", String(64), index=True), Column("status", String(16)),
               Column("brief", JSON), Column("cost_usd", Float), Column("latency_s", Float), Column("created_at", String(32)),
               Column("decided_by", String(64), nullable=True), Column("decided_at", String(32), nullable=True))
tasks = Table("tasks", metadata,
              Column("task_id", String(64), primary_key=True), Column("brief_id", String(64), index=True), Column("action", Text),
              Column("owner_role", String(64)), Column("deadline", String(32)), Column("status", String(16), default="open"))
audit_log = Table("audit_log", metadata,
                  Column("id", Integer, primary_key=True, autoincrement=True), Column("ts", String(32)), Column("actor", String(64)),
                  Column("action", String(64)), Column("subject", String(128)), Column("detail", JSON))


def create_all() -> None:
    metadata.create_all(get_db_engine())


def load_company(gen_dir: Path = DATA_DIR / "generated") -> dict:
    """Idempotent load of the generated company into the relational tables."""
    create_all()
    eng = get_db_engine()
    sup = json.loads((gen_dir / "suppliers.json").read_text(encoding="utf-8"))
    prt = json.loads((gen_dir / "parts.json").read_text(encoding="utf-8"))
    prd = json.loads((gen_dir / "products.json").read_text(encoding="utf-8"))
    docs = json.loads((gen_dir / "documents.json").read_text(encoding="utf-8"))
    with eng.begin() as c:
        for t in (bom_lines, products, parts, suppliers, documents):
            c.execute(t.delete())
        c.execute(insert(suppliers), sup)
        c.execute(insert(parts), prt)
        c.execute(insert(products), [{"product_id": p["product_id"], "name": p["name"], "markets": p["markets"]} for p in prd])
        c.execute(insert(bom_lines), [{"product_id": p["product_id"], "part_id": pid, "qty": q} for p in prd for pid, q in p["bom"]])
        c.execute(insert(documents), [{k: d[k] for k in ("doc_id", "doc_type", "title", "topics", "jurisdiction", "supplier_id", "effective_date")} | {"n_chunks": len(d["sections"]), "status": "active"} for d in docs])
    return {"suppliers": len(sup), "parts": len(prt), "products": len(prd), "documents": len(docs)}
