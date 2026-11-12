import json

from sqlalchemy import func, select

from radar.data.generate import write_all
from radar.data.schemas import MATERIAL_TOPIC, Topic
from radar.db import load_company
from radar.db.schema import bom_lines, parts, suppliers
from radar.infra import get_db_engine


def test_generator_is_deterministic_and_linked(tmp_path):
    a = write_all(tmp_path / "a", seed=3)
    b = write_all(tmp_path / "b", seed=3)
    assert a == b and a["suppliers"] == 200 and a["products"] == 20
    assert (tmp_path / "a" / "suppliers.json").read_text() == (tmp_path / "b" / "suppliers.json").read_text()
    docs = json.loads((tmp_path / "a" / "documents.json").read_text(encoding="utf-8"))
    assert a["by_type"]["contract"] == 30 and a["by_type"]["policy"] == 12
    prts = json.loads((tmp_path / "a" / "parts.json").read_text(encoding="utf-8"))
    sids = {s["supplier_id"] for s in json.loads((tmp_path / "a" / "suppliers.json").read_text(encoding="utf-8"))}
    assert all(p["supplier_id"] in sids for p in prts)
    assert all(p["material"] in MATERIAL_TOPIC for p in prts)
    pfas_docs = [d for d in docs if Topic.PFAS.value in d["topics"]]
    assert any(d["doc_id"] == "POL-ENV-001" for d in pfas_docs)


def test_load_company_into_db(tmp_path):
    write_all(tmp_path / "g", seed=5)
    loaded = load_company(tmp_path / "g")
    eng = get_db_engine()
    with eng.connect() as c:
        assert c.execute(select(func.count()).select_from(suppliers)).scalar() == loaded["suppliers"]
        n_pfas_parts = c.execute(select(func.count()).select_from(parts).where(parts.c.material.in_(["PTFE coating", "fluoropolymer gasket", "PFAS surfactant"]))).scalar()
        assert n_pfas_parts > 0
        assert c.execute(select(func.count()).select_from(bom_lines)).scalar() > 200
