import pytest

from radar.data.generate import write_all
from radar.retrieval.chunking import chunk_document, chunk_markdown
from radar.retrieval.store import DocumentStore


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    import json

    d = tmp_path_factory.mktemp("gen")
    write_all(d, seed=9)
    docs = json.loads((d / "documents.json").read_text(encoding="utf-8"))
    s = DocumentStore(collection="test_docs")
    s.ensure_collection(recreate=True)
    s.upsert_chunks([c for doc in docs for c in chunk_document(doc)])
    return s


def test_chunking_keeps_section_citations():
    doc = {"doc_id": "X-1", "doc_type": "policy", "title": "T", "topics": ["pfas"], "jurisdiction": "EU", "supplier_id": None, "effective_date": "2025-01-01",
           "sections": [("1 A", "short."), ("2 B", "Sentence one. " * 200)]}
    ch = chunk_document(doc)
    assert ch[0].citation == "X-1 §1 A"
    assert len(ch) > 2 and ch[1].section == "2 B" and ch[2].section.startswith("2 B (part")
    md = chunk_markdown("M-1", "# Title\n\n## 1 Intro\n\ntext here\n\n## 2 More\n\nmore text", doc_type="sop", topics=[], jurisdiction="GLOBAL")
    assert [c.section for c in md] == ["1 Intro", "2 More"]


@pytest.mark.parametrize("query,expected_doc", [
    ("PFAS declaration thresholds for suppliers", "POL-ENV-001"),
    ("lead in solder exemption expiry", "POL-ENV-002"),
    ("new SVHC candidate list screening deadline", "POL-ENV-003"),
    ("what happens when a smelter in the chain is non-conformant", "POL-SRC-004"),
    ("entity list sanctions screening of suppliers", "POL-TRD-006"),
    ("who approves an impact brief before tasks are issued", "POL-GOV-012"),
])
def test_hybrid_search_finds_the_right_policy(store, query, expected_doc):
    hits = store.search(query, k=3, mode="hybrid")
    assert hits and hits[0].doc_id == expected_doc, [h.citation for h in hits]


def test_filters_scope_results(store):
    hits = store.search("material declarations", k=5, doc_type="contract")
    assert hits and all(h.meta["doc_type"] == "contract" for h in hits)
    sid = hits[0].meta["supplier_id"]
    scoped = store.search("notification of material change", k=5, supplier_id=sid)
    assert scoped and all(h.meta["supplier_id"] == sid for h in scoped)


def test_delete_document_removes_all_chunks(store):
    assert store.count_document("POL-ENV-009") > 0
    removed = store.delete_document("POL-ENV-009")
    assert removed > 0 and store.count_document("POL-ENV-009") == 0
    hits = store.search("EPS foam PVC packaging phase out", k=5)
    assert all(h.doc_id != "POL-ENV-009" for h in hits)
