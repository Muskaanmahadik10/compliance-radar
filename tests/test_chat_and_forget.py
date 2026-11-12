import pytest

from radar import chat
from radar.forget import forget_document, sign, verify_certificate


@pytest.fixture(scope="module", autouse=True)
def indexed():
    from radar.data.build import build_all
    from radar.retrieval.ingest import ingest

    build_all()
    ingest(recreate=True)


def test_pii_is_redacted_before_model_call():
    text, n = chat.redact_pii("Ask jane.doe@supplier.com or call +886 912 345 678 about PFAS")
    assert n == 2 and "jane.doe" not in text and "912" not in text and "PFAS" in text


def test_chat_answers_with_valid_citations_and_streams():
    res = chat.answer("What PFAS content must suppliers declare?")
    assert res.hits and res.hits[0]["doc_id"] == "POL-ENV-001"
    assert res.citations and not res.invalid_citations
    events = list(chat.stream_answer("How often are suppliers screened against sanctions lists?"))
    assert events[0]["type"] == "sources" and events[-1]["type"] == "done"
    assert any(e["type"] == "token" for e in events)
    assert events[0]["hits"][0]["doc_id"] == "POL-TRD-006"


def test_forget_document_is_verified_and_certified():
    before = chat.answer("By when is EPS foam packaging phased out?", rewrite=False)
    assert any(h["doc_id"] == "POL-ENV-009" for h in before.hits)
    cert = forget_document("POL-ENV-009", requested_by="dpo@company.example", reason="test erasure")
    assert cert["verified"] and all(cert["checks"].values()), cert["checks"]
    assert cert["chunks_removed"] > 0 and sum(cert["retrieved_before"].values()) > 0 and sum(cert["retrieved_after"].values()) == 0
    assert verify_certificate(cert)
    tampered = dict(cert, chunks_removed=0)
    assert not verify_certificate(tampered)
    assert sign({k: v for k, v in cert.items() if k != "signature"}) == cert["signature"]
    after = chat.answer("By when is EPS foam packaging phased out?", rewrite=False)
    assert all(h["doc_id"] != "POL-ENV-009" for h in after.hits)
    with pytest.raises(KeyError):
        forget_document("POL-ENV-009", requested_by="x")  # already gone
