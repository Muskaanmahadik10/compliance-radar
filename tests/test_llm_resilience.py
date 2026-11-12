"""The retry/fallback policy is exercised with fake models, so no provider is needed."""

import pytest
from pydantic import BaseModel

from radar.llm import router
from radar.llm.router import _retry_after_seconds, _short_error, call_structured


class Out(BaseModel):
    value: str


class FakeModel:
    """Minimal stand-in: `with_structured_output` returns a runner that raises the queued errors, then answers."""

    _llm_type = "fake"

    def __init__(self, name: str, errors: list[Exception] | None = None):
        self.model_name, self.errors, self.calls = name, list(errors or []), 0

    def with_structured_output(self, schema, **kwargs):
        outer = self

        class Runner:
            def invoke(self, messages, config=None):
                outer.calls += 1
                if outer.errors:
                    raise outer.errors.pop(0)
                return schema(value=outer.model_name)

        return Runner()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(router.time, "sleep", lambda s: None)


def test_retry_after_parsing():
    assert _retry_after_seconds("Please try again in 28m1.344s.") == pytest.approx(1681.344)
    assert _retry_after_seconds("try again in 697ms") == pytest.approx(0.697)
    assert _retry_after_seconds("try again in 1h23m31.2s") == pytest.approx(5011.2)
    assert _retry_after_seconds("no hint here") is None


def test_formatting_error_is_retried_on_the_same_model():
    m = FakeModel("strong", [ValueError("Tool choice is required, but model did not call a tool")])
    out = call_structured(m, Out, [], attempts=3)
    assert out.value == "strong" and m.calls == 2 and out._model_used == "strong"


def test_daily_quota_goes_straight_to_fallback_model():
    strong = FakeModel("strong", [RuntimeError("Error code: 429 - Rate limit reached on tokens per day (TPD). Please try again in 28m1.3s.")] * 3)
    cheap = FakeModel("cheap")
    out = call_structured(strong, Out, [], attempts=3, fallback=cheap)
    assert out.value == "cheap" and out._model_used == "cheap"
    assert strong.calls == 1  # no pointless retries against a quota that clears in half an hour


def test_per_minute_limit_is_waited_out():
    strong = FakeModel("strong", [RuntimeError("429 rate limit reached on tokens per minute. Please try again in 7.5s.")])
    out = call_structured(strong, Out, [], attempts=3, fallback=FakeModel("cheap"))
    assert out.value == "strong" and strong.calls == 2


def test_oversized_request_is_not_retried_and_error_is_sanitised():
    strong = FakeModel("strong", [RuntimeError("413 Request too large for model in organization org_01abcXYZ: reduce your message size")] * 3)
    with pytest.raises(RuntimeError) as e:
        call_structured(strong, Out, [], attempts=3)
    assert strong.calls == 1 and "too large" in str(e.value)
    assert "org_01abcXYZ" not in _short_error(RuntimeError("429 rate limit in organization org_01abcXYZ"))
