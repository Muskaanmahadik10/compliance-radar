"""Model router and cost ledger.

Tiers: 'cheap' (classification, decomposition, tool choice) and 'strong' (drafting, critique). Providers: mock, any
OpenAI-compatible endpoint (Groq, OpenAI, Ollama, vLLM) via langchain-openai, or Anthropic via langchain-anthropic.
Every call's reported token usage is recorded through a LangChain callback and priced from the table below. That is an
estimate, not a bill: a model missing from PRICES, or a response without usage data, is counted as $0.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import urlparse

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel

from radar.config import settings

# USD per 1M tokens (input, output). Edit for your provider; used only for accounting.
PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-20b": (0.10, 0.50), "openai/gpt-oss-120b": (0.15, 0.75), "qwen/qwen3.8-27b": (0.20, 0.60),
    "claude-sonnet-5-5": (3.0, 15.0), "claude-haiku-5-5": (1.0, 5.0), "gpt-4.1-mini": (0.4, 1.6), "mock": (0.0, 0.0),
}


@dataclass(eq=False)  # callbacks must be hashable; identity equality is what we want
class CostLedger(BaseCallbackHandler):
    """LangChain callback: sums tokens and cost across every model call in a run."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict = field(default_factory=dict)

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for g in gens:
                msg = getattr(g, "message", None)
                usage = getattr(msg, "usage_metadata", None) or {}
                model = (getattr(msg, "response_metadata", {}) or {}).get("model_name") or (getattr(msg, "response_metadata", {}) or {}).get("model") or "unknown"
                i, o = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
                pin, pout = PRICES.get(model, PRICES.get(model.split("/")[-1], (0.0, 0.0)))
                cost = (i * pin + o * pout) / 1e6
                self.calls += 1
                self.input_tokens += i
                self.output_tokens += o
                self.cost_usd += cost
                m = self.by_model.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
                m["calls"] += 1
                m["input_tokens"] += i
                m["output_tokens"] += o
                m["cost_usd"] += cost

    def summary(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "cost_usd": round(self.cost_usd, 6), "by_model": self.by_model}


class MissingCredentials(RuntimeError):
    """A real provider was selected but its key is missing. Raised instead of quietly substituting the mock model."""


def _is_local_endpoint(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1", "host.docker.internal"} or host.endswith(".localhost")


def _make(model: str, temperature: float = 0.0) -> BaseChatModel:
    """Build one chat model. The mock is used only when RADAR_LLM_PROVIDER=mock was chosen explicitly."""
    if settings.llm_provider == "mock":
        from radar.llm.mock import MockChatModel

        return MockChatModel(model_name="mock")
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise MissingCredentials("RADAR_LLM_PROVIDER=anthropic but RADAR_ANTHROPIC_API_KEY is empty. Set the key, or set RADAR_LLM_PROVIDER=mock for offline mode.")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=settings.anthropic_api_key, temperature=temperature, max_tokens=settings.max_output_tokens)
    key = settings.llm_api_key
    if not key:
        if not _is_local_endpoint(settings.llm_base_url):
            raise MissingCredentials("RADAR_LLM_PROVIDER=openai_compatible but RADAR_LLM_API_KEY is empty. Set the key, or set RADAR_LLM_PROVIDER=mock for offline mode.")
        key = "not-needed"  # a keyless local server (Ollama, vLLM) is a legitimate real backend
    from langchain_openai import ChatOpenAI

    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
    return ChatOpenAI(model=model, api_key=key, base_url=settings.llm_base_url, temperature=temperature, max_tokens=settings.max_output_tokens, max_retries=2, **extra)


@dataclass
class ModelRouter:
    cheap: BaseChatModel
    strong: BaseChatModel

    def for_task(self, task: str) -> BaseChatModel:
        """Explicit policy table; logged in traces so cost decisions are auditable."""
        strong_tasks = {"draft_brief", "critic", "chat_answer"}
        return self.strong if task in strong_tasks else self.cheap

    @property
    def names(self) -> dict:
        return {"cheap": getattr(self.cheap, "model_name", None) or getattr(self.cheap, "model", "?"), "strong": getattr(self.strong, "model_name", None) or getattr(self.strong, "model", "?")}


def structured(llm: BaseChatModel, schema):
    """Structured output via tool/function calling, which every provider here supports. (Strict JSON-schema mode rejects
    optional fields and free-form dicts on some OpenAI-compatible servers.)"""
    if llm._llm_type == "mock":
        return llm.with_structured_output(schema)
    return llm.with_structured_output(schema, method="function_calling")


def _short_error(e: Exception) -> str:
    """Exception type plus a short reason, with account identifiers stripped (provider messages can include org ids)."""
    msg = str(e)
    low = msg.lower()
    code = ("request too large for the provider's token limit" if _too_large(msg) else "rate limited" if "429" in msg or "rate limit" in low
            else "bad request" if "400" in msg else "timeout" if "timeout" in low else msg[:80])
    return f"{type(e).__name__}: {re.sub(r'org_[A-Za-z0-9]+', 'org_***', code)}"


def _too_large(msg: str) -> bool:
    low = msg.lower()
    return "413" in msg or "too large" in low or "reduce your message size" in low or "context length" in low


def _retry_after_seconds(msg: str) -> float | None:
    """Parse 'Please try again in 28m1.3s' / '697ms' from a rate-limit message."""
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?(?:(\d+)ms)?", msg)
    if not m or not any(m.groups()):
        return None
    h, mi, sec, ms = m.groups()
    return int(h or 0) * 3600 + int(mi or 0) * 60 + float(sec or 0) + int(ms or 0) / 1000


def call_structured(llm: BaseChatModel, schema, messages: list, config: dict | None = None, attempts: int = 3, fallback: BaseChatModel | None = None):
    """Invoke with structured output, retrying sensibly.

    - Formatting failures (prose instead of a tool call, invalid arguments): short pause, retry.
    - Rate limits: wait only if the provider says the limit clears within a minute (per-minute limits). A daily quota
      that clears in half an hour is not worth blocking a request for.
    - Oversized requests: never retried.
    - If `fallback` is given and the primary model is exhausted or rate limited, the fallback model gets one try.
      The returned object carries `_model_used` so the caller can record who actually wrote it.
    """
    last: Exception | None = None
    for model in [llm] + ([fallback] if fallback is not None and fallback is not llm else []):
        runner = structured(model, schema)
        n = attempts if model is llm else 2
        for i in range(n):
            try:
                out = runner.invoke(messages, config=config)
                if out is not None:
                    try:
                        object.__setattr__(out, "_model_used", getattr(model, "model_name", None) or getattr(model, "model", "?"))
                    except Exception:
                        pass
                    return out
                last = ValueError("model returned no structured output")
            except Exception as e:  # provider 400s, validation errors, timeouts, 429s
                last = e
                msg = str(e)
                if _too_large(msg):
                    break  # retrying an oversized request cannot succeed
                if "429" in msg or "rate limit" in msg.lower():
                    wait = _retry_after_seconds(msg)
                    if wait is None or wait > 60:
                        break  # daily/long quota: go to the fallback model instead of blocking
                    if i < n - 1:
                        time.sleep(min(wait + 1.0, 61.0))
                    continue
            if i < n - 1:
                time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"structured call failed ({_short_error(last)})")


@lru_cache(maxsize=1)
def get_router() -> ModelRouter:
    return ModelRouter(cheap=_make(settings.cheap_model), strong=_make(settings.strong_model))
