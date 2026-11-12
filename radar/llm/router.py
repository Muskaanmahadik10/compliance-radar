"""Model router and cost ledger.

Tiers: 'cheap' (classification, decomposition, tool choice) and 'strong' (drafting, critique). Providers: mock, any
OpenAI-compatible endpoint (Groq, OpenAI, Ollama, vLLM) via langchain-openai, or Anthropic via langchain-anthropic.
Every call's token usage is recorded through a LangChain callback so cost per brief is a real number, not a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel

from radar.config import settings

# USD per 1M tokens (input, output). Edit for your provider; used only for accounting.
PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-20b": (0.10, 0.50), "openai/gpt-oss-120b": (0.15, 0.75), "qwen/qwen3.8-27b": (0.20, 0.60),
    "claude-sonnet-5-5": (3.0, 15.0), "claude-haiku-5-5": (1.0, 5.0), "gpt-4.1-mini": (0.4, 1.6), "mock": (0.0, 0.0),
}


@dataclass
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
                m["calls"] += 1; m["input_tokens"] += i; m["output_tokens"] += o; m["cost_usd"] += cost

    def summary(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "cost_usd": round(self.cost_usd, 6), "by_model": self.by_model}


def _make(model: str, temperature: float = 0.0) -> BaseChatModel:
    if settings.llm_provider == "mock" or (settings.llm_provider == "openai_compatible" and not settings.llm_api_key) or (settings.llm_provider == "anthropic" and not settings.anthropic_api_key):
        from radar.llm.mock import MockChatModel

        return MockChatModel(model_name="mock")
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=settings.anthropic_api_key, temperature=temperature, max_tokens=4000)
    from langchain_openai import ChatOpenAI

    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
    return ChatOpenAI(model=model, api_key=settings.llm_api_key, base_url=settings.llm_base_url, temperature=temperature, max_tokens=4000, max_retries=4, **extra)


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


@lru_cache(maxsize=1)
def get_router() -> ModelRouter:
    return ModelRouter(cheap=_make(settings.cheap_model), strong=_make(settings.strong_model))
