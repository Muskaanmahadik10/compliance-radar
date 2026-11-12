"""Settings. Two runtime modes share one code path:

  RADAR_MODE=local    embedded Qdrant on disk, fakeredis (Streams supported), SQLite. No Docker needed.
  RADAR_MODE=compose  Qdrant, Redis, Postgres (and Langfuse) from docker-compose or CI service containers.

Everything secret comes from environment variables or .env; nothing is hard-coded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
LOCAL_STATE_DIR = ROOT / ".local"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore", env_prefix="RADAR_")

    mode: Literal["local", "compose"] = "local"

    # infrastructure (compose mode)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+psycopg://radar:radar@localhost:5432/radar"
    # infrastructure (local mode)
    local_qdrant_path: Path = LOCAL_STATE_DIR / "qdrant"
    local_sqlite_path: Path = LOCAL_STATE_DIR / "radar.sqlite"

    # retrieval
    collection: str = "company_docs"
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # CI/laptop default; set BAAI/bge-m3 for multilingual demo
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"  # set BAAI/bge-reranker-v2-m3 for multilingual demo
    rerank: bool = True

    # LLMs
    llm_provider: Literal["mock", "openai_compatible", "anthropic"] = "mock"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    anthropic_api_key: str = ""
    cheap_model: str = "openai/gpt-oss-20b"
    strong_model: str = "openai/gpt-oss-120b"
    max_cost_per_brief_usd: float = 0.10
    # Output budget per call. Prompt + this must fit the provider's per-request token limit (8k TPM on Groq's free tier).
    max_output_tokens: int = 3000

    # observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # ingestion
    feed_poll_seconds: int = 300
    allowed_fetch_domains: list[str] = ["eur-lex.europa.eu", "www.federalregister.gov", "www.moea.gov.tw", "www.bsmi.gov.tw"]

    data_dir: Path = DATA_DIR
    artifacts_dir: Path = ARTIFACTS_DIR

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings()
