"""Centralized configuration: env loading, model registry, feature flags.

The app must boot with only ``HF_TOKEN`` set; everything else has a default.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# HuggingFace tokens look like ``hf_`` followed by ~34 alphanumerics. We detect
# them by VALUE so any variable name works (HF_TOKEN, token1, alternate_…).
_HF_TOKEN_PATTERN = re.compile(r"^hf_[A-Za-z0-9]{20,}$")


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _get_bool(key: str, default: bool = False) -> bool:
    return _get(key, str(default)).lower() in {"1", "true", "yes", "on"}


def _collect_hf_tokens() -> tuple[str, ...]:
    """Gather all HuggingFace tokens for round-robin rotation.

    Sources (in priority order), de-duplicated, blanks dropped:
    - ``HF_TOKENS`` (comma-separated)
    - ``HF_TOKEN``, ``HF_TOKEN_2``, ``HF_TOKEN_3`` ...
    - ANY environment variable whose VALUE is an ``hf_...`` token, regardless of
      its name (so ``token1``, ``alternate_hf_token``, etc. all work).
    """
    candidates: list[str] = []
    candidates += [t for t in _get("HF_TOKENS").split(",")]
    candidates.append(_get("HF_TOKEN"))
    for i in range(2, 10):
        candidates.append(_get(f"HF_TOKEN_{i}"))
    # Value-based detection — robust to any variable name.
    for val in os.environ.values():
        v = val.strip()
        if _HF_TOKEN_PATTERN.match(v):
            candidates.append(v)

    seen: set[str] = set()
    tokens: list[str] = []
    for tok in candidates:
        tok = tok.strip()
        if tok and tok not in seen:
            seen.add(tok)
            tokens.append(tok)
    return tuple(tokens)


@dataclass(frozen=True)
class Settings:
    # --- LLM (HuggingFace Inference Providers, OpenAI-compatible router) ---
    hf_tokens: tuple[str, ...]
    hf_base_url: str
    hf_model: str
    hf_model_fast: str
    hf_temperature: float
    hf_max_tokens: int

    # --- Vector store ---
    chroma_host: str
    chroma_port: int
    chroma_collection: str

    # --- Embeddings ---
    embedding_model: str

    # --- Web search ---
    search_backend: str
    tavily_api_key: str

    # --- Workflow ---
    max_revisions: int

    # --- Observability ---
    langsmith_tracing: bool
    langsmith_project: str
    langsmith_api_key: str
    langsmith_endpoint: str

    @property
    def has_llm(self) -> bool:
        return bool(self.hf_tokens)

    @property
    def hf_token(self) -> str:
        """Primary token (first in the rotation); '' if none configured."""
        return self.hf_tokens[0] if self.hf_tokens else ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings(
        hf_tokens=_collect_hf_tokens(),
        hf_base_url=_get("HF_BASE_URL", "https://router.huggingface.co/v1"),
        hf_model=_get("HF_MODEL", "deepseek-ai/DeepSeek-V3-0324"),
        hf_model_fast=_get("HF_MODEL_FAST", "Qwen/Qwen2.5-72B-Instruct"),
        hf_temperature=float(_get("HF_TEMPERATURE", "0.2")),
        hf_max_tokens=int(_get("HF_MAX_TOKENS", "2048")),
        chroma_host=_get("CHROMA_HOST", "localhost"),
        chroma_port=int(_get("CHROMA_PORT", "8000")),
        chroma_collection=_get("CHROMA_COLLECTION", "tbcg_kb"),
        embedding_model=_get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        search_backend=_get("SEARCH_BACKEND", "ddgs").lower(),
        tavily_api_key=_get("TAVILY_API_KEY"),
        max_revisions=int(_get("MAX_REVISIONS", "2")),
        langsmith_tracing=_get_bool("LANGSMITH_TRACING", False),
        langsmith_project=_get("LANGSMITH_PROJECT", "tbcg"),
        langsmith_api_key=_get("LANGSMITH_API_KEY"),
        langsmith_endpoint=_get("LANGSMITH_ENDPOINT"),
    )

    # Enable tracing across both the modern LANGSMITH_* and legacy LANGCHAIN_*
    # env conventions so it works regardless of langchain/langsmith version.
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        if settings.langsmith_endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

    return settings
