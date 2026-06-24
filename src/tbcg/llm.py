"""LLM factory + structured-output helper, with HuggingFace key rotation.

LLMs are reached through HuggingFace Inference Providers, whose router is
OpenAI-compatible, so we drive it with ``ChatOpenAI`` pointed at
``HF_BASE_URL``. Credits are billed against the ``HF_TOKEN`` family.

Multiple tokens are rotated round-robin to spread load/credits, and on a
transient or rate-limit/auth error we fail over to the next token before
giving up. ``structured_call`` is provider-agnostic: it injects the JSON
schema into the prompt, parses the reply into a Pydantic model, and retries on
failure — so it doesn't depend on tool-calling support, which varies across
the models served by the HF router.
"""

from __future__ import annotations

import itertools
import json
import re
import threading
from functools import lru_cache
from typing import TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from .config import get_settings

T = TypeVar("T", bound=BaseModel)

# Round-robin pointer shared across calls so load spreads over all tokens.
_rr_lock = threading.Lock()
_rr_counter = itertools.count()


@lru_cache(maxsize=16)
def _build_llm(model: str, token: str, temperature: float, max_tokens: int) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=model,
        base_url=s.hf_base_url,
        api_key=token,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=120,
        max_retries=2,  # in-client retry for transient connection blips
    )


def _tokens() -> tuple[str, ...]:
    s = get_settings()
    if not s.has_llm:
        raise RuntimeError(
            "No HuggingFace token set. Copy .env.example to .env and add at "
            "least one HF_TOKEN before running the workflow."
        )
    return s.hf_tokens


def get_llm(fast: bool = False, temperature: float | None = None) -> ChatOpenAI:
    """Return a chat model bound to the next token in the rotation.

    ``fast=True`` selects the smaller/cheaper model for lightweight nodes
    (e.g. engagement classification) to conserve credits.
    """
    s = get_settings()
    tokens = _tokens()
    with _rr_lock:
        idx = next(_rr_counter) % len(tokens)
    return _build_llm(
        model=s.hf_model_fast if fast else s.hf_model,
        token=tokens[idx],
        temperature=s.hf_temperature if temperature is None else temperature,
        max_tokens=s.hf_max_tokens,
    )


def chat_invoke(
    messages: list[BaseMessage],
    *,
    fast: bool = False,
    temperature: float | None = None,
) -> BaseMessage:
    """Invoke the LLM, failing over across all tokens on error.

    Starts at the round-robin position and tries each token once; raises only
    if every token fails.
    """
    s = get_settings()
    tokens = _tokens()
    model = s.hf_model_fast if fast else s.hf_model
    temp = s.hf_temperature if temperature is None else temperature

    with _rr_lock:
        start = next(_rr_counter) % len(tokens)

    last_error: Exception | None = None
    for offset in range(len(tokens)):
        token = tokens[(start + offset) % len(tokens)]
        llm = _build_llm(model, token, temp, s.hf_max_tokens)
        try:
            return llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - rotate on any failure
            last_error = exc
            continue

    raise RuntimeError(
        f"All {len(tokens)} HuggingFace token(s) failed for model {model}: {last_error}"
    )


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a model reply (handles code fences)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    match = _JSON_BLOCK.search(text)
    if match:
        return match.group(0)
    return text


def structured_call(
    schema: type[T],
    system: str,
    user: str,
    *,
    fast: bool = False,
    temperature: float | None = None,
) -> T:
    """Call the LLM and return a validated instance of ``schema``.

    Retries once, feeding the validation error back to the model. Each call
    rotates/fails over across tokens via ``chat_invoke``.
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)

    instructions = (
        f"{system}\n\n"
        "Respond with a SINGLE JSON object that validates against this JSON "
        "Schema. Do not include any prose, explanation, or markdown fences "
        "around it — only the JSON object.\n\n"
        f"JSON Schema:\n{schema_json}"
    )

    messages: list[BaseMessage] = [
        SystemMessage(content=instructions),
        HumanMessage(content=user),
    ]
    last_error: Exception | None = None
    max_attempts = 3

    for _ in range(max_attempts):
        reply = chat_invoke(messages, fast=fast, temperature=temperature)
        raw = reply.content if isinstance(reply.content, str) else str(reply.content)
        try:
            return schema.model_validate_json(_extract_json(raw))
        except (ValidationError, json.JSONDecodeError) as exc:
            last_error = exc
            messages.append(reply)
            messages.append(
                HumanMessage(
                    content=(
                        "Your previous response did not validate. Error:\n"
                        f"{exc}\n\nReturn ONLY a corrected JSON object that "
                        "satisfies every constraint."
                    )
                )
            )

    raise ValueError(
        f"structured_call failed to produce valid {schema.__name__} "
        f"after {max_attempts} attempts: {last_error}"
    )
