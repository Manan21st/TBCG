"""Local embedding model — runs in-process, consumes no HF credits.

First use downloads the model weights (~80 MB) from HuggingFace; afterwards it
works fully offline.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings


@lru_cache(maxsize=1)
def get_embeddings():
    # Imported lazily so building the graph / running unit tests doesn't require
    # the (heavy) sentence-transformers stack.
    from langchain_huggingface import HuggingFaceEmbeddings

    s = get_settings()
    return HuggingFaceEmbeddings(
        model_name=s.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
