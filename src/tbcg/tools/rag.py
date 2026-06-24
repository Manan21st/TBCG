"""RAG retrieval against a Dockerized ChromaDB instance.

Embeddings are computed locally (see ``rag.embeddings``); only vector storage
lives in the Chroma container.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ..config import get_settings
from ..rag.embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_chroma_client():
    # Lazy import so building the graph / unit tests don't require chromadb.
    import chromadb

    s = get_settings()
    return chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port)


def get_collection():
    """Get (or create) the knowledge-base collection.

    We manage embeddings ourselves, so the collection is created without an
    embedding function and we always pass precomputed vectors.
    """
    s = get_settings()
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=s.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    """Drop the KB collection so a fresh ingest can't leave stale chunks behind."""
    s = get_settings()
    client = get_chroma_client()
    try:
        client.delete_collection(s.chroma_collection)
    except Exception:  # noqa: BLE001 - fine if it doesn't exist yet
        pass


def chroma_available() -> bool:
    """Cheap health check so callers can degrade gracefully."""
    try:
        get_chroma_client().heartbeat()
        return True
    except Exception:
        return False


def retrieve(query: str, k: int = 4) -> list[dict[str, Any]]:
    """Return up to ``k`` relevant chunks with source + similarity score.

    Returns an empty list if Chroma is unreachable or the store is empty, so
    the Research agent can fall back to web search alone.
    """
    if not chroma_available():
        return []

    collection = get_collection()
    if collection.count() == 0:
        return []

    query_vec = get_embeddings().embed_query(query)
    res = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: list[dict[str, Any]] = []
    for text, meta, dist in zip(docs, metas, dists):
        out.append(
            {
                "text": text,
                "source": (meta or {}).get("source", "internal-kb"),
                # cosine distance -> rough similarity confidence
                "score": round(max(0.0, 1.0 - float(dist)), 3),
            }
        )
    return out
