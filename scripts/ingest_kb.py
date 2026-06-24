"""CLI: (re)build the vector store from the local knowledge base.

Usage:
    python scripts/ingest_kb.py [path-to-kb]

Defaults to the repo's ``knowledge_base/`` directory. Requires the Chroma
container to be running (``docker compose up -d``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``src`` importable when run directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbcg.rag.ingest import ingest_directory  # noqa: E402
from tbcg.tools.rag import chroma_available  # noqa: E402


def main() -> int:
    kb_dir = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "knowledge_base")

    if not chroma_available():
        print(
            "ERROR: ChromaDB is not reachable. Start it with "
            "`docker compose up -d` and try again.",
            file=sys.stderr,
        )
        return 1

    print(f"Ingesting knowledge base from: {kb_dir}")
    stats = ingest_directory(kb_dir)
    print(f"Done. Files: {stats['files']}, chunks: {stats['chunks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
