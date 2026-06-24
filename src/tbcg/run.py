"""Headless entry point: run an engagement end-to-end (no human gate).

Usage:
    python -m tbcg.run "Should we expand our wholesale retail partnerships?"
"""

from __future__ import annotations

import sys
import uuid

from .graph.build import build_graph
from .state import initial_state


def run_engagement(query: str, human_in_the_loop: bool = False) -> dict:
    """Run the full workflow and return the final state dict."""
    graph = build_graph(human_in_the_loop=human_in_the_loop)
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    return graph.invoke(initial_state(query), config=config)


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python -m tbcg.run "<business question>"', file=sys.stderr)
        return 1

    query = " ".join(sys.argv[1:])
    final = run_engagement(query)

    print("\n" + "=" * 70)
    for m in final.get("messages", []):
        content = getattr(m, "content", "")
        if content:
            print(content)
    print("=" * 70 + "\n")
    print(final.get("final_report", "(no report produced)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
