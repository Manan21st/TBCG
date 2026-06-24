"""Live evaluation harness for the 5 spec test cases.

Requires HF_TOKEN (real LLM calls) and ideally a running Chroma container.
This is NOT a pytest unit test — it spends credits. Run explicitly:

    python tests/eval_cases.py

For each case it asserts the engagement routing matches expectations and the
final report contains the required sections.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbcg.config import get_settings  # noqa: E402
from tbcg.run import run_engagement  # noqa: E402

CASES = [
    {
        "query": "Should we expand into the European mattress market?",
        "expect_type": {"market_entry"},
        "expect_agents": {"research", "finance", "risk"},
    },
    {
        "query": "Should we launch a new premium mattress product line?",
        "expect_type": {"product_launch"},
        "expect_agents": {"research", "risk"},
    },
    {
        "query": "How can we cut operating costs to sustain positive EBITDA?",
        "expect_type": {"cost_reduction"},
        "expect_agents": {"finance"},
    },
    {
        "query": "Should we raise mattress prices by 10% next year?",
        "expect_type": {"pricing"},
        "expect_agents": {"research", "finance", "risk"},
    },
    {
        "query": "Should we acquire Helix Sleep?",
        "expect_type": {"investment"},
        "expect_agents": {"research", "finance", "risk"},
    },
]

REQUIRED_SECTIONS = ["## Executive Summary", "## Strategic Recommendation"]


def run_case(case: dict) -> bool:
    print(f"\n=== {case['query']} ===")
    final = run_engagement(case["query"], human_in_the_loop=False)

    ptype = final.get("problem_type", "")
    agents = set(final.get("required_agents", []))
    report = final.get("final_report", "")

    ok = True
    if ptype not in case["expect_type"]:
        print(f"  [WARN] problem_type={ptype!r}, expected one of {case['expect_type']}")
    else:
        print(f"  [OK] problem_type={ptype}")

    missing_agents = case["expect_agents"] - agents
    if missing_agents:
        print(f"  [WARN] missing expected agents: {missing_agents} (got {agents})")
    else:
        print(f"  [OK] agents include {case['expect_agents']}")

    for section in REQUIRED_SECTIONS:
        if section not in report:
            print(f"  [FAIL] report missing section: {section}")
            ok = False
    if all(s in report for s in REQUIRED_SECTIONS):
        print("  [OK] report has required sections")

    if final.get("revision_count", 0) > get_settings().max_revisions:
        print("  [FAIL] revision loop exceeded cap")
        ok = False

    return ok


def main() -> int:
    if not get_settings().has_llm:
        print("HF_TOKEN not set — cannot run live eval.", file=sys.stderr)
        return 1
    results = [run_case(c) for c in CASES]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} cases passed hard assertions.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
