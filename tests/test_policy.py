"""Problem-policy tests — single source of truth for scoping."""

import typing

from tbcg.policy import (
    PROBLEM_POLICY,
    detect_sanctioned_jurisdictions,
    jurisdiction_advisory,
    required_analysis_agents,
)
from tbcg.schemas import ProblemType


def test_policy_covers_every_problem_type():
    declared = set(typing.get_args(ProblemType))
    assert declared == set(PROBLEM_POLICY), "PROBLEM_POLICY must cover all problem types"


def test_market_entry_and_compare_require_full_analysis():
    for pt in ("market_entry", "market_compare"):
        floor = set(required_analysis_agents(pt))
        assert {"research", "finance", "risk"} <= floor


def test_cost_reduction_requires_finance():
    assert "finance" in required_analysis_agents("cost_reduction")


def test_general_has_no_floor():
    assert required_analysis_agents("general") == []


def test_strategy_never_in_floor():
    # strategy is implicit (appended by engagement), never part of the floor.
    for agents in PROBLEM_POLICY.values():
        assert "strategy" not in agents


def test_detect_sanctioned_jurisdictions():
    assert detect_sanctioned_jurisdictions("what about entering Russia?") == ["Russia"]
    assert "Iran" in detect_sanctioned_jurisdictions("Iran and the UK")
    assert detect_sanctioned_jurisdictions("should we enter Spain?") == []


def test_jurisdiction_advisory():
    assert "sanctions" in jurisdiction_advisory("enter the Russian market").lower()
    assert jurisdiction_advisory("enter Japan") == ""
