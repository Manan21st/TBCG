"""Revision dispatcher: re-enters at the model-chosen stage and clamps to an
engaged stage. Verifies the fix for 'strategy feedback flips finance'."""

from unittest.mock import patch

from tbcg.graph import build
from tbcg.graph.routing import route_after_revision
from tbcg.schemas import RevisionPlan


def test_dispatcher_sets_target_from_plan():
    state = {
        "required_agents": ["research", "finance", "risk", "strategy"],
        "critic_feedback": "expand on strategy",
        "strategy_recommendation": {"recommendation": "Enter via partnership."},
        "revision_count": 0,
    }
    with patch.object(
        build, "structured_call", return_value=RevisionPlan(target="strategy", reason="expand only")
    ):
        out = build._revise_node(state)
    assert out["revision_target"] == "strategy"
    assert out["revision_count"] == 1
    assert route_after_revision(out) == "strategy"


def test_dispatcher_targets_finance_when_numbers_questioned():
    state = {
        "required_agents": ["research", "finance", "risk", "strategy"],
        "critic_feedback": "the ROI looks far too optimistic",
        "revision_count": 1,
    }
    with patch.object(
        build, "structured_call", return_value=RevisionPlan(target="finance", reason="numbers")
    ):
        out = build._revise_node(state)
    assert out["revision_target"] == "finance"


def test_dispatcher_clamps_to_engaged_stage():
    # Model picks research, but research wasn't engaged → fall back to strategy.
    state = {
        "required_agents": ["finance", "strategy"],
        "critic_feedback": "tweak the framing",
        "revision_count": 0,
    }
    with patch.object(
        build, "structured_call", return_value=RevisionPlan(target="research", reason="x")
    ):
        out = build._revise_node(state)
    assert out["revision_target"] == "strategy"


def test_dispatcher_falls_back_on_error():
    state = {
        "required_agents": ["research", "strategy"],
        "critic_feedback": "x",
        "revision_count": 0,
    }
    with patch.object(build, "structured_call", side_effect=RuntimeError("LLM down")):
        out = build._revise_node(state)
    assert out["revision_target"] == "strategy"
