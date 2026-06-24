"""Strategy confidence-cap guardrail for high-stakes investment decisions."""

from unittest.mock import patch

from tbcg.agents import strategy
from tbcg.schemas import StrategyRecommendation


def _fake_rec(confidence):
    return StrategyRecommendation(
        recommendation="Acquire the target.",
        alternatives=["Do nothing"],
        confidence=confidence,
        rationale="Strong synergy.",
    )


def test_investment_confidence_is_capped():
    with patch.object(strategy, "structured_call", return_value=_fake_rec(0.85)):
        out = strategy.strategy_node({"query": "acquire X", "problem_type": "investment"})
    rec = out["strategy_recommendation"]
    assert rec["confidence"] == 0.6
    assert "due diligence" in rec["rationale"].lower()


def test_non_investment_confidence_untouched():
    with patch.object(strategy, "structured_call", return_value=_fake_rec(0.85)):
        out = strategy.strategy_node({"query": "reduce costs", "problem_type": "cost_reduction"})
    assert out["strategy_recommendation"]["confidence"] == 0.85
