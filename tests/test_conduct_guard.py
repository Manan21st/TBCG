"""Conduct/ethics guardrail: flagged requests force risk + carry an advisory."""

from unittest.mock import patch

from tbcg.agents import engagement
from tbcg.schemas import ConductScreen, EngagementPlan


def _fake_calls(flagged):
    plan = EngagementPlan(
        problem_type="cost_reduction",
        required_agents=["finance"],
        risk_material=False,
        rationale="Cost focus.",
    )
    screen = ConductScreen(
        flagged=flagged,
        concern="Obstructing lawful refunds may breach consumer-protection law." if flagged else "",
    )

    def fake(schema, *a, **k):
        return screen if schema is ConductScreen else plan

    return fake


def test_flagged_request_forces_risk_and_advisory():
    with patch.object(engagement, "structured_call", side_effect=_fake_calls(True)):
        out = engagement.engagement_node({"query": "make refunds harder to deny customers"})
    assert "risk" in out["required_agents"]
    assert "CONDUCT ALERT" in out["conduct_advisory"]


def test_clean_request_has_no_advisory():
    with patch.object(engagement, "structured_call", side_effect=_fake_calls(False)):
        out = engagement.engagement_node({"query": "reduce our shipping costs"})
    assert out["conduct_advisory"] == ""


def test_risk_material_forces_risk_agent():
    from tbcg.schemas import ConductScreen, EngagementPlan

    plan = EngagementPlan(
        problem_type="churn",  # floor is just [research]
        required_agents=["research"],
        risk_material=True,  # but the model flagged risk as material
        rationale="Concentration risk.",
    )
    screen = ConductScreen(flagged=False, concern="")

    def fake(schema, *a, **k):
        return screen if schema is ConductScreen else plan

    with patch.object(engagement, "structured_call", side_effect=fake):
        out = engagement.engagement_node({"query": "are we too dependent on two partners?"})
    assert "risk" in out["required_agents"]
