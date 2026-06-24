"""Routing logic tests — pure functions, no LLM/network required."""

from tbcg.graph import routing


def test_fan_out_includes_only_required_analysis_agents():
    state = {"required_agents": ["research", "finance", "strategy"]}
    assert routing.route_after_engagement(state) == ["research", "finance"]


def test_fan_out_research_only():
    state = {"required_agents": ["research", "strategy"]}
    assert routing.route_after_engagement(state) == ["research"]


def test_fan_out_defaults_to_strategy_when_no_analysis():
    state = {"required_agents": ["strategy"]}
    assert routing.route_after_engagement(state) == ["strategy"]


def test_strategy_routes_to_risk_when_required():
    assert routing.route_after_strategy({"required_agents": ["strategy", "risk"]}) == "risk"


def test_strategy_routes_to_critic_without_risk():
    assert routing.route_after_strategy({"required_agents": ["strategy"]}) == "critic"


def test_critic_routing():
    assert routing.route_after_critic({"approved": True}) == "approved"
    assert routing.route_after_critic({"approved": False}) == "revise"


def test_human_routing():
    assert routing.route_after_human({"human_approved": True}) == "report"
    assert routing.route_after_human({"human_approved": False}) == "revise"
