"""End-to-end graph execution with the LLM + tools stubbed (no credits, no net).

Verifies the wiring the unit tests can't: engagement fan-out, fan-in to
strategy, conditional risk edge, the revision loop + cap, the human-in-the-loop
interrupt/resume, and final report assembly.
"""

import uuid

import pytest
from langgraph.types import Command

from tbcg.schemas import (
    ConductScreen,
    CriticVerdict,
    EngagementPlan,
    FinancialAnalysis,
    ResearchFindings,
    RevisionPlan,
    RiskAssessment,
    StrategyRecommendation,
)
from tbcg.state import initial_state

FAKE = {
    EngagementPlan: EngagementPlan(
        problem_type="market_entry",
        required_agents=["research", "finance", "risk"],
        risk_material=True,
        rationale="High-stakes entry needs full team.",
    ),
    ConductScreen: ConductScreen(flagged=False, concern=""),
    ResearchFindings: ResearchFindings(
        summary="Market is large.",
        findings=[{"claim": "Big market", "source": "internal-kb", "confidence": 0.8}],
    ),
    FinancialAnalysis: FinancialAnalysis(
        quantitative_applicable=True,
        estimated_roi=0.27,
        break_even_months=18,
        revenue_projection=2_500_000,
        viable=True,
        assumptions=["x"],
        operational_levers=[],
        notes="Derived from the company revenue baseline and margins; revenue from estimated new sales.",
    ),
    StrategyRecommendation: StrategyRecommendation(
        recommendation="Enter via partnership.",
        alternatives=["Direct"],
        confidence=0.81,
        rationale="Eases regulation.",
    ),
    RiskAssessment: RiskAssessment(
        risk_level="medium",
        major_risks=["Regulatory"],
        mitigations=["Counsel"],
    ),
    RevisionPlan: RevisionPlan(target="strategy", reason="re-synthesize"),
}


@pytest.fixture
def stub_agents(monkeypatch):
    """Stub structured_call + tools across all agent modules."""

    def make_fake(critic_verdict):
        def fake_structured_call(schema, system, user, **kwargs):
            if schema is CriticVerdict:
                return critic_verdict
            return FAKE[schema]

        return fake_structured_call

    def install(critic_verdict):
        for mod in ["engagement", "research", "finance", "strategy", "risk", "critic"]:
            monkeypatch.setattr(f"tbcg.agents.{mod}.structured_call", make_fake(critic_verdict))
        # The revision dispatcher lives in build.py and also calls structured_call.
        monkeypatch.setattr("tbcg.graph.build.structured_call", make_fake(critic_verdict))
        # Stub tools used inside agents.
        monkeypatch.setattr("tbcg.agents.research.gather_web", lambda *a, **k: [])
        monkeypatch.setattr("tbcg.agents.research.retrieve", lambda *a, **k: [])
        monkeypatch.setattr("tbcg.agents.finance.gather_web", lambda *a, **k: [])
        monkeypatch.setattr("tbcg.agents.finance.retrieve", lambda *a, **k: [])
        monkeypatch.setattr("tbcg.agents.finance.run_python", lambda code: "roi=0.27")

        class _Reply:
            content = "```python\nprint('roi=0.27')\n```"

        monkeypatch.setattr("tbcg.agents.finance.chat_invoke", lambda messages, **k: _Reply())

    return install


def _build():
    from tbcg.graph.build import build_graph

    return build_graph


def test_full_run_no_hitl_approved_first_pass(stub_agents):
    stub_agents(CriticVerdict(approved=True, feedback="Good."))
    graph = _build()(human_in_the_loop=False)
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}
    final = graph.invoke(initial_state("Should we enter Indonesia?"), config=cfg)

    assert final["problem_type"] == "market_entry"
    assert set(final["required_agents"]) >= {"research", "finance", "risk", "strategy"}
    assert final["research_findings"] and final["financial_analysis"]
    assert final["risk_assessment"] and final["strategy_recommendation"]
    assert "## Strategic Recommendation" in final["final_report"]
    assert final["revision_count"] == 0


def test_revision_loop_respects_cap(stub_agents):
    # Critic always rejects → loop must terminate at the cap, not run forever.
    stub_agents(CriticVerdict(approved=False, feedback="Needs work."))
    graph = _build()(human_in_the_loop=False)
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}
    final = graph.invoke(initial_state("Should we enter Indonesia?"), config=cfg)

    from tbcg.config import get_settings

    assert final["revision_count"] == get_settings().max_revisions
    assert final["final_report"]  # still produced after cap


def test_human_in_the_loop_interrupt_and_resume(stub_agents):
    stub_agents(CriticVerdict(approved=True, feedback="Good."))
    graph = _build()(human_in_the_loop=True)
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}

    # First pass should pause at human_review (no report yet).
    result = graph.invoke(initial_state("Should we enter Indonesia?"), config=cfg)
    assert "__interrupt__" in result

    # Approve → resume to completion.
    final = graph.invoke(Command(resume={"approved": True}), config=cfg)
    assert final.get("human_approved") is True
    assert "## Strategic Recommendation" in final["final_report"]
