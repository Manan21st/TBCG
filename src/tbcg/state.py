"""Shared LangGraph state for the consulting workflow."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class ConsultingState(TypedDict, total=False):
    # Input
    query: str

    # Engagement manager
    problem_type: str
    required_agents: list[str]
    # Ethics/legality advisory when the request raises conduct concerns ('' if none)
    conduct_advisory: str

    # Specialist outputs (validated dicts)
    research_findings: dict[str, Any]
    financial_analysis: dict[str, Any]
    strategy_recommendation: dict[str, Any]
    risk_assessment: dict[str, Any]

    # Critic
    critic_feedback: str
    approved: bool

    # Human-in-the-loop gate (None = pending / not yet reached)
    human_approved: bool | None

    # Control
    revision_count: int
    # Which stage a revision re-enters the pipeline at (set by the dispatcher)
    revision_target: str

    # Running trace for UI / debugging (append-only)
    messages: Annotated[list, add_messages]

    # Final deliverable
    final_report: str


def initial_state(query: str) -> ConsultingState:
    """Build a fresh state for a new engagement."""
    return ConsultingState(
        query=query,
        problem_type="",
        required_agents=[],
        conduct_advisory="",
        research_findings={},
        financial_analysis={},
        strategy_recommendation={},
        risk_assessment={},
        critic_feedback="",
        approved=False,
        human_approved=None,
        revision_count=0,
        revision_target="",
        messages=[],
        final_report="",
    )
