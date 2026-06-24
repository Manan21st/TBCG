"""Conditional routing functions for the consulting graph.

Routing is data-driven: the Engagement Manager populates ``required_agents``
and these functions translate that into graph paths. Keeping them here keeps
``build.py`` declarative.
"""

from __future__ import annotations

from ..state import ConsultingState

ANALYSIS_AGENTS = ["research", "finance"]


def route_after_engagement(state: ConsultingState) -> list[str]:
    """Fan out to whichever analysis agents the engagement plan requires.

    Returns a list so LangGraph runs them in parallel; both edges converge on
    the strategy node. If neither is needed, go straight to strategy.
    """
    required = state.get("required_agents", [])
    analysis = [a for a in ANALYSIS_AGENTS if a in required]
    return analysis or ["strategy"]


def route_after_strategy(state: ConsultingState) -> str:
    """Run risk assessment when required, otherwise straight to the critic."""
    return "risk" if "risk" in state.get("required_agents", []) else "critic"


def route_after_critic(state: ConsultingState) -> str:
    """Approved → move on; rejected → revise (the critic enforces the cap)."""
    return "approved" if state.get("approved") else "revise"


def route_after_human(state: ConsultingState) -> str:
    """Human approved → report; rejected → revision cycle."""
    return "report" if state.get("human_approved") else "revise"


def route_after_revision(state: ConsultingState) -> str:
    """Re-enter the pipeline at the stage the dispatcher chose; everything
    downstream re-runs via the existing edges. Defaults to strategy."""
    return state.get("revision_target") or "strategy"
