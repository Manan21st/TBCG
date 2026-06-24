"""Risk Analyst Agent — stress-tests the proposed strategy."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from ..llm import structured_call
from ..policy import jurisdiction_advisory
from ..schemas import RiskAssessment
from ..state import ConsultingState

SYSTEM = """You are a Risk Analyst at TBCG. Critically examine the proposed
strategy and surface the most material risks: regulatory, competitive,
operational, and financial. Rate the overall risk_level (low/medium/high) and
pair each major risk with a concrete mitigation. Be specific to this case, not
generic.

If a JURISDICTION ALERT is present, sanctions/legal-exposure is the #1 risk:
list it first, rate the engagement HIGH risk, and make clear that financial
upside cannot offset an unlawful or sanctioned market entry."""


def risk_node(state: ConsultingState) -> dict:
    parts = [f"Client problem ({state.get('problem_type', 'general')}):\n{state['query']}"]
    advisory = jurisdiction_advisory(state["query"])
    if advisory:
        parts.append(advisory)
    if state.get("conduct_advisory"):
        parts.append(state["conduct_advisory"])
    if state.get("strategy_recommendation"):
        parts.append(f"Proposed strategy:\n{state['strategy_recommendation']}")
    if state.get("research_findings"):
        parts.append(f"Research findings:\n{state['research_findings']}")
    if state.get("financial_analysis"):
        parts.append(f"Financial analysis:\n{state['financial_analysis']}")

    risk = structured_call(RiskAssessment, system=SYSTEM, user="\n\n".join(parts))

    msg = AIMessage(
        content=(
            f"[Risk] Overall {risk.risk_level.upper()} — "
            f"{len(risk.major_risks)} major risks identified."
        )
    )
    return {"risk_assessment": risk.model_dump(), "messages": [msg]}
