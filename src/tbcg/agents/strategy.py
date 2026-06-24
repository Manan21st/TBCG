"""Strategy Consultant Agent — synthesizes all inputs into a recommendation.

Reads research + finance from state and, on a revision pass, incorporates the
critic's feedback.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from ..llm import structured_call
from ..schemas import StrategyRecommendation
from ..state import ConsultingState

SYSTEM = """You are a Strategy Consultant at TBCG. Synthesize the available
research and financial analysis into a single, actionable recommendation.
Compare the main strategic alternatives, then commit to one. Be specific and
decision-ready. Set confidence (0-1) honestly based on evidence strength.
If revision feedback is provided, address it directly. When the feedback asks to
EXPAND, clarify, or restructure, add depth and specificity while PRESERVING your
recommendation and any figures from the financial analysis — do not reverse the
call unless the feedback gives a substantive reason to.

DO NOT FABRICATE NUMBERS. Use only financial figures (ROI, costs, revenue,
break-even, investment amounts, timelines) that appear in the provided
financial analysis. If a figure was not produced by the finance team, do not
invent one — make the point qualitatively instead. Citing made-up numbers is a
serious error.

If the financial analysis indicates the venture never breaks even (e.g. a
break-even of 600 months or $0 revenue), describe it as "does not break even
within a viable horizon" — do NOT translate the sentinel into a literal figure
like "50 years".

For irreversible, high-stakes commitments (acquisitions, large investments,
major partnerships): treat any third-party figures (e.g. a target's revenue)
that aren't in the evidence as UNVERIFIED assumptions, make due diligence an
explicit prerequisite, and CAP confidence at ~0.6 until the key numbers are
validated. Do not give a confident "go" on unverified data."""


def strategy_node(state: ConsultingState) -> dict:
    parts = [f"Client problem ({state.get('problem_type', 'general')}):\n{state['query']}"]
    if state.get("conduct_advisory"):
        parts.insert(0, state["conduct_advisory"])
    if state.get("research_findings"):
        parts.append(f"Research findings:\n{state['research_findings']}")
    if state.get("financial_analysis"):
        parts.append(f"Financial analysis:\n{state['financial_analysis']}")
    if state.get("critic_feedback"):
        parts.append(
            "Critic feedback to address (this is a revision):\n" f"{state['critic_feedback']}"
        )

    rec = structured_call(StrategyRecommendation, system=SYSTEM, user="\n\n".join(parts))

    # Deterministic guardrail: high-stakes "investment" decisions in this system
    # hinge on third-party data we cannot verify, so the model's confidence is
    # not trustworthy. Cap it and flag due diligence regardless of what it
    # claimed — prompt-level caps proved unreliable.
    data = rec.model_dump()
    CONF_CAP = 0.6
    if state.get("problem_type") == "investment" and data.get("confidence", 0) > CONF_CAP:
        data["confidence"] = CONF_CAP
        dd = (
            "Confidence is capped pending due diligence: key third-party "
            "figures are unverified assumptions, not confirmed data."
        )
        data["rationale"] = f"{data.get('rationale', '').rstrip()} {dd}".strip()

    msg = AIMessage(
        content=f"[Strategy] {data['recommendation']} (confidence {data['confidence']:.0%})"
    )
    return {"strategy_recommendation": data, "messages": [msg]}
