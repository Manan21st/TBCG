"""Engagement Manager — the senior partner.

Classifies the business problem and decides which specialists are needed.
Uses the fast/cheap model to conserve credits.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from ..llm import structured_call
from ..policy import ANALYSIS_ORDER, required_analysis_agents
from ..schemas import ConductScreen, EngagementPlan
from ..state import ConsultingState

CONDUCT_SYSTEM = """You are a compliance and ethics screener at a consulting
firm. Decide whether the client's request seeks action that is illegal,
deceptive, consumer-harmful, anticompetitive, privacy-violating, or otherwise
unethical (e.g. obstructing lawful refunds, dark patterns, collusion, deceiving
customers). Normal aggressive-but-lawful business tactics are NOT flagged. When
flagged, state the specific concern in one sentence."""

SYSTEM = """You are the Engagement Manager at TBCG, a strategy consulting firm.
Read the client's business problem and:
1. Classify it into exactly one problem_type.
2. Choose the specialist agents required to solve it, in execution order.

problem_type guide:
- market_entry: entering a new country/market.
- product_launch: launching a new product or feature.
- cost_reduction: cutting operational cost / improving efficiency.
- churn: customers leaving / retention problems (root-cause analysis).
- market_compare: choosing between several markets/segments to target.
- pricing: changing prices, discounts, packaging, or monetization.
- investment: acquisitions/M&A, large capital outlays, major partnerships, or
  any high-stakes, hard-to-reverse commitment.
- general: open/qualitative questions that fit none of the above.

Classify high-stakes or financially material decisions into a specific type
(pricing, investment, market_entry, ...), NOT "general" — "general" is only for
genuinely open or qualitative questions.

Available specialists: research, finance, strategy, risk.
- Always include "strategy" (it synthesizes the recommendation).
- Include "research" when external/market evidence matters.
- Include "finance" when numbers (ROI, cost, revenue) matter.
- Include "risk" for high-stakes commitments (market entry, large launches).
Set risk_material=true whenever the decision has material downside or the
question itself is about risk, threats, dependence/concentration, resilience, or
preparing for adverse scenarios (e.g. a recession) — risk will then be engaged
regardless of problem_type.
Keep the team as small as the problem allows."""


def engagement_node(state: ConsultingState) -> dict:
    plan = structured_call(
        EngagementPlan,
        system=SYSTEM,
        user=f"Client problem:\n{state['query']}",
        fast=True,
    )

    # Union the model's pick with the deterministic floor (single source of
    # truth in policy.py) so routing is robust to model judgment.
    chosen = {a for a in plan.required_agents if a != "strategy"}
    chosen |= set(required_analysis_agents(plan.problem_type))

    # Force risk whenever the decision is risk-material, even if the problem_type
    # floor (or the model's pick) omitted it. Risk questions must get a risk lens.
    if plan.risk_material:
        chosen.add("risk")

    # Ethics/legality screen — if the request raises conduct concerns, force risk
    # onto the team and carry an advisory for the downstream agents to honor.
    conduct_advisory = ""
    screen = structured_call(
        ConductScreen, system=CONDUCT_SYSTEM, user=f"Client request:\n{state['query']}", fast=True
    )
    if screen.flagged:
        chosen.add("risk")
        conduct_advisory = (
            f"CONDUCT ALERT — {screen.concern} Treat legal/ethical exposure as a "
            "primary consideration. Only recommend lawful, ethical actions; surface "
            "the risk explicitly and steer toward a compliant alternative."
        )

    # Stable ordering, with strategy always last.
    agents = [a for a in ANALYSIS_ORDER if a in chosen]
    agents.append("strategy")

    note = f"Team: {', '.join(agents)}. {plan.rationale}"
    if conduct_advisory:
        note += " [conduct concern flagged — risk engaged]"
    msg = AIMessage(content=f"[Engagement Manager] Classified as **{plan.problem_type}**. {note}")
    return {
        "problem_type": plan.problem_type,
        "required_agents": agents,
        "conduct_advisory": conduct_advisory,
        "messages": [msg],
    }
