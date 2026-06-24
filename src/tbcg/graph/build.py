"""Assemble the consulting workflow as a LangGraph StateGraph.

Flow:
    engagement
       └─(fan-out)─> research / finance ──> strategy
    strategy ──> risk (if required) ──> critic
    critic ──(approved)──> human_review ──(approved)──> report ──> END
           └─(rejected)──> revise ──(dispatcher picks entry stage)──> research /
             finance / strategy / risk ──> ... ──> critic   (capped by MAX_REVISIONS)
    human_review ──(rejected)──> revise

Set ``human_in_the_loop=False`` (e.g. for tests/eval) to skip the human gate;
the critic's approval then flows straight to the report.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from ..agents.critic import critic_node
from ..agents.engagement import engagement_node
from ..agents.finance import finance_node
from ..agents.research import research_node
from ..agents.risk import risk_node
from ..agents.strategy import strategy_node
from ..llm import structured_call
from ..schemas import RevisionPlan
from ..state import ConsultingState
from ..tools.report import build_markdown_report
from . import routing

REVISION_SYSTEM = """A revision was requested on consulting work. The pipeline is
a dependency chain: research → finance → strategy → risk → critic. Decide the
EARLIEST stage whose output must actually change to address the feedback —
everything after it re-runs automatically, everything before it is preserved.

- research: the evidence/market facts are wrong, missing, or need more depth.
- finance: the numbers, ROI, or financial assumptions need to change.
- strategy: the recommendation itself — or only its depth/wording/structure —
  needs to change or be expanded (the evidence and numbers are fine).
- risk: only the risk assessment needs work; the recommendation stands.

Pick the single best entry point. If the feedback only asks to expand, clarify,
or restructure one stage, target THAT stage alone — do not re-run earlier stages
whose outputs are not in question."""

# Dependency order; used to clamp the dispatcher's choice to an engaged stage.
_STAGE_ORDER = ["research", "finance", "strategy", "risk"]


def _revise_node(state: ConsultingState) -> dict:
    """Dispatch the revision: pick the earliest stage that must change.

    Model-driven (interprets the feedback), not keyword-based. Downstream stages
    re-run via the existing edges; upstream stages are preserved.
    """
    feedback = state.get("critic_feedback", "")
    team = [a for a in state.get("required_agents", []) if a != "strategy"]
    rec = (state.get("strategy_recommendation") or {}).get("recommendation", "")

    target = "strategy"
    reason = "default re-synthesis"
    try:
        plan = structured_call(
            RevisionPlan,
            system=REVISION_SYSTEM,
            user=(
                f"Feedback:\n{feedback}\n\n"
                f"Stages that ran (besides strategy): {team or 'none'}\n"
                f"Current recommendation: {rec}"
            ),
            fast=True,
        )
        target, reason = plan.target, plan.reason
    except Exception:  # noqa: BLE001 - dispatch is best-effort; fall back to strategy
        pass

    # Clamp to a stage that actually ran (strategy always runs); never re-enter
    # at a stage that wasn't part of this engagement.
    if target != "strategy" and target not in state.get("required_agents", []):
        target = "strategy"

    return {
        "revision_count": state.get("revision_count", 0) + 1,
        "revision_target": target,
        "messages": [AIMessage(content=f"[Revision] Re-running from '{target}' — {reason}")],
    }


def _human_review_node(state: ConsultingState) -> dict:
    """Pause for human approval of high-impact recommendations.

    ``interrupt`` suspends the graph and surfaces the recommendation to the
    caller (CLI/Streamlit), who resumes with ``Command(resume={...})``.
    """
    decision = interrupt(
        {
            "type": "human_review",
            "query": state.get("query"),
            "recommendation": state.get("strategy_recommendation"),
            "risk_assessment": state.get("risk_assessment"),
            "critic_feedback": state.get("critic_feedback"),
        }
    )
    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
        feedback = decision.get("feedback", "")
    else:
        approved = bool(decision)
        feedback = ""

    update: dict = {"human_approved": approved}
    if not approved and feedback:
        update["critic_feedback"] = feedback
    return update


def _report_node(state: ConsultingState) -> dict:
    return {"final_report": build_markdown_report(dict(state))}


def build_graph(human_in_the_loop: bool = True, checkpointer=None):
    """Compile and return the consulting workflow graph."""
    g = StateGraph(ConsultingState)

    g.add_node("engagement", engagement_node)
    g.add_node("research", research_node)
    g.add_node("finance", finance_node)
    g.add_node("strategy", strategy_node)
    g.add_node("risk", risk_node)
    g.add_node("critic", critic_node)
    g.add_node("revise", _revise_node)
    g.add_node("report", _report_node)

    g.set_entry_point("engagement")

    # Engagement fans out to the required analysis agents (or straight to strategy).
    g.add_conditional_edges(
        "engagement", routing.route_after_engagement, ["research", "finance", "strategy"]
    )
    g.add_edge("research", "strategy")
    g.add_edge("finance", "strategy")

    # Strategy -> risk (if required) -> critic
    g.add_conditional_edges(
        "strategy", routing.route_after_strategy, {"risk": "risk", "critic": "critic"}
    )
    g.add_edge("risk", "critic")

    # Revision loop: the dispatcher (in _revise_node) chose the earliest stage
    # that must change; re-enter there and let the existing edges re-run
    # everything downstream. Upstream stages are preserved (their state is kept).
    g.add_conditional_edges(
        "revise",
        routing.route_after_revision,
        {"research": "research", "finance": "finance", "strategy": "strategy", "risk": "risk"},
    )
    g.add_edge("report", END)

    if human_in_the_loop:
        g.add_node("human_review", _human_review_node)
        g.add_conditional_edges(
            "critic",
            routing.route_after_critic,
            {"approved": "human_review", "revise": "revise"},
        )
        g.add_conditional_edges(
            "human_review",
            routing.route_after_human,
            {"report": "report", "revise": "revise"},
        )
    else:
        g.add_conditional_edges(
            "critic",
            routing.route_after_critic,
            {"approved": "report", "revise": "revise"},
        )

    return g.compile(checkpointer=checkpointer or MemorySaver())
