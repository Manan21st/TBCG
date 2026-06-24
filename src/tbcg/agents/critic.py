"""Critic Agent — independent quality reviewer.

Approves the recommendation or requests a revision. The graph caps how many
revision cycles can run (see config.MAX_REVISIONS).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from ..config import get_settings
from ..llm import structured_call
from ..schemas import CriticVerdict
from ..state import ConsultingState

SYSTEM = """You are an independent Critic at TBCG, reviewing a colleague's work
before it reaches the client. Approve ONLY if the recommendation is:
- supported by the analysis that was actually performed (no unsupported leaps),
- internally consistent (strategy vs. risk vs. numbers),
- specific and actionable.

IMPORTANT — scope your review to the specialists that were engaged for this
problem (listed below). Only request revisions THIS team can act on:
- If finance was NOT engaged, do NOT demand financial projections/ROI — and
  flag any numbers the strategist invented without a finance basis as
  unsupported (they should be removed or hedged, not "supported" by analysis
  that doesn't exist).
- If risk was NOT engaged, do NOT demand a risk assessment.
- If a missing analysis is genuinely material, note it as a brief caveat, but
  do NOT block approval over work that was never scoped.

NUMERIC SANITY — reject figures that are implausible or self-contradictory:
- Revenue projections wildly out of scale with the company's actual revenue, a
  break-even that contradicts the ROI, or impossible values.
PROVENANCE — flag claims that misuse evidence:
- Region/segment-specific facts presented as truths about a different market,
  or conclusions leaning on stale/weak sources. These must be hedged or dropped.
Be demanding but fair. Give precise, actionable feedback the team can act on."""


def critic_node(state: ConsultingState) -> dict:
    settings = get_settings()
    engaged = state.get("required_agents", [])
    parts = [
        f"Client problem:\n{state['query']}",
        f"Specialists engaged for this problem: {', '.join(engaged) or 'strategy only'}",
    ]
    for key, label in [
        ("research_findings", "Research"),
        ("financial_analysis", "Finance"),
        ("strategy_recommendation", "Strategy recommendation"),
        ("risk_assessment", "Risk assessment"),
    ]:
        if state.get(key):
            parts.append(f"{label}:\n{state[key]}")

    verdict = structured_call(CriticVerdict, system=SYSTEM, user="\n\n".join(parts))

    revision_count = state.get("revision_count", 0)
    # If we've hit the cap, force-approve so we don't loop forever.
    forced = False
    approved = verdict.approved
    if not approved and revision_count >= settings.max_revisions:
        approved = True
        forced = True

    note = " (revision cap reached — forwarding with caveats)" if forced else ""
    msg = AIMessage(
        content=(
            f"[Critic] {'APPROVED' if approved else 'REVISION REQUESTED'}{note}. "
            f"{verdict.feedback}"
        )
    )
    return {
        "approved": approved,
        "critic_feedback": verdict.feedback,
        "messages": [msg],
    }
