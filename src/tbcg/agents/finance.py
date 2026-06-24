"""Financial Analyst Agent (incl. operations analysis).

Two-step: (1) the model writes a short Python script to compute the key
metrics, anchored on the company's REAL financials retrieved from the
knowledge base; we execute it in the REPL. (2) the model turns the computed
numbers into a validated FinancialAnalysis. Schema bounds (see schemas.py)
reject impossible outputs, and the anchors keep projections to a realistic
scale instead of fabricated baselines.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..llm import chat_invoke, structured_call
from ..policy import jurisdiction_advisory
from ..schemas import FinanceDataNeeds, FinancialAnalysis
from ..state import ConsultingState
from ..tools.python_repl import run_python
from ..tools.rag import retrieve
from ..tools.web_search import gather as gather_web

# Query used to pull the company's actual financial baseline from the KB.
# Deliberately generic so it works for any ingested company (SaaS, retailer,
# manufacturer, etc.) — not tied to one business model's metrics.
_ANCHOR_QUERY = (
    "company financials revenue gross margin profit net income operating costs "
    "cash debt guidance unit economics customers channels segments"
)

NEEDS_SYSTEM = """You are a financial analyst planning your data gathering.
Given the client problem, our company's baseline figures, and the research
context, list up to 3 web search queries for EXTERNAL figures you need but do
NOT already have — e.g. a target company's revenue, a market's size, or
region-specific costs. If everything you need is already in the baseline or
research, return an empty list. Be specific (name the company / market / metric)."""

CODE_SYSTEM = """You are a financial analyst. Write a SHORT, self-contained
Python script (standard library only) estimating the key financial metrics for
the client's situation.

GROUND every estimate in the COMPANY BASELINE figures provided (whatever the
company actually reports — revenue, margins, costs, cash, debt, unit economics)
and any EXTERNAL DATA from web search. Derive the new opportunity from them — do
NOT invent a baseline. If a figure you need is not in the baseline, research, or
external data, use a clearly-commented conservative ASSUMPTION (do not pass off a
guessed number as sourced). Rules:
- Define assumptions as commented variables, citing which baseline figure each
  derives from.
- revenue_projection is the GROSS incremental annual revenue from this decision
  (upside only, never negative) and must be realistic versus current total
  revenue (a new initiative is a fraction of revenue in year one, not a multiple
  of it). Put the downside (costs, attrition) into estimated_roi, NOT revenue.
- break_even_months must be a POSITIVE whole number of months (>= 1); use 600
  if it never breaks even.
- estimated_roi is a decimal (0.27 = 27%); it MAY be negative for a bad idea.
- revenue_projection must be in ABSOLUTE dollars (e.g. 21000000 for $21M), NEVER
  in millions (not 21).
Then PRINT estimated_roi, break_even_months, revenue_projection.
Output ONLY a Python code block."""

SYNTH_SYSTEM = """You are a financial analyst at TBCG. Using the client problem,
the company baseline, research context, and the computed figures, produce a
financial assessment that is internally consistent and realistic.

FIRST decide if the decision is quantifiable as a revenue/cost initiative:
- If YES (market entry, pricing, product launch, cost program, acquisition):
  set quantitative_applicable=true and provide estimated_roi, break_even_months,
  and revenue_projection.
- If NO — capital structure / go-private, governance, org, a pure diagnostic
  ("which is the bigger threat"), scenario / contingency planning ("how should
  we prepare for a recession"), or open prioritization ("what's the highest-
  impact move") — set quantitative_applicable=false, LEAVE estimated_roi /
  break_even_months / revenue_projection null, and give the financial
  assessment qualitatively in notes (cost/cash/debt/resilience implications).
  NEVER invent a revenue/break-even number, and do NOT force a $0 "no-go" frame
  onto a question that isn't a yes/no investment decision.

- Keep numbers consistent with the company baseline scale (do not report
  revenue that dwarfs current total revenue, or a break-even that contradicts
  the ROI).
- revenue_projection is GROSS incremental annual USD (upside, >=0) in ABSOLUTE
  dollars (e.g. 21000000 for $21M, never 21). Never clamp a negative net to 0 —
  report the gross upside and let estimated_roi (may be negative) carry downside.
- Set viable=false ONLY for a genuine no-go (sanctions, or economics so poor the
  decision should not proceed); otherwise viable=true. When viable=false, set
  revenue_projection=0 and break_even_months=600.
- break_even_months must be >= 1.
- State every assumption explicitly and tie it to a baseline figure.
- Populate operational_levers with concrete operational/efficiency measures and
  execution steps the strategy depends on (hiring, onboarding, partnerships,
  cost controls) — for any problem type, not only cost reduction.
- When risks are flagged, reflect them in the assumptions and projections.
- Prefer figures from EXTERNAL DATA (web search) over guesses. If a pivotal
  input is still NOT available in baseline, research, or external data (e.g. a
  private target's revenue that could not be found), treat it as an explicit
  ASSUMED figure, stay conservative, and say in notes that it could not be
  sourced, the result is sensitive to it, and due diligence is required — do not
  present it as established fact.
- In "notes", explain HOW you got the numbers: the method, which baseline
  figures you used (the company's actual revenue, margins, costs, unit
  economics), and the key drivers of the ROI / break-even / revenue. This is the
  reasoning a partner will read.
- If a JURISDICTION ALERT is present, model LAWFULLY realizable revenue. Unless a
  compliant, lawful path is clearly established, treat realizable revenue as ~0
  and break_even_months as 600 (does not break even) — do not report an
  unconstrained commercial upside that sanctions make unrealizable."""

_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    m = _CODE_FENCE.search(text)
    return m.group(1).strip() if m else text.strip()


def _format_anchors(facts: list[dict]) -> str:
    if not facts:
        return "(no internal financial baseline found — state assumptions explicitly)"
    return "\n".join(f"- [{f['source']}] {f['text']}" for f in facts)


def _gather_external_data(query: str, anchors: str, research: dict) -> str:
    """Search the web for external figures finance needs but doesn't have.

    Returns formatted evidence, or a clear note that the gap couldn't be filled
    — so the model flags an assumption rather than fabricating a sourced fact.
    """
    try:
        needs = structured_call(
            FinanceDataNeeds,
            system=NEEDS_SYSTEM,
            user=f"Client problem:\n{query}\n\nCompany baseline:\n{anchors}\n\nResearch:\n{research}",
            fast=True,
        )
        queries = needs.queries
    except Exception:  # noqa: BLE001 - planning is best-effort
        queries = []

    if not queries:
        return "(no external data required)"

    results = gather_web(queries, total=6)
    if not results:
        return (
            "EXTERNAL DATA: searched the web but found no reliable figures for: "
            f"{'; '.join(queries)}. Treat these inputs as unsourced assumptions "
            "requiring due diligence — do NOT fabricate them."
        )
    lines = [f"EXTERNAL DATA (searched: {'; '.join(queries)}):"]
    for r in results:
        lines.append(f"- {r.get('title', '')} ({r.get('url', '')}): {r.get('snippet', '')}")
    return "\n".join(lines)


def finance_node(state: ConsultingState) -> dict:
    query = state["query"]
    research = state.get("research_findings", {})
    problem_type = state.get("problem_type", "general")

    # Retrieve the company's real financial baseline to anchor the analysis.
    anchors = _format_anchors(retrieve(_ANCHOR_QUERY, k=5))
    # Search the web for any external figures we need but don't have — rather
    # than letting the model invent them.
    external = _gather_external_data(query, anchors, research)
    advisory = jurisdiction_advisory(query)
    advisory_block = f"{advisory}\n\n" if advisory else ""

    # Step 1 — generate + run analysis code, grounded on the baseline.
    # Temperature 0 for determinism — finance numbers should be reproducible.
    code_reply = chat_invoke(
        [
            SystemMessage(content=CODE_SYSTEM),
            HumanMessage(
                content=(
                    f"{advisory_block}"
                    f"Client problem:\n{query}\n\n"
                    f"COMPANY BASELINE (real figures — anchor on these):\n{anchors}\n\n"
                    f"{external}\n\n"
                    f"Research context:\n{research}"
                )
            ),
        ],
        temperature=0.0,
    )
    code = _extract_code(
        code_reply.content if isinstance(code_reply.content, str) else str(code_reply.content)
    )
    computed = run_python(code)

    # Step 2 — synthesize a validated assessment.
    synth_user = (
        f"{advisory_block}"
        f"Client problem ({problem_type}):\n{query}\n\n"
        f"COMPANY BASELINE (real figures — stay consistent with these):\n{anchors}\n\n"
        f"{external}\n\n"
        f"Research context:\n{research}\n\n"
        f"Computed figures (stdout from analysis script):\n{computed}"
    )
    if state.get("critic_feedback"):
        synth_user += (
            "\n\nThis is a revision. The reviewer asked for:\n"
            f"{state['critic_feedback']}\nAddress it directly — add the "
            "operational levers and risk-adjusted assumptions they want."
        )

    analysis = structured_call(
        FinancialAnalysis, system=SYNTH_SYSTEM, user=synth_user, temperature=0.0
    )

    if analysis.quantitative_applicable and analysis.estimated_roi is not None:
        headline = (
            f"[Finance] ROI ~{analysis.estimated_roi:.0%}, "
            f"break-even {analysis.break_even_months} mo, "
            f"revenue ~${analysis.revenue_projection:,.0f}."
        )
    else:
        headline = "[Finance] Qualitative assessment (no revenue/ROI projection applies)."
    msg = AIMessage(content=f"{headline} {analysis.notes}")
    return {"financial_analysis": analysis.model_dump(), "messages": [msg]}
