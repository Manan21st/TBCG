"""Research Agent — gathers external (web) and internal (RAG) evidence.

Quality comes from *planning* the search around what the company actually
sells and where, rather than firing the raw question at the web (which surfaced
generic stock-market outlooks before). The synthesis step then enforces
relevance, factual accuracy, and provenance discipline.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from ..llm import structured_call
from ..policy import jurisdiction_advisory
from ..schemas import ResearchFindings, SearchPlan
from ..state import ConsultingState
from ..tools.rag import retrieve
from ..tools.web_search import gather as gather_web

# Pulls a short company profile to anchor both search planning and synthesis.
_CONTEXT_QUERY = "company business model industry products customers segments markets"

PLAN_SYSTEM = """You are a research lead. Given a client problem and a short
profile of OUR company, produce 1-3 focused web search queries that will return
DECISION-RELEVANT intelligence for this specific business.

Make queries concrete and business-oriented — target market size, competitors,
regulation, and adoption for OUR industry in the target geography. Avoid generic
finance phrases like "market trends" or "market outlook" (they return stock-
market noise, not business-entry intelligence). Name the industry and the place
explicitly."""

SYSTEM = """You are a Research Consultant at TBCG. You are given a client
problem, a short profile of OUR company, web search results, and internal
knowledge-base excerpts. Synthesize the most decision-relevant findings.

RELEVANCE (be ruthless):
- Keep ONLY findings that inform THIS company's decision (our industry, our
  segments, the target geography). DISCARD generic stock-market/equity outlooks,
  macro punditry, and anything not tied to the business question.

ACCURACY:
- Get basic facts right. Do not mislabel a developed economy as "emerging".
  Identify the actual regulatory regime of the target geography (e.g. GDPR for
  the EU/UK) rather than forcing an unrelated internal proxy.
- If a JURISDICTION ALERT is present, lead with sanctions / legal-permissibility
  findings before commercial ones — it is the most decision-relevant fact.

GROUNDING & PROVENANCE:
- Ground every finding in the supplied evidence; do not invent specifics.
- Set "source" to the URL (web) or document name (internal-kb).
- Set "confidence" (0-1) by source quality, recency, and corroboration; heavily
  discount or drop stale (>~3 yr) or weak sources.
- Internal facts may be region/segment-specific. Do NOT restate a region-
  specific figure as a fact about a different target market — label it a
  PROXY/ASSUMPTION with reduced confidence."""


def _format_evidence(web: list[dict], kb: list[dict]) -> str:
    lines = ["## Web results"]
    for r in web:
        lines.append(f"- {r['title']} ({r['url']}): {r['snippet']}")
    lines.append("\n## Internal knowledge base")
    if kb:
        for r in kb:
            lines.append(f"- [{r['source']}] (sim={r['score']}): {r['text']}")
    else:
        lines.append("- (no internal documents matched)")
    return "\n".join(lines)


def _company_context() -> str:
    facts = retrieve(_CONTEXT_QUERY, k=3)
    if not facts:
        return "(no internal company profile available)"
    return "\n".join(f"- {f['text']}" for f in facts)


def _plan_queries(query: str, context: str) -> list[str]:
    try:
        plan = structured_call(
            SearchPlan,
            system=PLAN_SYSTEM,
            user=f"Client problem:\n{query}\n\nOur company profile:\n{context}",
            fast=True,
        )
        return plan.queries or [query]
    except Exception:  # noqa: BLE001 - fall back to the raw query on any failure
        return [query]


def research_node(state: ConsultingState) -> dict:
    query = state["query"]
    context = _company_context()
    queries = _plan_queries(query, context)

    # If the target is a sanctioned/high-risk jurisdiction, force a sanctions
    # search and tell the synthesizer to lead with it.
    advisory = jurisdiction_advisory(query)
    if advisory:
        queries.append(f"{query} international sanctions export restrictions foreign companies")
    web = gather_web(queries)
    kb = retrieve(query, k=4)

    user = (
        f"Client problem:\n{query}\n\n"
        f"Our company profile:\n{context}\n\n"
        f"Evidence:\n{_format_evidence(web, kb)}"
    )
    if advisory:
        user = f"{advisory}\n\n{user}"
    if state.get("conduct_advisory"):
        user = f"{state['conduct_advisory']}\n\n{user}"
    if state.get("critic_feedback"):
        user += (
            "\n\nThis is a revision. The reviewer asked for:\n"
            f"{state['critic_feedback']}\nFocus the findings on addressing this."
        )

    findings = structured_call(ResearchFindings, system=SYSTEM, user=user)

    msg = AIMessage(
        content=(
            f"[Research] {len(findings.findings)} findings "
            f"({len(web)} web, {len(kb)} internal). {findings.summary}"
        )
    )
    return {"research_findings": findings.model_dump(), "messages": [msg]}
