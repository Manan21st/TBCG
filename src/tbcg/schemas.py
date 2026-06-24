"""Pydantic schemas for each agent's structured output.

These are the contracts the graph relies on: every agent returns a validated
object, so downstream nodes never parse free text.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

ProblemType = Literal[
    "market_entry",
    "product_launch",
    "cost_reduction",
    "churn",
    "market_compare",
    "pricing",
    "investment",
    "general",
]

AgentName = Literal["research", "finance", "strategy", "risk"]


class EngagementPlan(BaseModel):
    """Engagement manager: classify the problem and select specialists."""

    problem_type: ProblemType
    required_agents: list[AgentName] = Field(
        description="Specialist agents required, in execution order."
    )
    risk_material: bool = Field(
        description=(
            "True if the decision carries material downside (financial, "
            "operational, legal, concentration, macro/recession, reputational) "
            "or the question is itself about risk/threats/dependence/resilience. "
            "When true, a risk assessment is engaged regardless of problem_type."
        )
    )
    rationale: str = Field(description="One-sentence justification for the routing.")


class ConductScreen(BaseModel):
    """Engagement manager: ethics/legality screen on the client request."""

    flagged: bool = Field(
        description=(
            "True if the request seeks action that is illegal, deceptive, "
            "consumer-harmful, anticompetitive, privacy-violating, or otherwise "
            "unethical. Normal aggressive-but-lawful tactics are NOT flagged."
        )
    )
    concern: str = Field(
        default="",
        description="One-sentence statement of the ethical/legal concern when flagged.",
    )


class Finding(BaseModel):
    claim: str
    source: str = Field(description="URL, document name, or 'internal-kb'.")
    confidence: float = Field(ge=0.0, le=1.0)


class SearchPlan(BaseModel):
    """Research agent: focused web queries to run, derived from the problem
    plus the company's actual business context."""

    queries: list[str] = Field(
        min_length=1,
        max_length=3,
        description="Specific, business-relevant web search queries (not generic).",
    )


class ResearchFindings(BaseModel):
    """Research agent: external + internal evidence."""

    summary: str = Field(
        description=(
            "A 1-2 sentence SYNTHESIS of the key conclusion relevant to the "
            "decision (the actual takeaway), NOT a description of what was "
            "examined. Bad: 'Findings on X considering A, B, C.' Good: 'X is "
            "attractive because ..., but ... is the main risk.'"
        )
    )
    findings: list[Finding]


class FinanceDataNeeds(BaseModel):
    """Finance agent: external figures it must source (not in baseline/research)
    rather than fabricate — e.g. a target's revenue, a market size, regional
    costs. Empty when no external data is needed."""

    queries: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Web search queries for missing EXTERNAL figures. Empty if none.",
    )


class FinancialAnalysis(BaseModel):
    """Finance agent (incl. operations analysis).

    Field bounds are guardrails: they reject physically-impossible outputs
    (negative break-even, absurd ROI) so ``structured_call`` forces the model
    to correct rather than letting nonsense reach the report.
    """

    quantitative_applicable: bool = Field(
        description=(
            "True if this decision is a revenue/cost-quantifiable initiative with "
            "a meaningful ROI / break-even / revenue projection. False for "
            "governance, capital-structure, pure-diagnostic, or qualitative "
            "questions — then leave the three numeric fields null and give the "
            "financial assessment qualitatively in notes."
        )
    )
    estimated_roi: float | None = Field(
        default=None,
        ge=-1.0,
        le=20.0,
        description="ROI as a decimal, e.g. 0.27 = 27%. Null if not quantifiable.",
    )
    break_even_months: int | None = Field(
        default=None,
        ge=1,
        le=600,
        description="Whole months to break even (>=1; 600 = never). Null if N/A.",
    )
    revenue_projection: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "GROSS incremental annual revenue attributable to the decision (USD, "
            ">=0). Upside only, never a net delta — capture downside via ROI. "
            "Null if not quantifiable; 0 only when not viable (viable=false)."
        ),
    )
    viable: bool = Field(
        description=(
            "True if proceeding is financially/legally advisable. False for a "
            "no-go (sanctions, prohibitive economics). Drives report rendering."
        )
    )
    assumptions: list[str] = Field(min_length=1)
    operational_levers: list[str] = Field(
        default_factory=list,
        description="Concrete operational/efficiency levers and execution steps.",
    )
    notes: str = Field(
        min_length=40,
        description=(
            "2-4 sentence narrative explaining HOW the figures were derived: "
            "the method, which baseline figures were used, and the key drivers "
            "behind the ROI / break-even / revenue numbers."
        ),
    )

    @model_validator(mode="after")
    def _check_revenue_scale(self):
        # Catch the recurring units error where the model emits revenue in
        # millions-as-integer (e.g. 21 meaning $21M). A viable, quantifiable
        # strategic initiative never has incremental revenue under ~$1,000;
        # such a value is almost certainly a scale mistake — reject so
        # structured_call retries and the model returns absolute dollars.
        if (
            self.quantitative_applicable
            and self.viable
            and self.revenue_projection is not None
            and 0 < self.revenue_projection < 1_000
        ):
            raise ValueError(
                f"revenue_projection={self.revenue_projection} is implausibly "
                "small — express it in ABSOLUTE dollars (e.g. 21000000 for $21M), "
                "not in millions."
            )
        return self


class StrategyRecommendation(BaseModel):
    """Strategy agent: synthesized recommendation."""

    recommendation: str
    alternatives: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class RiskAssessment(BaseModel):
    """Risk agent: weaknesses and threats in the proposed strategy."""

    risk_level: Literal["low", "medium", "high"]
    major_risks: list[str]
    mitigations: list[str] = Field(default_factory=list)


class CriticVerdict(BaseModel):
    """Critic agent: independent quality review."""

    approved: bool
    feedback: str = Field(description="Actionable feedback; required when not approved.")
    issues: list[str] = Field(default_factory=list)


class RevisionPlan(BaseModel):
    """Revision dispatcher: decides where a revision must re-enter the pipeline.

    The agents form a dependency chain (research → finance → strategy → risk →
    critic), so we only need the EARLIEST stage whose output must change to
    address the feedback — everything downstream re-runs automatically. This is
    model-driven (interpret the feedback), not keyword-based.
    """

    target: Literal["research", "finance", "strategy", "risk"] = Field(
        description=(
            "Earliest stage whose output must change to satisfy the feedback. "
            "research=evidence wrong/missing; finance=numbers/assumptions; "
            "strategy=the recommendation or its depth/wording; risk=only the risk "
            "assessment (recommendation stands)."
        )
    )
    reason: str = Field(description="One sentence on why this is the right entry point.")
