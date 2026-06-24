"""Single source of truth for problem-type scoping.

Both the Engagement Manager (which builds the team) and the Critic (which knows
which specialists *should* have been engaged) read from here, so scoping and
review can never drift apart.

``strategy`` is implicit — it always runs last and is appended by the
engagement node, so it is not listed in the analysis floor here.
"""

from __future__ import annotations

# problem_type -> minimum analysis specialists required (besides strategy).
PROBLEM_POLICY: dict[str, list[str]] = {
    "market_entry": ["research", "finance", "risk"],
    "product_launch": ["research", "risk"],
    "cost_reduction": ["finance"],
    "churn": ["research"],
    # "Is X a good fit?" decisions need financial grounding + risk, not just research.
    "market_compare": ["research", "finance", "risk"],
    # Pricing changes hit revenue AND churn risk — model both.
    "pricing": ["research", "finance", "risk"],
    # M&A / large investments / partnerships are high-stakes & irreversible.
    "investment": ["research", "finance", "risk"],
    "general": [],
}

# Canonical execution order for the analysis tier (strategy appended separately).
ANALYSIS_ORDER = ["research", "finance", "risk"]


def required_analysis_agents(problem_type: str) -> list[str]:
    """Return the deterministic analysis floor for a problem type."""
    return PROBLEM_POLICY.get(problem_type, [])


# --------------------------------------------------------------------------- #
# Jurisdiction risk guardrail
# --------------------------------------------------------------------------- #
# Jurisdictions under broad international sanctions/embargoes or otherwise
# carrying severe legal & reputational exposure for market entry. This is a
# guardrail to force the analysis to surface sanctions risk — NOT legal advice
# and not exhaustive.
SANCTIONED_JURISDICTIONS: dict[str, tuple[str, ...]] = {
    "Russia": ("russia", "russian"),
    "Belarus": ("belarus", "belarusian"),
    "Iran": ("iran", "iranian"),
    "North Korea": ("north korea", "dprk"),
    "Syria": ("syria", "syrian"),
    "Cuba": ("cuba", "cuban"),
    "Crimea": ("crimea",),
    "Myanmar": ("myanmar", "burma"),
    "Venezuela": ("venezuela", "venezuelan"),
}


def detect_sanctioned_jurisdictions(text: str) -> list[str]:
    """Return canonical names of any sanctioned jurisdictions mentioned."""
    low = (text or "").lower()
    return [
        name
        for name, aliases in SANCTIONED_JURISDICTIONS.items()
        if any(alias in low for alias in aliases)
    ]


def jurisdiction_advisory(text: str) -> str:
    """Advisory injected into research/risk prompts when a flagged jurisdiction
    is detected; empty string otherwise."""
    hits = detect_sanctioned_jurisdictions(text)
    if not hits:
        return ""
    return (
        f"JURISDICTION ALERT — {', '.join(hits)}: this target is subject to "
        "broad international sanctions/embargoes and severe legal & reputational "
        "exposure. Treat sanctions compliance and legal permissibility as the "
        "PRIMARY consideration; it can override financial attractiveness. Surface "
        "it explicitly and prominently, and prefer a no-go unless a compliant, "
        "lawful path is clearly established."
    )
