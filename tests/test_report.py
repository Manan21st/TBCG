"""Report assembly tests — no LLM/network required."""

from tbcg.tools.report import build_markdown_report, export_pdf_bytes

SAMPLE_STATE = {
    "query": "Should we enter Indonesia?",
    "problem_type": "market_entry",
    "research_findings": {
        "summary": "Indonesia is the largest SEA market.",
        "findings": [
            {
                "claim": "Data localization rules apply.",
                "source": "market_notes_sea.md",
                "confidence": 0.8,
            }
        ],
    },
    "financial_analysis": {
        "estimated_roi": 0.27,
        "break_even_months": 18,
        "revenue_projection": 2500000,
        "assumptions": ["20% channel margin"],
        "operational_levers": [],
        "quantitative_applicable": True,
        "viable": True,
        "notes": "Based on $468M revenue baseline and ~40% margin; new-line sales over year one.",
    },
    "strategy_recommendation": {
        "recommendation": "Enter via local partnership.",
        "alternatives": ["Direct entry", "Acquisition"],
        "confidence": 0.81,
        "rationale": "Partnerships ease regulatory burden.",
    },
    "risk_assessment": {
        "risk_level": "medium",
        "major_risks": ["Regulatory uncertainty", "Local competition"],
        "mitigations": ["Local legal counsel"],
    },
    "critic_feedback": "Looks solid.",
}


def test_markdown_report_has_all_sections():
    md = build_markdown_report(SAMPLE_STATE)
    for section in [
        "# TBCG Consulting Report",
        "## Executive Summary",
        "## Research Findings",
        "## Financial Analysis",
        "## Strategic Recommendation",
        "## Risk Assessment",
    ]:
        assert section in md
    assert "Enter via local partnership." in md
    assert "81%" in md  # confidence rendered
    assert "**Basis:**" in md  # finance narrative surfaced


def test_markdown_report_handles_empty_state():
    md = build_markdown_report({"query": "x", "problem_type": "general"})
    assert "# TBCG Consulting Report" in md


def test_no_go_renders_as_not_recommended():
    state = {
        **SAMPLE_STATE,
        "financial_analysis": {
            **SAMPLE_STATE["financial_analysis"],
            "viable": False,
            "revenue_projection": 0,
            "break_even_months": 600,
        },
    }
    md = build_markdown_report(state)
    assert "N/A — not recommended" in md
    assert "600 months" not in md


def test_viable_negative_revenue_delta_not_flagged_no_go():
    # A clamped/zero net should NOT render no-go when finance says it's viable.
    state = {
        **SAMPLE_STATE,
        "financial_analysis": {
            **SAMPLE_STATE["financial_analysis"],
            "viable": True,
            "revenue_projection": 1_440_000,
        },
    }
    md = build_markdown_report(state)
    assert "N/A — not recommended" not in md
    assert "$1,440,000" in md


def test_non_quantifiable_finance_renders_qualitatively():
    state = {
        **SAMPLE_STATE,
        "financial_analysis": {
            "quantitative_applicable": False,
            "viable": True,
            "estimated_roi": None,
            "break_even_months": None,
            "revenue_projection": None,
            "assumptions": ["Removing public-company costs"],
            "operational_levers": [],
            "notes": "Capital-structure decision assessed via debt servicing and control, not ROI.",
        },
    }
    md = build_markdown_report(state)
    assert "not applicable to this decision" in md
    assert "Financial assessment:" in md
    assert "Estimated ROI" not in md  # no fabricated numeric lines


def test_pdf_export_returns_bytes():
    pdf = export_pdf_bytes(build_markdown_report(SAMPLE_STATE))
    assert isinstance(pdf, bytes)
    assert pdf[:4] == b"%PDF"
