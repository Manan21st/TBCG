"""Report assembly: build a markdown consulting report and export it to PDF."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langsmith import traceable


def _confidence_pct(value: Any) -> str:
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "n/a"


@traceable(run_type="chain", name="build_report")
def build_markdown_report(state: dict) -> str:
    """Assemble the final consulting report as markdown from graph state."""
    query = state.get("query", "")
    problem_type = state.get("problem_type", "general")
    research = state.get("research_findings", {}) or {}
    finance = state.get("financial_analysis", {}) or {}
    strategy = state.get("strategy_recommendation", {}) or {}
    risk = state.get("risk_assessment", {}) or {}

    lines: list[str] = []
    lines.append("# TBCG Consulting Report")
    lines.append("")
    lines.append(f"**Engagement:** {query}")
    lines.append(f"**Problem type:** {problem_type}")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    if strategy:
        lines.append(strategy.get("recommendation", "—"))
        lines.append("")
        lines.append(
            f"*Confidence: {_confidence_pct(strategy.get('confidence'))} · "
            f"Overall risk: {risk.get('risk_level', 'n/a')}*"
        )
    else:
        lines.append("_No recommendation produced._")
    lines.append("")

    # Research
    if research:
        lines.append("## Research Findings")
        if research.get("summary"):
            lines.append(research["summary"])
            lines.append("")
        for f in research.get("findings", []):
            lines.append(
                f"- {f.get('claim', '')} "
                f"_(source: {f.get('source', '?')}, "
                f"confidence: {_confidence_pct(f.get('confidence'))})_"
            )
        lines.append("")

    # Finance
    if finance:
        lines.append("## Financial Analysis")
        proj = finance.get("revenue_projection")
        be = finance.get("break_even_months")
        no_go = finance.get("viable") is False
        # Some decisions (governance, capital structure, pure diagnostics) aren't
        # revenue/ROI-quantifiable — present a qualitative assessment instead of
        # fabricated numbers.
        quantitative = finance.get("quantitative_applicable", True) and (
            finance.get("estimated_roi") is not None
        )

        if not quantitative:
            lines.append("- **Quantitative projection:** not applicable to this decision")
            if finance.get("notes"):
                lines.append(f"- **Financial assessment:** {finance['notes']}")
        else:
            lines.append(f"- **Estimated ROI:** {_confidence_pct(finance.get('estimated_roi'))}")
            if no_go:
                lines.append("- **Break-even:** N/A — not recommended")
                lines.append("- **Revenue projection:** N/A — not recommended")
            else:
                lines.append(f"- **Break-even:** {be if be is not None else 'n/a'} months")
                proj_str = f"${proj:,.0f}" if isinstance(proj, (int, float)) else "n/a"
                lines.append(f"- **Revenue projection:** {proj_str}")
            if finance.get("notes"):
                lines.append(f"- **Basis:** {finance['notes']}")
        if finance.get("operational_levers"):
            lines.append("- **Operational levers:**")
            for lever in finance["operational_levers"]:
                lines.append(f"  - {lever}")
        if finance.get("assumptions"):
            lines.append("- **Assumptions:**")
            for a in finance["assumptions"]:
                lines.append(f"  - {a}")
        lines.append("")

    # Strategy detail
    if strategy:
        lines.append("## Strategic Recommendation")
        lines.append(strategy.get("recommendation", "—"))
        if strategy.get("rationale"):
            lines.append("")
            lines.append(f"**Rationale:** {strategy['rationale']}")
        if strategy.get("alternatives"):
            lines.append("")
            lines.append("**Alternatives considered:**")
            for alt in strategy["alternatives"]:
                lines.append(f"- {alt}")
        lines.append("")

    # Risk
    if risk:
        lines.append("## Risk Assessment")
        lines.append(f"**Overall risk level:** {risk.get('risk_level', 'n/a')}")
        lines.append("")
        if risk.get("major_risks"):
            lines.append("**Major risks:**")
            for r in risk["major_risks"]:
                lines.append(f"- {r}")
        if risk.get("mitigations"):
            lines.append("")
            lines.append("**Mitigations:**")
            for m in risk["mitigations"]:
                lines.append(f"- {m}")
        lines.append("")

    # Reviewer caveats — only surface when the recommendation went through
    # revision(s) or was force-approved at the cap. On a clean first-pass
    # approval the critic's note is just praise, so we omit it.
    if state.get("revision_count", 0) > 0 and state.get("critic_feedback"):
        lines.append("## Reviewer Caveats")
        lines.append(state["critic_feedback"])
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_pdf(markdown_text: str):
    """Build and return an fpdf2 ``FPDF`` document from report markdown.

    Lightweight renderer (headings, bullets, paragraphs) — not a full markdown
    engine — but produces a clean, shareable PDF with no system dependencies.
    """
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    def write(text: str, size: int, style: str = ""):
        pdf.set_font("Helvetica", style, size)
        # Latin-1 safe (fpdf2 core fonts) — replace unsupported chars.
        safe = text.encode("latin-1", "replace").decode("latin-1")
        # Pin the cursor back to the left margin after each block so the next
        # line always has full width to render into.
        pdf.multi_cell(0, 6, safe, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    for raw in markdown_text.splitlines():
        line = raw.rstrip()
        if not line:
            pdf.ln(3)
        elif line.startswith("# "):
            write(line[2:], 18, "B")
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.ln(2)
            write(line[3:], 14, "B")
            pdf.ln(1)
        elif line.startswith("  - "):
            write(f"    - {line[4:]}", 11)
        elif line.startswith("- "):
            write(f"- {line[2:]}", 11)
        else:
            # strip simple markdown emphasis markers
            clean = line.replace("**", "").replace("*", "").replace("_", "")
            write(clean, 11)

    return pdf


def export_pdf(markdown_text: str, out_path: str | Path) -> Path:
    """Render the report markdown to a PDF file on disk."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _render_pdf(markdown_text).output(str(out))
    return out


def export_pdf_bytes(markdown_text: str) -> bytes:
    """Render the report markdown to PDF and return the raw bytes."""
    return bytes(_render_pdf(markdown_text).output())
