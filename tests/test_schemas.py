"""Schema guardrail tests — impossible finance outputs must be rejected."""

import pytest
from pydantic import ValidationError

from tbcg.schemas import FinancialAnalysis

VALID = dict(
    quantitative_applicable=True,
    estimated_roi=0.27,
    break_even_months=18,
    revenue_projection=2_500_000,
    viable=True,
    assumptions=["20% channel margin"],
    notes="Derived from the $468M revenue baseline and ~40% gross margin; new-line sales drive the projection.",
)


def test_non_quantifiable_allows_null_numbers():
    fa = FinancialAnalysis(
        quantitative_applicable=False,
        viable=True,
        assumptions=["Going-private removes public-company costs"],
        notes="Capital-structure decision; assessed via debt servicing and control implications, not ROI.",
    )
    assert fa.estimated_roi is None
    assert fa.break_even_months is None
    assert fa.revenue_projection is None


def test_valid_financial_analysis_passes():
    fa = FinancialAnalysis(**VALID)
    assert fa.break_even_months == 18


@pytest.mark.parametrize(
    "override",
    [
        {"break_even_months": 0},  # zero break-even (was seen in UK run)
        {"break_even_months": -75},  # negative break-even (was seen in UK run)
        {"break_even_months": 9999},  # absurdly far out
        {"estimated_roi": -2.0},  # worse than total loss
        {"estimated_roi": 50.0},  # 5000% ROI
        {"revenue_projection": -100},  # negative revenue
        {"assumptions": []},  # no stated assumptions
        {"notes": "too short"},  # no real explanation of derivation
        {"revenue_projection": 21},  # millions-as-integer units error ($21 ≠ $21M)
    ],
)
def test_impossible_values_rejected(override):
    bad = {**VALID, **override}
    with pytest.raises(ValidationError):
        FinancialAnalysis(**bad)
