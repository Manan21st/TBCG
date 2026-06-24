"""Finance sources external data via search instead of fabricating it."""

from unittest.mock import patch

from tbcg.agents import finance
from tbcg.schemas import FinanceDataNeeds


def test_no_external_data_needed():
    with patch.object(finance, "structured_call", return_value=FinanceDataNeeds(queries=[])):
        out = finance._gather_external_data("reduce costs", "ARR $24M", {})
    assert "no external data required" in out.lower()


def test_search_gap_is_flagged_not_fabricated():
    needs = FinanceDataNeeds(queries=["DataLite ARR revenue"])
    with (
        patch.object(finance, "structured_call", return_value=needs),
        patch.object(finance, "gather_web", return_value=[]),
    ):
        out = finance._gather_external_data("acquire DataLite", "ARR $24M", {})
    # Must instruct the model NOT to fabricate, and require due diligence.
    assert "due diligence" in out.lower()
    assert "do not fabricate" in out.lower()


def test_found_external_data_is_passed_through():
    needs = FinanceDataNeeds(queries=["Spain SaaS market size"])
    hits = [{"title": "Spain SaaS", "url": "http://x", "snippet": "market ~$2B"}]
    with (
        patch.object(finance, "structured_call", return_value=needs),
        patch.object(finance, "gather_web", return_value=hits),
    ):
        out = finance._gather_external_data("enter Spain", "ARR $24M", {})
    assert "EXTERNAL DATA" in out
    assert "$2B" in out
