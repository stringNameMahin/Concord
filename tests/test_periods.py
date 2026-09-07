from datetime import date

import pytest

from concord.normalize.periods import (
    Period,
    infer_fiscal_year_end,
    overlaps,
    parse_as_of,
    parse_period,
    same_interval,
)

FY = 3


@pytest.mark.parametrize(
    "label",
    ["FY24", "FY 2023-24", "2023-24", "FY2023-24", "fiscal year 2023-24", "FY-24"],
)
def test_all_spellings_of_one_fiscal_year_agree(label):
    """The corpus writes the same interval six ways. They must collapse to one."""
    period = parse_period(label, fy_end_month=FY)
    assert period.start == date(2023, 4, 1)
    assert period.end == date(2024, 3, 31)


def test_imf_and_rbi_spellings_of_the_same_year_are_equal():
    imf = parse_period("FY2025/26", fy_end_month=FY)
    rbi = parse_period("2025-26", fy_end_month=FY)
    assert same_interval(imf, rbi)
    assert imf.start == date(2025, 4, 1) and imf.end == date(2026, 3, 31)


def test_calendar_year_is_not_a_fiscal_year():
    cy = parse_period("CY2024", fy_end_month=FY)
    assert cy.start == date(2024, 1, 1) and cy.end == date(2024, 12, 31)
    assert not same_interval(cy, parse_period("FY24", fy_end_month=FY))


def test_year_ended_statement_sets_its_own_basis():
    period = parse_period("for the year ended March 31, 2024")
    assert period.start == date(2023, 4, 1) and period.end == date(2024, 3, 31)
    assert period.fiscal_basis.startswith("stated")
    assert not period.assumed


def test_december_year_end_is_respected_not_overridden():
    period = parse_period("for the year ended December 31, 2023")
    assert period.start == date(2023, 1, 1) and period.end == date(2023, 12, 31)


def test_nine_month_partial_period():
    period = parse_period("nine months period ended December 31, 2021")
    assert period.start == date(2021, 4, 1)
    assert period.end == date(2021, 12, 31)
    assert period.granularity == "months"


def test_partial_period_is_not_the_same_as_the_full_year():
    partial = parse_period("nine months ended December 31, 2021")
    full = parse_period("FY22", fy_end_month=FY)
    assert not same_interval(partial, full)
    assert overlaps(partial, full)


def test_fiscal_quarters_land_in_the_right_months():
    q4 = parse_period("Q4 FY24", fy_end_month=FY)
    assert q4.start == date(2024, 1, 1) and q4.end == date(2024, 3, 31)

    q1 = parse_period("Q1 FY25", fy_end_month=FY)
    assert q1.start == date(2024, 4, 1) and q1.end == date(2024, 6, 30)


def test_quarter_is_not_the_same_interval_as_its_year():
    assert not same_interval(
        parse_period("Q4 FY24", fy_end_month=FY), parse_period("FY24", fy_end_month=FY)
    )


def test_fiscal_basis_is_recorded_as_assumed_when_unstated():
    assert parse_period("FY24").assumed
    assert not parse_period("FY24", fy_end_month=FY).assumed


def test_a_different_fiscal_basis_gives_a_different_interval():
    india = parse_period("FY24", fy_end_month=3)
    us = parse_period("FY24", fy_end_month=9)
    assert not same_interval(india, us)
    assert us.start == date(2023, 10, 1) and us.end == date(2024, 9, 30)


def test_fiscal_year_end_is_inferred_from_the_document():
    text = (
        "Notes to the Consolidated Financial Statements for the year ended March 31, 2024. "
        "Comparatives are for the year ended March 31, 2023."
    )
    assert infer_fiscal_year_end(text) == 3


def test_partial_period_statements_do_not_set_the_year_end():
    assert infer_fiscal_year_end("three months ended June 30, 2024") is None


def test_as_of_is_a_point_not_an_interval():
    assert parse_as_of("as at March 31, 2024") == date(2024, 3, 31)
    assert parse_as_of("as of December 31, 2021") == date(2021, 12, 31)
    assert parse_as_of("for the year ended March 31, 2024") is None


@pytest.mark.parametrize("junk", ["", "revenue from services", "the board of directors", "12"])
def test_unparseable_text_abstains(junk):
    assert parse_period(junk) is None


def test_same_interval_is_false_when_either_side_is_missing():
    assert not same_interval(None, parse_period("FY24", fy_end_month=FY))
    assert not same_interval(None, None)
