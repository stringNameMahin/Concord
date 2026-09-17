from datetime import date

import pytest

from concord.normalize.periods import (
    Period,
    infer_fiscal_year_end,
    mentioned_years,
    overlaps,
    parse_as_of,
    parse_date,
    parse_period,
    same_interval,
    years_compatible,
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


# --- a year label that names less than the year --------------------------


@pytest.mark.parametrize(
    "label, start, end",
    [
        ("first eight months of FY24", date(2023, 4, 1), date(2023, 11, 30)),
        ("first nine months of FY25", date(2024, 4, 1), date(2024, 12, 31)),
        ("first quarter of FY25", date(2024, 4, 1), date(2024, 6, 30)),
        ("last three months of FY24", date(2024, 1, 1), date(2024, 3, 31)),
        ("H1 FY25", date(2024, 4, 1), date(2024, 9, 30)),
        ("H1 of FY24", date(2023, 4, 1), date(2023, 9, 30)),
        ("second half of FY25", date(2024, 10, 1), date(2025, 3, 31)),
    ],
)
def test_a_run_cut_out_of_a_year_resolves_to_that_run(label, start, end):
    period = parse_period(label, fy_end_month=FY)
    assert (period.start, period.end) == (start, end)


def test_a_partial_year_is_never_equal_to_its_whole_year():
    """Widening a part into its whole made a partial-year figure and a
    full-year figure read as one stated condition, and their disagreement then
    came out `contradicts`."""
    part = parse_period("first eight months of FY24", fy_end_month=FY)
    whole = parse_period("FY24", fy_end_month=FY)
    assert not same_interval(part, whole)
    assert overlaps(part, whole)


def test_the_last_run_of_a_year_meets_its_quarter():
    assert same_interval(
        parse_period("last three months of FY24", fy_end_month=FY),
        parse_period("Q4 FY24", fy_end_month=FY),
    )


def test_one_quarter_spelled_two_ways_is_one_condition():
    assert same_interval(
        parse_period("Q2 of FY25", fy_end_month=FY),
        parse_period("Q2 FY25", fy_end_month=FY),
    )


@pytest.mark.parametrize(
    "label",
    [
        "FY20 to FY24",
        "FY23-FY26",
        "Between FY22 and FY26",
        "till FY26",
        "inception till FY26",
        "FY25 (April-December)",
    ],
)
def test_a_label_that_is_not_one_whole_year_abstains(label):
    """Refusing leaves the label to be compared as text, which keeps two
    different labels different. Widening it to a year we cannot justify does
    not."""
    assert parse_period(label, fy_end_month=FY) is None


@pytest.mark.parametrize("label", ["2024-25 (P)", "2024-25 (RE)", "full year FY24"])
def test_words_that_do_not_change_the_span_still_resolve(label):
    assert parse_period(label, fy_end_month=FY) is not None


# --- which years a label names ---------------------------------------------


@pytest.mark.parametrize(
    "label, years",
    [
        ("till FY26", {2026}),
        ("inception till FY26", {2026}),
        ("March 31, 2024", {2024}),
        ("end of FY24", {2024}),
        ("financial year under review", set()),
        ("April-December 2023", {2023}),
    ],
)
def test_a_label_names_the_years_it_spells(label, years):
    assert mentioned_years(label) == years


@pytest.mark.parametrize(
    "left, right, compatible",
    [
        ("till FY26", "inception till FY26", True),
        ("financial year under review", "financial year ended March 31, 2026", True),
        ("March 31, 2024", "end of FY24", True),
        ("April-December 2024", "April-December 2023", False),
        ("end of FY24", "end of FY23", False),
    ],
)
def test_years_compatible_when_neither_label_rules_the_other_out(left, right, compatible):
    assert years_compatible(left, right) is compatible


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


def test_a_value_that_cites_a_date_is_not_a_date():
    """Observed live: two different contract conditions citing one anchor date
    compared as the same condition, because a date was found anywhere in the
    string. That turned a genuine reconciliation into a contradiction."""
    from concord.normalize.periods import bare_date, parse_date

    cited = "cessation of employment prior to one year from August 24, 2021"
    other = "cessation of employment after one year but prior to two years from August 24, 2021"

    assert parse_date(cited) == parse_date(other)  # both mention the same day
    assert bare_date(cited) is None and bare_date(other) is None


@pytest.mark.parametrize(
    "written",
    ["March 31, 2024", "As at March 31, 2024", "as of 31 March 2024", "31/03/2024"],
)
def test_a_date_wearing_a_preposition_is_still_a_date(written):
    from concord.normalize.periods import bare_date

    assert bare_date(written) == date(2024, 3, 31)


# --- the fiscal basis is a claim about a whole document ---------------------


def test_an_event_that_ended_on_a_date_does_not_set_the_fiscal_basis():
    """F2, in the shape that cost two verdicts.

    One sentence in ninety-five pages put the IMF Article IV on a September
    fiscal year, and every FY label in it then resolved six months away from
    every other publisher's reading of the same label. The sentence was the
    mission's own meeting schedule. `ended on <date>` is not a reporting basis
    unless what ended was a year.
    """
    text = (
        "The report was prepared for the Board's consideration on November 21, 2025, "
        "following discussions that ended on September 18, 2025, with the officials."
    )
    assert infer_fiscal_year_end(text) is None


def test_a_year_ending_still_sets_it_from_a_single_statement():
    """The precision is in which sentences count, not in how many."""
    text = "Notes to the Financial Statements for the year ended March 31, 2024."
    assert infer_fiscal_year_end(text) == 3


def test_a_year_ending_outvotes_an_unrelated_event_in_the_same_document():
    text = (
        "Consolidated statements for the year ended March 31, 2024. "
        "The engagement ended on September 18, 2025."
    )
    assert infer_fiscal_year_end(text) == 3


def test_a_tie_is_not_an_answer():
    """Two months with equal support means the document has not said.

    Picking a winner out of dict ordering would be a coin flip dressed as a
    reading, and the cost of abstaining is only that the label carries the
    `assumed` tier it should have had anyway.
    """
    text = (
        "the year ended March 31, 2024 ... the year ended December 31, 2024"
    )
    assert infer_fiscal_year_end(text) is None


def test_a_partial_period_still_does_not_set_the_year_end():
    assert infer_fiscal_year_end("nine months ended December 31, 2021") is None
    assert infer_fiscal_year_end("three months ended June 30, 2024") is None


def test_the_evidence_behind_the_basis_is_reported():
    """An inference over a whole document that nobody can see is how F2 hid."""
    from concord.normalize.periods import fiscal_year_end_evidence

    text = (
        "for the year ended March 31, 2024 and the year ended March 31, 2023, "
        "against the year ended December 31, 2022."
    )
    assert fiscal_year_end_evidence(text) == {3: 2, 12: 1}
    assert infer_fiscal_year_end(text) == 3


def test_an_unstated_basis_marks_every_label_assumed_rather_than_guessing():
    """The whole point of returning None: the label still resolves, on the
    default, and says so. Two publishers then read `FY25` the same way."""
    imf = parse_period("FY2024/25", infer_fiscal_year_end("discussions ended on September 18, 2025"))
    rbi = parse_period("2024-25", infer_fiscal_year_end("for the year ended March 31, 2025"))
    assert same_interval(imf, rbi)
    assert imf.assumed and not rbi.assumed


# --- spellings the corpus uses that used to fall through to string equality --

@pytest.mark.parametrize(
    "label,expected",
    [
        ("Fiscal 2021", ("2020-04-01", "2021-03-31")),
        ("fiscal year 2021", ("2020-04-01", "2021-03-31")),
        ("financial year 2024", ("2023-04-01", "2024-03-31")),
        ("FY2021", ("2020-04-01", "2021-03-31")),
    ],
)
def test_a_fiscal_year_resolves_however_the_document_spells_it(label, expected):
    """F12. `Fiscal 2021` is one document's house style, 212 occurrences of it,
    and it did not resolve - so it compared as text against every `FY2021`."""
    period = parse_period(label, fy_end_month=FY)
    assert (period.start.isoformat(), period.end.isoformat()) == expected


def test_every_spelling_of_one_fiscal_year_is_one_condition():
    spellings = ["FY21", "FY2021", "Fiscal 2021", "financial year 2021", "2020-21"]
    resolved = [parse_period(label, fy_end_month=FY) for label in spellings]
    assert all(same_interval(resolved[0], other) for other in resolved[1:])


def test_the_end_of_a_year_is_an_instant_and_not_the_year():
    """`4.6 MW as at "end of FY24"` and `as at "March 31, 2024"` are one
    instant written two ways. Read as the whole year they are not equal; read
    as an instant they are the same condition."""
    edge = parse_period("end of FY24", fy_end_month=FY)
    assert (edge.start.isoformat(), edge.end.isoformat()) == ("2024-03-31", "2024-03-31")
    assert edge.granularity == "instant"

    opening = parse_period("beginning of FY24", fy_end_month=FY)
    assert opening.start.isoformat() == "2023-04-01"


@pytest.mark.parametrize(
    "label,expected",
    [
        ("March 2026", ("2026-03-01", "2026-03-31")),
        ("Mar-26", ("2026-03-01", "2026-03-31")),
        ("Sep 2024", ("2024-09-01", "2024-09-30")),
        ("April-December 2024", ("2024-04-01", "2024-12-31")),
        ("Apr-Dec 2024", ("2024-04-01", "2024-12-31")),
    ],
)
def test_a_month_named_without_a_day_resolves_to_that_month(label, expected):
    period = parse_period(label, fy_end_month=FY)
    assert (period.start.isoformat(), period.end.isoformat()) == expected


def test_two_month_ranges_in_different_years_stay_different():
    """The reason the range is worth resolving rather than widening."""
    assert not same_interval(
        parse_period("April-December 2024"), parse_period("April-December 2023")
    )


def test_a_value_that_merely_mentions_a_month_is_not_a_month():
    assert parse_period("revised upward since March 2026") is None
    assert parse_period("supply agreements signed in Sep 2024 and after") is None


def test_a_full_date_is_still_a_date_and_not_a_month():
    """`March 31, 2024` has to keep falling through to `bare_date`; reading the
    day as a two-digit year would put it in 2031."""
    assert parse_period("March 31, 2024") is None
    assert parse_period("as at March 31, 2024") is None
    assert parse_date("March 31, 2024").isoformat() == "2024-03-31"


def test_a_bare_year_is_still_refused():
    """Deliberate. `2024` is either the calendar year or the fiscal year ending
    in it, the two differ by a quarter, and the label does not say which."""
    assert parse_period("2024") is None
    assert parse_period("during 2024") is None


def test_an_ordinal_day_is_still_a_date():
    """`March 31st 2026` did not resolve at all, in either direction."""
    assert parse_date("March 31st 2026").isoformat() == "2026-03-31"
    assert parse_date("31st March 2026").isoformat() == "2026-03-31"
    assert parse_date("March 31, 2026").isoformat() == "2026-03-31"
