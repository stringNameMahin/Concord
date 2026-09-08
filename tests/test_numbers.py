import math

import pytest

from concord.normalize.numbers import (
    NotANumber,
    intervals_overlap,
    parse_quantity,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1,23,456", 123456.0),
        ("1,234,567", 1234567.0),
        ("81,415.38", 81415.38),
        ("740", 740.0),
        ("0.9", 0.9),
    ],
)
def test_indian_and_western_digit_grouping(raw, expected):
    assert parse_quantity(raw).number == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw,normalized",
    [
        ("8,142 Cr", 8.142e10),
        ("81,415.38 mn", 8.141538e10),
        ("1.4 Mn", 1.4e6),
        ("2.8 Bn", 2.8e9),
        ("5 lakh", 5e5),
        ("12 thousand", 12000.0),
    ],
)
def test_scale_words_reach_base_units(raw, normalized):
    assert parse_quantity(raw).normalized == pytest.approx(normalized, rel=1e-9)


def test_the_headline_corroboration_pair_overlaps():
    """8,142 Cr and 81,415.38 million are the same fact at different scales."""
    deck = parse_quantity("8,142 Cr")
    report = parse_quantity("81,415.38", default_scale="million")
    assert intervals_overlap(deck.interval, report.interval)


def test_a_genuinely_different_figure_does_not_overlap():
    a = parse_quantity("8,142 Cr")
    b = parse_quantity("7,224 Cr")
    assert not intervals_overlap(a.interval, b.interval)


def test_precision_interval_follows_written_digits():
    coarse = parse_quantity("8,142 Cr")
    assert coarse.interval == pytest.approx((8141.5e7, 8142.5e7))

    fine = parse_quantity("81,415.38", default_scale="million")
    assert fine.interval[1] - fine.interval[0] == pytest.approx(0.01e6)


def test_currency_symbols_and_words():
    assert parse_quantity("₹8,142 Cr").unit == "INR"
    assert parse_quantity("Rs. 500").unit == "INR"
    assert parse_quantity("$1.2 bn").unit == "USD"
    assert parse_quantity("500", default_unit="INR").unit == "INR"


def test_parenthesised_negative():
    q = parse_quantity("(452) Cr")
    assert q.number == -452.0
    assert q.normalized == pytest.approx(-4.52e9)


@pytest.mark.parametrize("raw", ["Nil", "-", "N/A", "", "   "])
def test_nil_markers_are_not_quantities(raw):
    with pytest.raises(NotANumber):
        parse_quantity(raw)


def test_percent_is_not_given_a_currency():
    q = parse_quantity("6.5%")
    assert q.is_percent and q.unit is None and q.number == 6.5


def test_bps_becomes_the_same_delta_as_percent():
    assert parse_quantity("781 bps").number == pytest.approx(7.81)
    assert parse_quantity("7.81%").number == pytest.approx(7.81)


@pytest.mark.parametrize(
    "raw,bound",
    [
        (">33,200", "greater_than"),
        ("over 33,250", "greater_than"),
        ("at least 100", "at_least"),
        ("less than 5", "less_than"),
        ("approximately 90", "about"),
    ],
)
def test_bounds_are_recognised(raw, bound):
    assert parse_quantity(raw).bound == bound


def test_lower_bound_is_half_open_and_contains_larger_values():
    bounded = parse_quantity(">33,200")
    assert bounded.interval[1] == math.inf
    assert intervals_overlap(bounded.interval, parse_quantity("33,278").interval)


def test_bound_excludes_values_below_it():
    bounded = parse_quantity("over 33,250")
    assert not intervals_overlap(bounded.interval, parse_quantity("33,100").interval)


def test_explicit_scale_beats_the_inherited_one():
    q = parse_quantity("8,142 Cr", default_scale="million")
    assert q.scale == "cr"
    assert q.normalized == pytest.approx(8.142e10)


def test_inherited_scale_is_ignored_for_percentages():
    q = parse_quantity("6.5%", default_scale="million")
    assert q.is_percent and q.normalized == pytest.approx(6.5)


def test_rounding_does_not_create_a_false_contradiction():
    """1.6% and 1.63% agree at the precision the coarser figure committed to."""
    assert intervals_overlap(
        parse_quantity("1.6%").interval, parse_quantity("1.63%").interval
    )


def test_parenthesised_negative_keeps_its_scale():
    q = parse_quantity("(452) Cr")
    assert q.number == -452.0
    assert q.scale == "cr"
    assert q.normalized == pytest.approx(-4.52e9)


def test_currency_prefixed_parenthesised_negative():
    q = parse_quantity("₹(404) Cr")
    assert q.unit == "INR"
    assert q.number == -404.0
    assert q.normalized == pytest.approx(-4.04e9)


def test_bare_parenthesised_negative():
    assert parse_quantity("(6.3%)").number == pytest.approx(-6.3)


@pytest.mark.parametrize(
    "written,code",
    [
        ("₹", "INR"),
        ("Rs.", "INR"),
        ("INR", "INR"),
        ("Indian Rupees", "INR"),
        ("Rupees in million", "INR"),
        ("US$", "USD"),
        ("US dollars", "USD"),
    ],
)
def test_the_same_currency_written_three_ways_resolves_to_one_code(written, code):
    """Observed live: '₹' and 'INR' were refused as incomparable currencies."""
    from concord.normalize.numbers import normalize_currency

    assert normalize_currency(written) == code


def test_a_unit_that_is_not_a_currency_is_left_alone():
    from concord.normalize.numbers import normalize_currency

    assert normalize_currency("Tons") == "Tons"
    assert normalize_currency(None) is None


def test_an_inherited_currency_matches_one_written_on_the_figure():
    inherited = parse_quantity("81,415.38", default_scale="million", default_unit="₹")
    written = parse_quantity("₹36,465.27 million")
    assert inherited.unit == written.unit == "INR"


@pytest.mark.parametrize("written", ["percent", "per cent", "%", "Per Cent", "percentage"])
def test_a_percentage_arriving_as_a_unit_is_still_a_percentage(written):
    """Observed live: the IMF wrote 'percent' and the RBI 'per cent', so two
    growth rates were refused as different units of measure."""
    q = parse_quantity("6.5", default_unit=written)
    assert q.is_percent is True
    assert q.unit is None
    assert q.normalized == 6.5


def test_two_spellings_of_percent_now_compare():
    from concord.normalize.numbers import is_percent_unit

    assert is_percent_unit("per cent") and is_percent_unit("percent")
    assert not is_percent_unit("INR") and not is_percent_unit("Tons")
    a = parse_quantity("7.8", default_unit="percent")
    b = parse_quantity("6.5", default_unit="per cent")
    assert a.is_percent == b.is_percent is True
    assert not intervals_overlap(a.interval, b.interval)
