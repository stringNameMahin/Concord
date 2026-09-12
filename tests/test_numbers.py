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
    assert parse_quantity("\u20b98,142 Cr").unit == "INR"
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
    q = parse_quantity("\u20b9(404) Cr")
    assert q.unit == "INR"
    assert q.number == -404.0
    assert q.normalized == pytest.approx(-4.04e9)


def test_bare_parenthesised_negative():
    assert parse_quantity("(6.3%)").number == pytest.approx(-6.3)


@pytest.mark.parametrize(
    "written,code",
    [
        ("\u20b9", "INR"),
        ("Rs.", "INR"),
        ("INR", "INR"),
        ("Indian Rupees", "INR"),
        ("Rupees in million", "INR"),
        ("US$", "USD"),
        ("US dollars", "USD"),
    ],
)
def test_the_same_currency_written_three_ways_resolves_to_one_code(written, code):
    """Observed live: the rupee sign and 'INR' were refused as incomparable."""
    from concord.normalize.numbers import normalize_currency

    assert normalize_currency(written) == code


def test_a_unit_that_is_not_a_currency_is_left_alone():
    from concord.normalize.numbers import normalize_currency

    assert normalize_currency("Tons") == "Tons"
    assert normalize_currency(None) is None


def test_an_inherited_currency_matches_one_written_on_the_figure():
    inherited = parse_quantity("81,415.38", default_scale="million", default_unit="\u20b9")
    written = parse_quantity("\u20b936,465.27 million")
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


# --- reading a magnitude back out of the document -------------------------
#
# `scale_after_figure` is the deterministic half of the fix for the defect
# recorded as F1 in docs/devRead.md: a model that returns `raw="81,415.38"`
# from a sentence saying "81,415.38 million" leaves a record wrong by a factor
# of a million, and wrong invisibly.

from concord.normalize.numbers import scale_after_figure, truncates


def test_a_scale_word_directly_behind_the_figure_is_read():
    assert scale_after_figure("5,444", "Total cash balance: Rs 5,444 Cr") == "cr"
    assert scale_after_figure("81,415.38", "revenue was 81,415.38 million.") == "million"
    assert scale_after_figure("131", "to 131 lakh in FY24") == "lakh"


def test_the_currency_glyph_on_the_figure_does_not_block_the_match():
    """`raw` keeps the symbol as written; the text may space it differently."""
    assert scale_after_figure("Rs8,142", "Rs 8,142 Cr FY24 revenue") == "cr"


def test_a_scale_word_belonging_to_a_different_figure_is_not_returned():
    """The commonest way a loose rule would go wrong: one line, two figures."""
    assert scale_after_figure("13,087", "across 13,087 PIN codes with 2.85 million") is None


def test_a_scale_word_that_is_not_behind_the_figure_is_not_returned():
    assert scale_after_figure("0.56", "(per one million-person hours) 0.56") is None
    assert scale_after_figure("622", "Trade payables 622") is None


def test_a_figure_the_text_does_not_contain_returns_nothing():
    assert scale_after_figure("9,999", "revenue was 81,415.38 million") is None
    assert scale_after_figure("", "81,415.38 million") is None


def test_an_accounting_negative_still_finds_its_scale():
    """`(217)` wraps the figure alone, so the scale sits outside the bracket."""
    assert scale_after_figure("(217)", "Adjusted EBITDA (217) Cr") == "cr"


def test_the_indian_compound_scales_are_read_as_one_phrase():
    """`lakh crore` is a trillion. Read as `lakh` it is out by ten million.

    The RBI and the Union Budget both write large rupee sums this way, and the
    misreading is silent - the figure still parses and still looks ordinary.
    """
    assert parse_quantity("2.8 lakh crore").normalized == 2.8e12
    assert parse_quantity("11.9 lakh crores").normalized == 11.9e12
    assert parse_quantity("5 thousand crore").normalized == 5e10
    assert parse_quantity("2.8", default_scale="lakh crore").normalized == 2.8e12


def test_a_plain_lakh_is_still_a_lakh():
    """The compound must not swallow the simple case."""
    assert parse_quantity("2.8 lakh").normalized == 2.8e5
    assert parse_quantity("131", default_scale="lakh").normalized == 131e5


def test_a_compound_scale_broken_across_a_line_still_resolves():
    """PyMuPDF breaks the line between the two halves as readily as not."""
    assert scale_after_figure("2.8", "NFA expanded by Rs 2.8 lakh\ncrore") == "lakh crore"


def test_a_currency_token_between_the_figure_and_its_scale_is_allowed():
    """`622 INR crores` is one quantity; a currency cannot be another figure.

    Without this the two halves of a restated pair recovered asymmetrically
    and two statements of one figure came out as a contradiction.
    """
    assert scale_after_figure("622", "The total trade payable is 622 INR crores.") == "crores"
    assert scale_after_figure("668.3", "reserves stood at US$ 668.3 billion") == "billion"


def test_a_digit_run_inside_a_longer_figure_is_not_this_figure():
    """The needle is a digit run, so it matches inside any bigger number.

    Searching for `6` in this sentence finds the final digit of `602.6`, and a
    scale word sits directly behind *that*. Read without a boundary check it
    stored six per cent as six billion - the figure, the quote and the displayed
    value all correct, only `normalized` wrong by nine orders of magnitude.
    """
    quote = (
        "reaching USD 602.6 billion, witnessing a YoY growth\nof 6 per cent."
    )
    assert scale_after_figure("6", quote) is None
    # The figure the scale actually belongs to still reads it.
    assert scale_after_figure("602.6", quote) == "billion"


@pytest.mark.parametrize(
    "needle,text",
    [
        ("5", "a provision of 125 crore was made"),      # suffix of a longer run
        ("12", "a provision of 125 crore was made"),      # prefix of a longer run
        ("2", "the ratio moved from 1.2 million units"),  # across the decimal point
        ("44", "5,444 Cr of cash"),                       # inside a grouped figure
    ],
)
def test_no_part_of_a_bigger_figure_borrows_its_scale(needle, text):
    """A digit, a thousands comma or a decimal point on either side means the
    run belongs to a bigger figure and says nothing about this one."""
    assert scale_after_figure(needle, text) is None


def test_a_percentage_never_takes_a_scale_from_its_unit():
    """The guard has to bind before the scale is applied, not after.

    Percentness has two sources - the figure's tail and the unit - and deciding
    it from the tail alone left `not is_percent` true at the moment the scale
    was applied. A percentage then carried a plus-or-minus-500-million interval.
    """
    q = parse_quantity("6", default_scale="billion", default_unit="per cent")
    assert q.is_percent is True
    assert q.scale is None
    assert q.normalized == 6.0
    assert q.interval == (5.5, 6.5)


def test_an_absolute_figure_still_takes_its_inherited_scale():
    """The percent guard must not cost the ordinary case its magnitude."""
    q = parse_quantity("81,415.38", default_scale="million", default_unit="INR")
    assert q.is_percent is False
    assert q.normalized == pytest.approx(81_415_380_000.0)


@pytest.mark.parametrize(
    "stated,recovered,expected",
    [
        ("lakh", "lakh crore", True),       # stopped reading after one word
        ("crore", "lakh crore", True),      # started reading one word late
        ("thousand", "thousand crore", True),
        ("million", "cr", False),           # a real disagreement, model wins
        ("cr", "crore", False),             # not a word of it, a substring
        ("lakh crore", "lakh crore", False),  # the same phrase is not a truncation
        (None, "lakh crore", False),
        ("lakh", None, False),
    ],
)
def test_a_truncated_compound_is_told_apart_from_a_disagreement(stated, recovered, expected):
    assert truncates(stated, recovered) is expected


def test_a_compound_truncated_inside_the_figure_loses_to_the_context():
    """The model can drop the second word of a compound inside `raw` itself.

    Handed "an outlay of Rs 1.5 lakh crore" it returned `raw="Rs1.5 lakh"`, so
    the scale read off the figure's own tail is `lakh` while the document says
    `lakh crore` - a factor of ten million. The tail normally wins; a truncation
    of what the context names is the exception, the same rule `materialize`
    applies to the model's `scale` field.
    """
    q = parse_quantity("₹1.5 lakh", default_scale="lakh crore", default_unit="INR")
    assert q.scale == "lakh crore"
    assert q.normalized == 1.5e12


def test_the_figures_own_tail_still_wins_a_real_disagreement():
    """`Cr` against an inherited `million` is a disagreement, not a truncation,
    and the word written on the figure is the more specific evidence."""
    q = parse_quantity("8,142 Cr", default_scale="million")
    assert q.scale == "cr"
    assert q.normalized == 8142 * 1e7


def test_a_lakh_with_no_compound_in_context_is_still_a_lakh():
    assert parse_quantity("2.8 lakh").normalized == 2.8e5
    assert parse_quantity("2.8 lakh", default_scale="million").normalized == 2.8e5
