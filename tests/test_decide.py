"""One test per row of the decision table, plus the two guards.

The table is the graded logic of the system, so each row is pinned by name:
if a rule stops firing, the failing test says which row broke.
"""

import pytest
from factories import make_fact

from concord.compare.decide import (
    AGREE,
    DISAGREE,
    INCOMPARABLE,
    compare_qualifiers,
    compare_values,
    decide,
    is_discriminating,
)
from concord.facts import comparison_keys_match, subjects_match


# --- row 1: comparison keys differ -----------------------------------------

def test_different_predicate_is_unrelated():
    decision = decide(
        make_fact(predicate="revenue_from_services"),
        make_fact(predicate="employee_benefit_expense"),
    )
    assert decision.verdict == "unrelated"
    assert decision.rule_fired == "comparison_key_differs"


def test_different_hard_keys_are_unrelated_despite_similar_names():
    decision = decide(
        make_fact(subject="Acme Logistics Limited", subject_key="CIN:AAA"),
        make_fact(subject="Acme Logistics Limited", subject_key="CIN:BBB"),
    )
    assert decision.verdict == "unrelated"


def test_surface_subset_counts_as_the_same_subject():
    assert subjects_match(
        make_fact(subject="Acme"), make_fact(subject="Acme Logistics Limited")
    )
    assert not subjects_match(
        make_fact(subject="Acme Logistics"), make_fact(subject="Borealis Freight")
    )


def test_alias_map_makes_two_predicates_one_key():
    a = make_fact(predicate="revenue_from_services")
    b = make_fact(predicate="revenue_from_operations", raw="81,420.00 mn")
    assert not comparison_keys_match(a, b)
    assert comparison_keys_match(a, b, {"revenue_from_operations": "revenue_from_services"})


# --- row 2: keys match, bags compatible, intervals overlap -----------------

def test_same_number_at_different_scales_corroborates():
    decision = decide(
        make_fact(raw="8,142 Cr", period="FY 2023-24"),
        make_fact(raw="81,415.38 mn", doc="d2", period="FY 2023-24"),
    )
    assert decision.verdict == "corroborates"
    assert decision.rule_fired == "keys_match_intervals_overlap"
    assert not decision.needs_llm


def test_corroboration_survives_a_qualifier_only_one_side_states():
    """The missing-qualifier guard must not turn agreement into abstention.

    A figure the other document simply did not qualify is the normal case for
    a summary deck against a full filing; if this demoted to
    insufficient_context, cross-document corroboration would never land.
    """
    decision = decide(
        make_fact(raw="8,142 Cr", period="FY 2023-24"),
        make_fact(raw="81,415.38 mn", doc="d2", period="FY 2023-24", consolidation="consolidated"),
    )
    assert decision.verdict == "corroborates"
    assert decision.missing_keys == ("consolidation",)
    assert "consolidation" in decision.explanation


def test_non_discriminating_qualifier_difference_still_corroborates():
    decision = decide(
        make_fact(raw="8,142 Cr", period="FY 2023-24", source_table="Note 21"),
        make_fact(raw="81,415.38 mn", doc="d2", period="FY 2023-24", source_table="Highlights"),
    )
    assert decision.verdict == "corroborates"


# --- row 3: values agree but the conditions differ -------------------------

def test_agreeing_values_under_different_periods_are_not_corroboration():
    decision = decide(
        make_fact(raw="8,142 Cr", period="FY 2023-24"),
        make_fact(raw="8,142 Cr", doc="d2", period="FY 2022-23"),
    )
    assert decision.verdict == "unrelated"
    assert decision.rule_fired == "context_differs_values_agree"


# --- row 4: missing-qualifier guard ----------------------------------------

def test_disagreement_with_a_qualifier_missing_on_one_side_abstains():
    decision = decide(
        make_fact(raw="8,142 Cr", period="FY 2023-24"),
        make_fact(raw="7,224 Cr", doc="d2", period="FY 2023-24", consolidation="standalone"),
    )
    assert decision.verdict == "insufficient_context"
    assert decision.rule_fired == "missing_qualifier_guard"
    assert decision.qualifier_key == "consolidation"
    assert not decision.needs_llm


def test_the_guard_names_which_side_is_missing_the_key():
    decision = decide(
        make_fact(raw="8,142 Cr", period="FY 2023-24", consolidation="consolidated"),
        make_fact(raw="7,224 Cr", doc="d2", period="FY 2023-24"),
    )
    assert decision.qualifier_key == "consolidation"
    assert "the first" in decision.explanation


# --- row 5: reconciled by context ------------------------------------------

def test_disagreement_explained_by_a_differing_qualifier():
    decision = decide(
        make_fact(raw="8,142 Cr", period="FY 2023-24", consolidation="consolidated"),
        make_fact(raw="7,224 Cr", doc="d2", period="FY 2023-24", consolidation="standalone"),
    )
    assert decision.verdict == "reconciled_by_context"
    assert decision.qualifier_key == "consolidation"
    assert decision.needs_llm  # prose only; the verdict is already decided


def test_different_periods_reconcile_rather_than_contradict():
    decision = decide(
        make_fact(raw="8,142 Cr", period="FY 2023-24"),
        make_fact(raw="7,224 Cr", doc="d2", period="FY 2022-23"),
    )
    assert decision.verdict == "reconciled_by_context"
    assert decision.qualifier_key == "period"


def test_overlapping_but_unequal_periods_are_a_context_difference():
    """Nine months inside a year is a different fact, not a conflicting one."""
    decision = decide(
        make_fact(raw="6,100 Cr", period="nine months ended December 31, 2024"),
        make_fact(raw="8,142 Cr", doc="d2", period="year ended March 31, 2025"),
    )
    assert decision.verdict == "reconciled_by_context"
    assert decision.qualifier_key == "period"


# --- row 6: contradicts ----------------------------------------------------

def test_same_context_and_disjoint_intervals_contradicts():
    decision = decide(
        make_fact(raw="8,142 Cr", period="FY 2023-24", consolidation="consolidated"),
        make_fact(raw="7,224 Cr", doc="d2", period="FY 2023-24", consolidation="consolidated"),
    )
    assert decision.verdict == "contradicts"
    assert decision.rule_fired == "same_context_disjoint_intervals"
    assert decision.needs_llm  # the only row that reaches an adjudicating call


def test_two_bare_facts_with_disjoint_values_contradict():
    decision = decide(make_fact(raw="740"), make_fact(raw="812", doc="d2"))
    assert decision.verdict == "contradicts"


# --- precision intervals ---------------------------------------------------

def test_rounding_does_not_produce_a_contradiction():
    """8,142 Cr committed to whole crore; the millions figure sits inside it."""
    outcome, _ = compare_values(make_fact(raw="8,142 Cr"), make_fact(raw="81,415.38 mn"))
    assert outcome == AGREE


def test_a_real_difference_still_disagrees_at_the_same_precision():
    outcome, why = compare_values(make_fact(raw="8,142 Cr"), make_fact(raw="8,143 Cr"))
    assert outcome == DISAGREE
    assert "disjoint" in why


def test_bounded_figures_compare_as_half_open_intervals():
    outcome, _ = compare_values(make_fact(raw="over 33,200"), make_fact(raw="33,412"))
    assert outcome == AGREE


def test_currencies_are_never_converted():
    outcome, why = compare_values(
        make_fact(raw="$970 mn"), make_fact(raw="Rs 81,415.38 mn", doc="d2")
    )
    assert outcome == INCOMPARABLE
    assert "conversion" in why


def test_a_percentage_is_not_an_absolute_quantity():
    outcome, _ = compare_values(make_fact(raw="8.2%"), make_fact(raw="8.2 mn"))
    assert outcome == INCOMPARABLE


def test_incomparable_values_abstain_rather_than_contradict():
    decision = decide(
        make_fact(raw="$970 mn", period="FY 2023-24"),
        make_fact(raw="Rs 81,415.38 mn", doc="d2", period="FY 2023-24"),
    )
    assert decision.verdict == "insufficient_context"
    assert decision.rule_fired == "incomparable_values"


# --- non-quantity values ---------------------------------------------------

def test_text_values_compare_after_normalisation():
    a = make_fact(raw="Resigned", kind="text", predicate="directorship_status")
    b = make_fact(raw="resigned.", kind="text", predicate="directorship_status", doc="d2")
    assert compare_values(a, b)[0] == AGREE


def test_dates_are_compared_as_dates_not_strings():
    a = make_fact(raw="March 31, 2024", kind="date", predicate="appointment_date")
    b = make_fact(raw="31 March 2024", kind="date", predicate="appointment_date", doc="d2")
    assert compare_values(a, b)[0] == AGREE
    c = make_fact(raw="April 1, 2024", kind="date", predicate="appointment_date", doc="d3")
    assert compare_values(a, c)[0] == DISAGREE


def test_a_quantity_and_a_text_value_are_not_compared():
    a = make_fact(raw="740", kind="quantity", predicate="headcount")
    b = make_fact(raw="seven hundred", kind="text", predicate="headcount", doc="d2")
    assert compare_values(a, b)[0] == INCOMPARABLE


# --- the qualifier bag itself ----------------------------------------------

@pytest.mark.parametrize(
    "key,expected",
    [
        ("period", True),
        ("reporting_period", True),
        ("geography", True),
        ("as_of", True),
        ("source_table", False),
        ("rounding", False),
    ],
)
def test_which_qualifier_keys_can_change_a_verdict(key, expected):
    assert is_discriminating(key) is expected


def test_period_qualifiers_compare_by_interval_not_by_wording():
    diff = compare_qualifiers(
        make_fact(period="FY24"), make_fact(period="year ended March 31, 2024", doc="d2")
    )
    assert diff.agreeing == ["period"]
    assert diff.conflicting == []


def test_inherited_qualifiers_count_as_stated():
    """A scale from a table header is known context, not a guess."""
    diff = compare_qualifiers(
        make_fact(consolidation=("consolidated", "inherited")),
        make_fact(consolidation=("consolidated", "stated"), doc="d2"),
    )
    assert diff.agreeing == ["consolidation"]
    assert diff.missing == []
