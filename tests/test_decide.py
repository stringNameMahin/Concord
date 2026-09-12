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


def test_a_trailing_legal_form_is_the_same_subject():
    assert subjects_match(
        make_fact(subject="Acme"), make_fact(subject="Acme Limited")
    )
    assert not subjects_match(
        make_fact(subject="Acme Logistics"), make_fact(subject="Borealis Freight")
    )


def test_a_longer_name_is_a_different_subject_not_a_shorter_one():
    """The subset rule this replaced let a parent absorb its subsidiaries."""
    assert not subjects_match(
        make_fact(subject="Acme Limited"),
        make_fact(subject="Acme Freight Services Private Limited"),
    )
    assert not subjects_match(
        make_fact(subject="Acme Limited"), make_fact(subject="Acme Corp Limited")
    )
    assert not subjects_match(
        make_fact(subject="cash and cash equivalents"),
        make_fact(subject="bank balances other than cash and cash equivalents"),
    )
    assert not subjects_match(
        make_fact(subject="current other assets"),
        make_fact(subject="current other financial assets"),
    )


def test_extra_words_that_are_not_a_legal_form_are_a_different_subject():
    assert not subjects_match(
        make_fact(subject="Acme"), make_fact(subject="Acme gateways")
    )
    assert not subjects_match(
        make_fact(subject="Company"), make_fact(subject="Acme Limited")
    )


def test_only_one_legal_form_may_be_stripped():
    """`Acme Corp Limited` is a company in its own right, not `Acme`."""
    assert not subjects_match(
        make_fact(subject="Acme"), make_fact(subject="Acme Corp Limited")
    )


@pytest.mark.parametrize(
    "parent, subsidiary, parent_key, subsidiary_key",
    [
        # A subsidiary whose name extends the parent's, no hard key on either
        # side. This is the shape the old subset rule could not reject.
        ("Acme Limited", "Acme Freight Services Private Limited", None, None),
        # The same shape with distinct company identifiers on both sides.
        (
            "Acme Limited",
            "Acme Cross Border Services Private Limited",
            "CIN:U00000AA2011PLC000001",
            "CIN:U00000AA2015PTC000002",
        ),
        # A foreign subsidiary, where the only extra word is a legal form.
        ("Acme Limited", "Acme Corp Limited", None, "UKCRN:00000001"),
    ],
)
def test_a_subsidiary_never_compares_against_its_parent(
    parent, subsidiary, parent_key, subsidiary_key
):
    """An incorporation date has no reason to match across two companies.

    The predicate is incidental: the pair must be rejected on the subject,
    before any value or qualifier is looked at, whatever is being compared.
    """
    decision = decide(
        make_fact(
            subject=parent,
            subject_key=parent_key,
            predicate="date_of_incorporation",
            kind="date",
            raw="June 22, 2011",
        ),
        make_fact(
            subject=subsidiary,
            subject_key=subsidiary_key,
            predicate="date_of_incorporation",
            kind="date",
            raw="April 21, 2020",
        ),
    )
    assert decision.verdict == "unrelated"
    assert decision.rule_fired == "comparison_key_differs"


def test_alias_map_makes_two_predicates_one_key():
    a = make_fact(predicate="revenue_from_services")
    b = make_fact(predicate="revenue_from_operations", raw="81,420.00 mn")
    assert not comparison_keys_match(a, b)
    assert comparison_keys_match(a, b, {"revenue_from_operations": "revenue_from_services"})


def test_a_segment_inherited_from_the_layout_blocks_a_contradiction():
    """The missing-qualifier guard binds on inherited keys as on stated ones.

    A figure sitting under a bullet that names one business line, and a
    company-wide figure elsewhere in the same document, are not a conflict.
    The engine cannot say they agree either, so it names the key it is
    missing rather than choosing between them.
    """
    segmented = make_fact(
        raw="7,900",
        predicate="active_customers",
        subject="Acme",
        period="nine months ended December 31, 2021",
        segment=("Express Parcel", "inherited"),
    )
    company_wide = make_fact(
        raw="23,113",
        predicate="active_customers",
        subject="Acme",
        period="nine months ended December 31, 2021",
    )
    decision = decide(segmented, company_wide)
    assert decision.verdict == "insufficient_context"
    assert decision.rule_fired == "missing_qualifier_guard"
    assert decision.qualifier_key == "segment"


def test_two_different_inherited_segments_reconcile_rather_than_conflict():
    a = make_fact(
        raw="20,498", predicate="active_customers", subject="Acme",
        period="FY24", segment=("Express Parcel", "inherited"),
    )
    b = make_fact(
        raw="77", predicate="active_customers", subject="Acme",
        period="FY24", segment=("Supply Chain Services", "inherited"),
    )
    decision = decide(a, b)
    assert decision.verdict == "reconciled_by_context"
    assert decision.qualifier_key == "segment"


def test_two_spellings_of_one_period_do_not_break_a_corroboration():
    """Neither label resolves, so their wording is not evidence of a difference.

    Both facts state the same cumulative total; one document writes the period
    as "till FY26" and the other as "inception till FY26".
    """
    a = make_fact(raw="290 million", predicate="meals_served", period="till FY26")
    b = make_fact(
        raw="290 million", predicate="meals_served", period="inception till FY26", doc="d2"
    )
    decision = decide(a, b)
    assert decision.verdict == "corroborates"
    assert "spell period differently" in decision.explanation


def test_unresolved_periods_naming_different_years_still_differ():
    a = make_fact(raw="210.8", predicate="trade_deficit", period="April-December 2024")
    b = make_fact(raw="210.8", predicate="trade_deficit", period="April-December 2023", doc="d2")
    decision = decide(a, b)
    assert decision.verdict == "unrelated"
    assert decision.rule_fired == "context_differs_values_agree"


def test_a_resolved_period_difference_still_breaks_agreement():
    """The relaxation only covers labels the parser could not read."""
    a = make_fact(raw="8,142 Cr", period="FY24")
    b = make_fact(raw="8,142 Cr", period="FY23", doc="d2")
    decision = decide(a, b)
    assert decision.verdict == "unrelated"
    assert decision.rule_fired == "context_differs_values_agree"


def test_an_unreadable_period_does_not_rescue_a_disagreement():
    """The relaxation binds on agreement only; disagreement still reconciles."""
    a = make_fact(raw="290 million", predicate="meals_served", period="till FY26")
    b = make_fact(raw="180 million", predicate="meals_served", period="inception till FY26")
    decision = decide(a, b)
    assert decision.verdict == "reconciled_by_context"


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


@pytest.mark.parametrize("coined", ["as_of_date", "as_at_date", "reporting_period"])
def test_the_guard_recognises_keys_the_extractor_coined(coined):
    """Observed on the real deck: the model wrote `as_of_date`, not `as_of`."""
    assert is_discriminating(coined)


def test_a_qualifier_named_date_alone_is_not_a_condition():
    assert not is_discriminating("date")
    assert not is_discriminating("report_date")


# --- open-vocabulary conditions, from the starter corpus --------------------
# Each of these was a false contradiction until the disagreement branch stopped
# consulting the allowlist. The keys are the ones the extractor actually coined.

@pytest.mark.parametrize(
    "key,left,right",
    [
        ("service", "express parcel delivery services", "supply chain services solutions"),
        ("category", "Workers", "Employees"),
        ("auditor", "S.R. Batliboi & Associates LLP", "Deloitte Haskins & Sells LLP"),
        ("condition", "prior to one year", "after one year but prior to two years"),
    ],
)
def test_a_coined_condition_reconciles_a_disagreement(key, left, right):
    decision = decide(
        make_fact(raw="20,498", period="FY 2023-24", **{key: left}),
        make_fact(raw="77", doc="d2", period="FY 2023-24", **{key: right}),
    )
    assert decision.verdict == "reconciled_by_context"
    assert decision.qualifier_key == key


def test_a_recognised_condition_is_named_ahead_of_an_incidental_one():
    decision = decide(
        make_fact(raw="8,142 Cr", period="FY 2023-24", source_table="Note 21"),
        make_fact(raw="7,224 Cr", doc="d2", period="FY 2022-23", source_table="Highlights"),
    )
    assert decision.qualifier_key == "period"


def test_an_incidental_key_still_cannot_split_two_agreeing_figures():
    """The allowlist keeps binding here: same value, same fact."""
    decision = decide(
        make_fact(raw="8,142 Cr", period="FY 2023-24", source_table="Note 21"),
        make_fact(raw="81,415.38 mn", doc="d2", period="FY 2023-24", source_table="Highlights"),
    )
    assert decision.verdict == "corroborates"


def test_facts_with_no_conditions_at_all_still_contradict():
    """Observed live: two different CINs, neither document qualifying either."""
    decision = decide(
        make_fact(raw="L63090DL2011PLC221234", kind="text", predicate="corporate_identity_number"),
        make_fact(
            raw="U63090DL2011PLC221234",
            kind="text",
            predicate="corporate_identity_number",
            doc="d2",
        ),
    )
    assert decision.verdict == "contradicts"


def test_a_coined_condition_on_one_side_only_forces_abstention():
    """`service` on one fact and not the other: we cannot tell, so we say so.

    Observed live: seven prospectus pairs counting active customers per service
    line against a company-wide total were called contradictions.
    """
    decision = decide(
        make_fact(raw="20,498", period="Fiscal 2021", service="express parcel"),
        make_fact(raw="449", doc="d2", period="Fiscal 2021"),
    )
    assert decision.verdict == "insufficient_context"
    assert decision.rule_fired == "missing_qualifier_guard"
    assert decision.qualifier_key == "service"


def test_a_recognised_missing_key_is_named_before_a_coined_one():
    decision = decide(
        make_fact(raw="20,498", period="Fiscal 2021", service="express parcel"),
        make_fact(raw="449", doc="d2"),
    )
    assert decision.qualifier_key == "period"


def test_contradiction_now_means_the_same_stated_conditions():
    """The rule the README states: same conditions, still disagreeing."""
    decision = decide(
        make_fact(raw="7,900", period="nine months ended December 31, 2021"),
        make_fact(raw="23,113", doc="d2", period="nine months ended December 31, 2021"),
    )
    assert decision.verdict == "contradicts"


# --- consolidated against standalone -----------------------------------------

def test_the_same_line_item_on_two_bases_is_reconciled_not_contradicted():
    """The defect F18 records, from the comparison side.

    A group and its parent report the same line item for the same year under the
    same subject, and the figures differ by a factor of five with both correct.
    Before the basis was captured both bags read `{period}` alone, so the table
    returned `contradicts` and only the LLM stood between that and the ledger.
    """
    consolidated = make_fact(
        raw="615", predicate="profit_before_tax", unit="INR", scale="crore",
        period="year ended March 31, 2026",
        consolidation=("consolidated", "inherited"),
    )
    standalone = make_fact(
        raw="2,966", predicate="profit_before_tax", unit="INR", scale="crore",
        period="year ended March 31, 2026",
        consolidation=("standalone", "inherited"),
    )
    decision = decide(consolidated, standalone)
    assert decision.verdict == "reconciled_by_context"
    assert decision.qualifier_key == "consolidation"


def test_a_basis_on_one_side_only_still_abstains():
    """The missing-qualifier guard binds on the basis like any other key."""
    labelled = make_fact(
        raw="615", predicate="profit_before_tax", unit="INR", scale="crore",
        period="year ended March 31, 2026",
        consolidation=("consolidated", "inherited"),
    )
    bare = make_fact(
        raw="2,966", predicate="profit_before_tax", unit="INR", scale="crore",
        period="year ended March 31, 2026",
    )
    decision = decide(labelled, bare)
    assert decision.verdict == "insufficient_context"
    assert decision.qualifier_key == "consolidation"


def test_figures_that_agree_across_the_two_bases_still_corroborate():
    """The basis explains a disagreement; it does not split an agreement.

    Consolidated and standalone statements are two reports of one entity over
    one year. A registered office, a CIN or an incorporation date restated in
    both is one fact stated twice, and calling it `unrelated` because the
    enclosing statement differs would lose a correct verdict to buy nothing.
    """
    a = make_fact(
        raw="1,781", predicate="carrying_value_of_investments", unit="INR", scale="crore",
        period="year ended March 31, 2026",
        consolidation=("consolidated", "inherited"),
    )
    b = make_fact(
        raw="1,781", predicate="carrying_value_of_investments", unit="INR", scale="crore",
        period="year ended March 31, 2026",
        consolidation=("standalone", "inherited"),
    )
    assert decide(a, b).verdict == "corroborates"


def test_a_recognised_condition_other_than_the_basis_still_splits_agreement():
    """The exemption is for the basis only; `segment` behaves as before."""
    a = make_fact(raw="1,781", predicate="revenue", unit="INR", scale="crore",
                  segment="Express Parcel")
    b = make_fact(raw="1,781", predicate="revenue", unit="INR", scale="crore",
                  segment="Cross-Border")
    assert decide(a, b).verdict == "unrelated"


# --- the unit no longer carries the magnitude -------------------------------

def test_one_figure_written_twice_is_not_blocked_by_its_column_header():
    """The defect F22 records.

    The model writes the column header verbatim, so one side's unit is `INR` and
    the other's `INR crore` while both magnitudes are already applied. Compared
    as strings that refused 12 of the 14 `incomparable_values` relations on the
    shipped ledger - including the same figure against itself.
    """
    # Both magnitudes are already applied - this is the shipped shape, where the
    # scale is on the `Quantity` and the word is *also* still in the unit field.
    header = make_fact(raw="54,364", predicate="revenue_from_operations",
                       unit="INR crore", scale="crore", period="FY26")
    cell = make_fact(raw="54,364", predicate="revenue_from_operations",
                     unit="INR", scale="crore", period="FY26")
    assert header.quantity.unit == cell.quantity.unit == "INR"
    assert decide(header, cell).verdict == "corroborates"


def test_two_genuinely_different_currencies_are_still_refused():
    """Stripping the magnitude must not start converting money."""
    rupees = make_fact(raw="8,142", predicate="revenue", unit="INR crore")
    dollars = make_fact(raw="8,142", predicate="revenue", unit="USD billion")
    decision = decide(rupees, dollars)
    assert decision.verdict == "insufficient_context"
    assert decision.rule_fired == "incomparable_values"
    assert "INR and USD" in decision.explanation
