"""The seam between extraction and comparison.

`materialize` is where the model's output stops being a claim and becomes a
record the engine can compare: the figure parsed into base units, the period
label resolved to an interval, the qualifier bag carrying provenance.
"""

from concord.extract.align import Alignment
from concord.extract.models import FactOut, QualifierOut, ValueOut
from concord.facts import (
    canonical_predicate,
    make_fact_id,
    materialize,
    normalize_surface,
)


def extraction(**overrides):
    payload = dict(
        passage_id=3,
        claim_text="Acme Logistics Limited reported revenue from services of 81,415.38 million.",
        subject_surface="Acme Logistics Limited",
        subject_key="CIN:X1",
        predicate="Revenue From Services",
        value=ValueOut(kind="quantity", raw="81,415.38", unit=None, scale=None),
        qualifiers=[
            QualifierOut(key="period", value="year ended March 31, 2024", provenance="inherited"),
            QualifierOut(key="Consolidation", value="consolidated", provenance="inherited"),
        ],
        quote="Revenue from services* 81,415.38 72,236.47",
        confidence=0.9,
    )
    payload.update(overrides)
    return FactOut(**payload)


def alignment(start=481203):
    quote = "Revenue from services* 81,415.38 72,236.47"
    return Alignment("exact", start, start + len(quote), quote, 100.0)


def test_the_context_stack_supplies_a_scale_the_cell_does_not_state():
    """A bare table cell under an 'in million' header is a million-scale figure."""
    fact = materialize(extraction(), "d2", alignment(), page=214, default_scale="million")
    assert fact.quantity.normalized == 81415.38 * 1e6
    assert fact.quantity.scale == "million"


def test_a_scale_written_on_the_figure_beats_the_inherited_one():
    fact = materialize(
        extraction(value=ValueOut(kind="quantity", raw="8,142 Cr")),
        "d1",
        alignment(),
        page=4,
        default_scale="million",
    )
    assert fact.quantity.scale == "cr"


def test_period_labels_become_intervals_at_materialisation():
    fact = materialize(extraction(), "d2", alignment(), page=214)
    period = fact.qualifier("period").period
    assert (period.start.isoformat(), period.end.isoformat()) == ("2023-04-01", "2024-03-31")


def test_qualifier_keys_are_canonicalised_and_keep_their_tier():
    fact = materialize(extraction(), "d2", alignment(), page=214)
    assert set(fact.qualifier_keys()) == {"period", "consolidation"}
    assert fact.qualifier("consolidation").provenance == "inherited"


def test_an_unrecorded_qualifier_answers_absent_rather_than_empty():
    fact = materialize(extraction(), "d2", alignment(), page=214)
    missing = fact.qualifier("segment")
    assert missing.provenance == "absent"
    assert not missing.known


def test_a_figure_that_will_not_parse_is_flagged_not_dropped():
    fact = materialize(
        extraction(value=ValueOut(kind="quantity", raw="Nil")), "d2", alignment(), page=1
    )
    assert fact.quantity is None
    assert "unparsed_quantity" in fact.flags
    assert fact.value_raw == "Nil"


def test_the_hard_key_wins_over_the_surface_form():
    fact = materialize(extraction(), "d2", alignment(), page=214)
    assert fact.comparison_key == "CIN:X1|revenue_from_services"


def test_without_a_hard_key_the_surface_form_carries_the_identity():
    fact = materialize(extraction(subject_key=None), "d2", alignment(), page=214)
    assert fact.comparison_key == "acme logistics limited|revenue_from_services"


def test_fact_ids_are_content_addressed_so_reingestion_is_idempotent():
    """Phase 7 appends documents; a run-order id would make that impossible."""
    first = materialize(extraction(), "d2", alignment(), page=214)
    again = materialize(extraction(), "d2", alignment(), page=214)
    elsewhere = materialize(extraction(), "d2", alignment(start=9), page=214)
    assert first.fact_id == again.fact_id != elsewhere.fact_id


def test_evidence_keeps_the_located_text_not_the_models_wording():
    located = Alignment("fuzzy", 100, 141, "Revenue from services*  81,415.38  72,236.47", 92.0)
    fact = materialize(extraction(), "d2", located, page=214)
    assert fact.evidence.quote == located.text
    assert fact.evidence.align_status == "fuzzy"


def test_predicate_and_surface_normalisation_are_case_and_punctuation_only():
    assert canonical_predicate("Revenue From Services*") == "revenue_from_services"
    assert normalize_surface("Acme Logistics Limited's") == "acme logistics limited"


def test_qualifier_keys_the_extractor_coins_still_match_known_conditions():
    """The bag is open, so the same condition arrives under several names."""
    from concord.facts import PERIOD_KEYS, key_matches

    for coined in ("as_of", "as_of_date", "as_at_date", "reporting_period"):
        assert key_matches(coined, PERIOD_KEYS)
    for unrelated in ("date", "scale", "membership_number"):
        assert not key_matches(unrelated, PERIOD_KEYS)


def test_a_date_qualifier_under_a_coined_key_is_still_parsed_as_a_period():
    fact = materialize(
        extraction(
            qualifiers=[
                QualifierOut(
                    key="as_of_date", value="March 31, 2024", provenance="stated"
                )
            ]
        ),
        "d2",
        alignment(),
        page=1,
    )
    assert fact.qualifier("as_of_date").period is not None


def test_a_row_label_used_as_the_subject_is_flagged():
    """Bug 23: the model names the metric twice and the entity never.

    `Adjusted EBITDA / adjusted_ebitda` grounds cleanly - the quote locates,
    the figure parses - and is still malformed, because the subject is half of
    the comparison key and two unrelated organisations' EBITDA rows will block
    together. Prompt rule 8 forbids it in words; this is the check in code.
    """
    fact = materialize(
        extraction(subject_surface="Adjusted EBITDA", predicate="adjusted_ebitda"),
        "d1",
        alignment(),
        page=12,
    )
    assert "subject_names_the_measurement" in fact.flags


def test_the_row_label_guard_reads_words_and_not_the_exact_string():
    """`Debt/Equity` against `debt_to_equity_ratio` is the same defect.

    The predicate is not the subject slugged, so an equality test misses it.
    Every word of the subject already being in the predicate is what the
    doubled row label actually looks like.
    """
    fact = materialize(
        extraction(subject_surface="Debt/Equity", predicate="debt_to_equity_ratio"),
        "d1",
        alignment(),
        page=12,
    )
    assert "subject_names_the_measurement" in fact.flags


def test_a_real_subject_is_not_flagged():
    """The guard has to stay quiet on the ordinary case, which is most of them.

    Rule 7 already bars the entity from the predicate, so a well-formed fact
    cannot have its subject contained in one. Measured on the shipped ledger:
    7 of 506 flagged, all seven genuine.
    """
    assert not materialize(extraction(), "d1", alignment(), page=214).flags


def test_a_flagged_fact_is_still_a_fact():
    """Flagged, not quarantined. The value, the quote and the qualifiers are
    all sound; only the subject is wrong, and a reviewer is the one who should
    decide what that is worth. Quarantine means the quote never located, and
    overloading it would make the grounding metric mean two things."""
    fact = materialize(
        extraction(subject_surface="Total income", predicate="total_income"),
        "d1",
        alignment(),
        page=12,
    )
    assert fact.evidence.align_status == "exact"
    assert fact.quantity is not None
