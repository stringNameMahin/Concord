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
    """A bare table cell under an 'in million' header is a million-scale figure.

    Neither the figure, the quote nor the sentence says `million` here, so the
    caller's context is the only source left and this test fails if it is
    ignored. The fixture's own claim text does state it, which would let this
    pass for the wrong reason.
    """
    bare = extraction(claim_text="Acme Logistics Limited reported revenue of 81,415.38.")
    fact = materialize(bare, "d2", alignment(), page=214, default_scale="million")
    assert fact.quantity.normalized == 81415.38 * 1e6
    assert fact.quantity.scale == "million"
    assert "scale_recovered_from_source" not in fact.flags


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
    fact = materialize(extraction(), "d1", alignment(), page=214)
    assert "subject_names_the_measurement" not in fact.flags


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


def test_a_scale_the_model_dropped_is_read_back_out_of_the_located_bytes():
    """The bug this guard exists for, in its commonest shape.

    A model handed `Total cash balance: Rs 5,444 Cr` returns `raw="5,444"` and
    leaves `scale` null, and the record then says five thousand rupees instead
    of fifty-four billion. Nothing downstream can tell: the quote is right, the
    displayed value is right, and only `normalized` - which every interval
    comparison uses - is wrong. Measured over the seven-document ledger before
    this existed: 133 of 728 quantity facts.
    """
    quote = "Total cash balance: Rs 5,444 Cr"
    fact = materialize(
        extraction(
            claim_text="The total cash balance was 5,444.",
            value=ValueOut(kind="quantity", raw="5,444", unit=None, scale=None),
            quote=quote,
        ),
        "d1",
        Alignment("exact", 100, 100 + len(quote), quote, 100.0),
        page=2,
    )
    assert fact.quantity.scale == "cr"
    assert fact.quantity.normalized == 5444 * 1e7
    assert "scale_recovered_from_source" in fact.flags


def test_the_sentence_is_read_when_the_quote_is_a_bare_cell():
    """A table cell quotes as the figure alone; the claim still carries it."""
    fact = materialize(
        extraction(
            claim_text="Revenue from services was 81,415.38 million for the year.",
            quote="81,415.38",
        ),
        "d1",
        Alignment("exact", 10, 19, "81,415.38", 100.0),
        page=2,
    )
    assert fact.quantity.scale == "million"
    assert fact.quantity.normalized == 81415.38 * 1e6


def test_a_scale_word_belonging_to_another_figure_is_not_borrowed():
    """The guard has to fail towards silence. A wrong magnitude is worse than
    a missing one, so a scale word that is not directly behind *this* figure is
    not evidence about it - the quote below states a second figure in millions
    and the first is a plain count."""
    quote = "across 13,087 PIN codes with 2.85 million shipments"
    fact = materialize(
        extraction(
            claim_text="The network covered 13,087 PIN codes.",
            value=ValueOut(kind="quantity", raw="13,087", unit=None, scale=None),
            quote=quote,
        ),
        "d1",
        Alignment("exact", 0, len(quote), quote, 100.0),
        page=2,
    )
    assert fact.quantity.scale is None
    assert fact.quantity.normalized == 13087.0
    assert "scale_recovered_from_source" not in fact.flags


def test_what_the_model_states_still_wins_over_what_is_read_back():
    """`value.scale` is structured output about this figure specifically. The
    byte-level read is the fallback for when it is missing, not a correction."""
    quote = "Revenue 5,444 Cr"
    fact = materialize(
        extraction(
            value=ValueOut(kind="quantity", raw="5,444", unit=None, scale="million"),
            quote=quote,
        ),
        "d1",
        Alignment("exact", 0, len(quote), quote, 100.0),
        page=2,
    )
    assert fact.quantity.scale == "million"
    assert "scale_recovered_from_source" not in fact.flags


def test_the_two_units_fields_are_read_by_content_not_by_position():
    """Observed on the RBI report: the model filled them exactly backwards.

    It returned `scale="Rs"` and `unit="lakh crore"` for five figures. Read by
    position that stores a rupee sign as the magnitude and a magnitude as the
    unit, so the value comes out unscaled - and the bogus unit then refuses
    every comparison the fact could have taken part in. Both fields are free
    text, so neither can be trusted by where it sits.
    """
    quote = "NFA expanded by Rs 2.8 lakh crore"
    fact = materialize(
        extraction(
            claim_text="During 2024-25, NFA expanded by Rs 2.8 lakh crore.",
            value=ValueOut(kind="quantity", raw="2.8", unit="lakh crore", scale="Rs"),
            quote=quote,
        ),
        "d1",
        Alignment("exact", 0, len(quote), quote, 100.0),
        page=2,
    )
    assert fact.quantity.scale == "lakh crore"
    assert fact.quantity.normalized == 2.8e12
    assert fact.quantity.unit != "lakh crore"


def test_a_model_scale_that_names_no_magnitude_does_not_shadow_the_document():
    """Fail towards the evidence: an unusable field must not block the read."""
    quote = "Total cash balance: Rs 5,444 Cr"
    fact = materialize(
        extraction(
            claim_text="The total cash balance was 5,444.",
            value=ValueOut(kind="quantity", raw="5,444", unit=None, scale="Rs"),
            quote=quote,
        ),
        "d1",
        Alignment("exact", 0, len(quote), quote, 100.0),
        page=2,
    )
    assert fact.quantity.scale == "cr"
    assert fact.quantity.normalized == 5444 * 1e7


def test_a_scale_the_model_read_only_half_of_loses_to_the_document():
    """The exception to the model winning: it stopped reading mid-phrase.

    `lakh` and `lakh crore` are two magnitudes a factor of ten million apart,
    and both parse on their own, so the mistake is silent. A model that returns
    `scale="lakh"` for a page reading `Rs 1.1 lakh crore` has not disagreed with
    the document, it has read one word of two - so the longer phrase the page
    actually wrote wins. Measured in the shipped ledger before this: cumulative
    FPI debt flows of Rs 1.1 lakh crore stored as 110,000.
    """
    quote = "cumulative flows of Rs 1.1 lakh crore from October 2023 to June 2024."
    fact = materialize(
        extraction(
            claim_text="Cumulative FPI debt flows were Rs 1.1 lakh crore.",
            value=ValueOut(kind="quantity", raw="1.1", unit=None, scale="lakh"),
            quote=quote,
        ),
        "d1",
        Alignment("exact", 0, len(quote), quote, 100.0),
        page=2,
    )
    assert fact.quantity.scale == "lakh crore"
    assert fact.quantity.normalized == 1.1e12


def test_reading_the_compound_late_loses_to_the_document_too():
    """`crore` for `lakh crore` is out by a hundred thousand, the same shape."""
    quote = "NFA expanded by Rs 2.8 lakh crore during the year."
    fact = materialize(
        extraction(
            claim_text="Net foreign assets expanded by Rs 2.8 lakh crore.",
            value=ValueOut(kind="quantity", raw="2.8", unit=None, scale="crore"),
            quote=quote,
        ),
        "d1",
        Alignment("exact", 0, len(quote), quote, 100.0),
        page=2,
    )
    assert fact.quantity.scale == "lakh crore"
    assert fact.quantity.normalized == 2.8e12


def test_a_percentage_is_not_scaled_by_a_figure_beside_it():
    """End to end, the defect F20 records.

    The sentence states an absolute in billions and a growth rate in per cent.
    The rate's digit occurs inside the absolute, so a boundary-blind read handed
    it `billion` and the ledger carried six per cent as six billion.
    """
    quote = (
        "exports have shown positive momentum, reaching USD 602.6 billion, "
        "witnessing a YoY growth of 6 per cent."
    )
    fact = materialize(
        extraction(
            claim_text="Total exports grew by 6 per cent in the first nine months of FY25.",
            value=ValueOut(kind="quantity", raw="6", unit="per cent", scale=None),
            quote=quote,
        ),
        "d1",
        Alignment("exact", 0, len(quote), quote, 100.0),
        page=2,
    )
    assert fact.quantity.is_percent is True
    assert fact.quantity.scale is None
    assert fact.quantity.normalized == 6.0
    assert "scale_recovered_from_source" not in fact.flags
