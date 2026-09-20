"""Blocking: does the candidate set contain the pairs that matter, and how
much of the O(n^2) space did it cost to get them."""

import itertools

import numpy as np
import pytest
from factories import make_fact

from concord.compare.block import (
    BlockingStats,
    block,
    comparison_key_pairs,
    pair_key,
    semantic_pairs,
    value_pairs,
)
from concord.compare.decide import decide
from concord.compare.embed import top_k, unit_rows


class BagOfWords:
    """A deterministic stand-in for the sentence encoder.

    Word overlap is a poor semantic model and a perfectly good test double:
    it lets the blocking logic be tested without loading torch, and the real
    encoder is exercised separately below.
    """

    def encode(self, texts):
        vocabulary = sorted({word for text in texts for word in text.lower().split()})
        index = {word: position for position, word in enumerate(vocabulary)}
        matrix = np.zeros((len(texts), max(len(vocabulary), 1)), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                matrix[row, index[word]] += 1.0
        return unit_rows(matrix)


def ids(pairs):
    return {tuple(sorted(pair)) for pair in pairs}


def test_pair_key_is_orderless():
    a, b = make_fact(fact_id="f_b"), make_fact(fact_id="f_a")
    assert pair_key(a, b) == pair_key(b, a) == ("f_a", "f_b")


# --- strategy 1: comparison key --------------------------------------------

def test_comparison_key_block_groups_on_subject_and_predicate():
    a = make_fact(fact_id="f_a", raw="8,142 Cr")
    b = make_fact(fact_id="f_b", raw="7,224 Cr", doc="d2")
    other = make_fact(fact_id="f_c", predicate="employee_count", raw="8,142")
    pairs = comparison_key_pairs([a, b, other])
    assert ids(pairs) == {("f_a", "f_b")}


def test_comparison_key_block_is_quadratic_only_within_a_group():
    facts = [make_fact(fact_id=f"f_{i}", raw=f"{100 + i} Cr") for i in range(5)]
    assert len(comparison_key_pairs(facts)) == 10


# --- strategy 2: semantic ---------------------------------------------------

def test_semantic_block_pairs_paraphrased_predicates():
    """The canonicaliser cannot know these are the same; wording can suggest it."""
    a = make_fact(
        fact_id="f_a",
        predicate="revenue_from_services",
        claim="Revenue from services for the year was 8,142 Cr.",
    )
    b = make_fact(
        fact_id="f_b",
        predicate="revenue_from_operations",
        doc="d2",
        claim="Revenue from operations for the year was 81,415.38 million.",
    )
    c = make_fact(
        fact_id="f_c",
        predicate="board_size",
        claim="The board comprised nine directors on the reporting date.",
    )
    pairs = ids(semantic_pairs([a, b, c], BagOfWords(), k=1))
    assert ("f_a", "f_b") in pairs
    assert not comparison_key_pairs([a, b])


def test_semantic_block_is_skipped_without_an_encoder():
    assert semantic_pairs([make_fact(), make_fact()], None) == set()


def test_top_k_never_returns_a_fact_as_its_own_neighbour():
    matrix = unit_rows(np.eye(4, dtype=np.float32) + 0.1)
    for row, neighbours in enumerate(top_k(matrix, k=3)):
        assert row not in neighbours
        assert len(neighbours) == 3


# --- strategy 3: value-anchored --------------------------------------------

def test_the_same_number_at_different_scales_lands_in_one_bucket():
    """The strategy that makes cross-scale corroboration reachable at all."""
    a = make_fact(fact_id="f_a", raw="8,142 Cr", predicate="revenue_from_services")
    b = make_fact(
        fact_id="f_b", raw="81,415.38 mn", predicate="turnover_reported", doc="d2"
    )
    assert not comparison_key_pairs([a, b])
    assert ids(value_pairs([a, b])) == {("f_a", "f_b")}


def test_magnitudes_a_decade_apart_do_not_collide():
    a = make_fact(fact_id="f_a", raw="8,142 Cr")
    b = make_fact(fact_id="f_b", raw="814.2 Cr", doc="d2")
    assert value_pairs([a, b]) == set()


def test_a_percentage_never_collides_with_an_absolute_of_the_same_digits():
    a = make_fact(fact_id="f_a", raw="8.14%")
    b = make_fact(fact_id="f_b", raw="8.14", doc="d2")
    assert value_pairs([a, b]) == set()


def test_a_negative_never_collides_with_its_positive():
    a = make_fact(fact_id="f_a", raw="(452) Cr")
    b = make_fact(fact_id="f_b", raw="452 Cr", doc="d2")
    assert value_pairs([a, b]) == set()


def test_the_window_reaches_across_a_bucket_boundary():
    """A figure and its rounded restatement must not be split by a boundary."""
    facts = [
        make_fact(fact_id="f_a", raw="81,415.38 mn"),
        make_fact(fact_id="f_b", raw="8,142 Cr", doc="d2"),
    ]
    assert ids(value_pairs(facts, window=1)) == {("f_a", "f_b")}


def test_facts_without_a_parsed_number_are_not_bucketed():
    facts = [
        make_fact(fact_id="f_a", raw="Resigned", kind="text"),
        make_fact(fact_id="f_b", raw="Appointed", kind="text", doc="d2"),
    ]
    assert value_pairs(facts) == set()


# --- the union --------------------------------------------------------------

def test_block_records_which_strategy_found_each_pair():
    a = make_fact(fact_id="f_a", raw="8,142 Cr", claim="Revenue was 8,142 Cr.")
    b = make_fact(
        fact_id="f_b", raw="81,415.38 mn", doc="d2", claim="Revenue was 81,415.38 million."
    )
    candidates, _ = block([a, b], encoder=BagOfWords(), k=1)
    assert set(candidates[("f_a", "f_b")]) == {"comparison_key", "value", "semantic"}


def test_two_extractions_of_one_span_are_not_a_pair():
    a = make_fact(fact_id="f_a", raw="8,142 Cr", span=500)
    b = make_fact(fact_id="f_b", raw="8,142 Cr", span=500)
    candidates, stats = block([a, b])
    assert candidates == {}
    assert stats.duplicates_dropped == 1


def test_blocking_reports_what_it_saved():
    facts = [
        make_fact(fact_id=f"f_{i}", raw=f"{100 + i} Cr", predicate=f"metric_{i % 4}")
        for i in range(20)
    ]
    _, stats = block(facts)
    assert stats.n_facts == 20
    assert stats.theoretical_pairs == 190
    assert stats.candidates < stats.theoretical_pairs
    assert 0.0 < stats.reduction < 1.0
    assert "theoretical pairs" in stats.summary()


def test_empty_stats_do_not_divide_by_zero():
    assert BlockingStats().reduction == 0.0
    assert block([])[0] == {}


# --- the real encoder -------------------------------------------------------

@pytest.mark.slow
def test_local_encoder_runs_without_an_api_key():
    """The evaluation path must not need a credential of any kind."""
    sentence_transformers = pytest.importorskip("sentence_transformers")
    del sentence_transformers

    from concord.compare.embed import LocalEncoder

    encoder = LocalEncoder()
    vectors = encoder.encode(
        [
            "Revenue from services for the year was 8,142 Cr.",
            "Revenue from operations for the year was 81,415.38 million.",
            "The registered office is at Plot 5, Sector 44.",
        ]
    )
    assert vectors.shape[0] == 3
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)

    similarity = vectors @ vectors.T
    assert similarity[0, 1] > similarity[0, 2]


# --- incremental ingest -----------------------------------------------------
# Phase 7's exit criterion: a fourth document must not re-judge the first three
# against each other.

def corpus():
    settled = [make_fact(fact_id=f"f_old{i}", raw=f"{100 + i} Cr", doc="d1") for i in range(6)]
    arriving = [make_fact(fact_id=f"f_new{i}", raw=f"{100 + i} Cr", doc="d2") for i in range(2)]
    return settled, arriving


def test_an_incremental_pass_only_forms_pairs_touching_the_new_facts():
    settled, arriving = corpus()
    fresh = frozenset(f.fact_id for f in arriving)

    candidates, _ = block(settled + arriving, encoder=BagOfWords(), fresh=fresh)
    assert candidates
    for a, b in candidates:
        assert a in fresh or b in fresh


def test_the_settled_pairs_are_exactly_what_an_incremental_pass_skips():
    settled, arriving = corpus()
    fresh = frozenset(f.fact_id for f in arriving)

    everything, _ = block(settled + arriving, encoder=BagOfWords())
    incremental, _ = block(settled + arriving, encoder=BagOfWords(), fresh=fresh)

    skipped = set(everything) - set(incremental)
    assert skipped
    assert all(a not in fresh and b not in fresh for a, b in skipped)
    # Nothing involving a new fact is lost by going incremental.
    assert {p for p in everything if p[0] in fresh or p[1] in fresh} == set(incremental)


def test_a_new_fact_can_still_find_an_old_one_semantically():
    """The old facts stay in the search space; they just stop querying it."""
    old = make_fact(fact_id="f_old", doc="d1", claim="Revenue from services was 8,142 Cr.")
    new = make_fact(fact_id="f_new", doc="d2", claim="Revenue from services was 81,415.38 million.")
    pairs = semantic_pairs([old, new], BagOfWords(), k=1, fresh=frozenset({"f_new"}))
    assert ids(pairs) == {("f_new", "f_old")}


def test_the_reduction_figure_reflects_the_work_actually_faced():
    settled, arriving = corpus()
    fresh = frozenset(f.fact_id for f in arriving)
    _, full = block(settled + arriving, encoder=BagOfWords())
    _, part = block(settled + arriving, encoder=BagOfWords(), fresh=fresh)

    assert full.theoretical_pairs == 28  # 8 facts
    assert part.theoretical_pairs == 13  # minus the 15 pairs already settled
    assert part.n_facts == 8


def test_an_incremental_pass_with_no_new_facts_does_nothing():
    settled, _ = corpus()
    candidates, stats = block(settled, encoder=BagOfWords(), fresh=frozenset())
    assert candidates == {}
    assert stats.theoretical_pairs == 0


# --- the vector cache -------------------------------------------------------

class CountingEncoder:
    """Fixed-width encoder that records how much work it was asked to do.

    Width has to be fixed, unlike `BagOfWords`, because a cached vector and a
    fresh one have to stack into one matrix.
    """

    def __init__(self, width: int = 16):
        self.width = width
        self.encoded: list[str] = []

    def encode(self, texts):
        self.encoded.extend(texts)
        matrix = np.zeros((len(texts), self.width), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                matrix[row, hash(word) % self.width] += 1.0
        return unit_rows(matrix)


def test_a_cached_vector_is_not_computed_twice():
    """Encoding is the whole cost of a comparison run and it grows with the
    ledger, not with the document being added."""
    from concord.compare.block import encode

    facts = [make_fact(fact_id=f"f_{i}", raw=f"{100 + i} Cr") for i in range(4)]
    encoder = CountingEncoder()
    vectors: dict = {}

    encode(facts, encoder, vectors)
    assert len(encoder.encoded) == 4
    assert set(vectors) == {f.fact_id for f in facts}

    encode(facts, encoder, vectors)
    assert len(encoder.encoded) == 4  # nothing re-encoded


def test_only_the_facts_without_a_vector_are_encoded():
    from concord.compare.block import encode

    facts = [make_fact(fact_id=f"f_{i}", raw=f"{100 + i} Cr") for i in range(4)]
    encoder = CountingEncoder()
    vectors: dict = {}
    encode(facts[:3], encoder, vectors)

    encoder.encoded.clear()
    matrix = encode(facts, encoder, vectors)
    assert encoder.encoded == [facts[3].embed_text]
    assert matrix.shape == (4, encoder.width)


def test_the_cache_does_not_change_which_pairs_are_proposed():
    facts = [
        make_fact(fact_id="f_a", claim="revenue from services was 8,142 Cr"),
        make_fact(fact_id="f_b", claim="revenue from services was 81,415.38 mn"),
        make_fact(fact_id="f_c", claim="the registered office is in Gurugram"),
    ]
    warm: dict = {}
    cold = semantic_pairs(facts, CountingEncoder(), k=1)
    semantic_pairs(facts, CountingEncoder(), k=1, vectors=warm)
    assert ids(semantic_pairs(facts, CountingEncoder(), k=1, vectors=warm)) == ids(cold)


# --- the exact block reads the key the way the decision table does ---------
#
# A pair is only ever stored if its comparison keys match, so once this
# strategy resolves predicates through the registry's aliases and subjects
# through the identities `same_entity` accepts, it proposes every pair that
# could become a relation - and it has no rank budget to be crowded out of.

def test_the_exact_block_follows_a_confirmed_alias():
    """The registry decided these two names are one relation. Before, the one
    strategy that costs nothing could not see that decision."""
    a = make_fact(fact_id="f_a", predicate="revenue_from_operations")
    b = make_fact(fact_id="f_b", predicate="total_revenue_from_operations", doc="d2")
    assert comparison_key_pairs([a, b]) == set()
    aliases = {"total_revenue_from_operations": "revenue_from_operations"}
    assert comparison_key_pairs([a, b], aliases) == {pair_key(a, b)}


def test_the_exact_block_bridges_a_trailing_legal_form():
    a = make_fact(fact_id="f_a", subject="Acme Logistics")
    b = make_fact(fact_id="f_b", subject="Acme Logistics Limited", doc="d2")
    assert comparison_key_pairs([a, b]) == {pair_key(a, b)}


def test_a_hard_key_on_one_side_only_still_meets_the_surface():
    """`comparison_key` is `subject_key or surface`, so a fact that read a CIN
    out of its sentence and one that did not sat in different buckets - for the
    same company, in the same document."""
    a = make_fact(fact_id="f_a", subject="Acme Logistics Limited", subject_key="CIN-1")
    b = make_fact(fact_id="f_b", subject="Acme Logistics Limited", doc="d2")
    assert comparison_key_pairs([a, b]) == {pair_key(a, b)}


def test_grouping_proposes_what_the_table_then_refuses():
    """Two hard keys are two entities however alike the names. Blocking may
    still put them in one bucket; the decision table is where that is settled."""
    a = make_fact(fact_id="f_a", subject="Acme Logistics Limited", subject_key="CIN-1")
    b = make_fact(fact_id="f_b", subject="Acme Logistics Limited", subject_key="CIN-2", doc="d2")
    assert comparison_key_pairs([a, b]) == {pair_key(a, b)}
    assert decide(a, b).verdict == "unrelated"


def test_a_fact_is_never_paired_with_itself_through_two_identities():
    """One fact is filed under several identities, so it meets itself in a
    bucket unless the pairing says otherwise."""
    fact = make_fact(fact_id="f_a", subject="Acme Logistics Limited", subject_key="CIN-1")
    assert comparison_key_pairs([fact]) == set()


def test_the_exact_block_proposes_every_pair_that_could_be_stored():
    """The guarantee that closes the corpus-size dependence.

    `unrelated` is never stored, and a pair is `unrelated` unless its
    comparison keys match - so the set of storable pairs is exactly the set
    this strategy enumerates. Anything the semantic block alone had to find
    was a pair whose survival depended on a top-k budget that every new
    document contests, which is how relations went missing.
    """
    facts = [
        make_fact(fact_id="f_a", subject="Acme Logistics", predicate="revenue_from_operations"),
        make_fact(fact_id="f_b", subject="Acme Logistics Limited",
                  predicate="total_revenue_from_operations", doc="d2"),
        make_fact(fact_id="f_c", subject="Acme Logistics Limited", subject_key="CIN-1",
                  predicate="revenue_from_operations", doc="d3"),
        make_fact(fact_id="f_d", subject="Borealis Freight", predicate="revenue_from_operations"),
        make_fact(fact_id="f_e", subject="Acme Logistics", predicate="employee_count"),
    ]
    aliases = {"total_revenue_from_operations": "revenue_from_operations"}
    proposed = comparison_key_pairs(facts, aliases)

    storable = {
        pair_key(x, y)
        for x, y in itertools.combinations(facts, 2)
        if decide(x, y, aliases=aliases).verdict != "unrelated"
    }
    assert storable, "the fixture must contain at least one storable pair"
    assert storable <= proposed
