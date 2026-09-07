"""Blocking: does the candidate set contain the pairs that matter, and how
much of the O(n^2) space did it cost to get them."""

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
