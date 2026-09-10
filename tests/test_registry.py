"""The emergent schema, and the discipline that keeps it honest.

The registry is the only component allowed to decide that two predicates mean
the same thing, and it may only reach that conclusion two ways: an exact match,
or a similarity above threshold that an LLM then confirms. Everything else
stays separate. These tests pin that asymmetry, because merging two unlike
predicates is the expensive error - it makes unrelated facts look like
contradictions - while leaving two alike ones apart only costs recall.
"""

import zlib

import numpy as np
import pytest
from factories import make_fact

from concord.registry import AliasAnswer, Entry, PredicateRegistry, contrastive


DIM = 64


class WordOverlap:
    """A deterministic stand-in for the sentence encoder.

    Words are hashed into a fixed number of dimensions, so vectors from
    separate calls stay comparable - which the real encoder guarantees and a
    per-call vocabulary would not. crc32 rather than `hash`, because Python
    salts string hashing per process and the double must not drift between
    runs.
    """

    def encode(self, texts):
        m = np.zeros((len(texts), DIM), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in t.lower().split():
                m[r, zlib.crc32(w.encode()) % DIM] += 1.0
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)


class Judge:
    """Answers every alias question the same way, and counts the asking."""

    def __init__(self, same=True):
        self.same = same
        self.prompts = []

    def complete(self, prompt, schema, system=None, temperature=0.0):
        self.prompts.append(prompt)
        return AliasAnswer(same_relation=self.same, reason="because")


def registry(**kwargs):
    kwargs.setdefault("encoder", WordOverlap())
    return PredicateRegistry(**kwargs)


# --- registering ------------------------------------------------------------

def test_a_first_predicate_is_registered_without_asking_anything():
    reg = registry(client=Judge())
    event = reg.observe("revenue_from_services", doc_id="d1")

    assert event.event_type == "predicate_registered"
    assert event.decided_by == "deterministic"
    assert reg.calls == 0
    assert len(reg) == 1


def test_seeing_a_known_predicate_again_only_counts_it():
    reg = registry(client=Judge())
    reg.observe("revenue_from_services", "d1")
    event = reg.observe("Revenue From Services", "d2")

    assert event.event_type == "predicate_seen"
    assert reg.calls == 0
    assert len(reg) == 1
    assert reg.entries["revenue_from_services"].count == 2


def test_an_unrelated_predicate_is_new_and_costs_no_call():
    reg = registry(client=Judge())
    reg.observe("revenue_from_services", "d1")
    event = reg.observe("board_meeting_attendance", "d2")

    assert event.event_type == "predicate_registered"
    assert reg.calls == 0
    assert len(reg) == 2


# --- aliasing ---------------------------------------------------------------

def test_a_similar_predicate_is_confirmed_by_one_question():
    reg = registry(client=Judge(same=True), threshold=0.5)
    reg.observe("revenue_from_services", "d1")
    event = reg.observe("revenue_from_operations", "d2")

    assert event.event_type == "alias_confirmed"
    assert event.target == "revenue_from_services"
    assert event.decided_by == "llm"
    assert reg.calls == 1
    assert len(reg) == 1
    assert reg.aliases() == {"revenue_from_operations": "revenue_from_services"}


def test_the_model_may_refuse_and_the_predicate_stays_separate():
    reg = registry(client=Judge(same=False), threshold=0.5)
    reg.observe("revenue_from_services", "d1")
    event = reg.observe("revenue_from_operations", "d2")

    assert event.event_type == "alias_rejected"
    assert len(reg) == 2
    assert reg.aliases() == {}


# --- measures in contrast ---------------------------------------------------

@pytest.mark.parametrize(
    "left, right",
    [
        ("gross_inflows", "total_inflows"),
        ("net_carrying_value", "gross_carrying_value"),
        ("basic_earnings_per_share", "diluted_earnings_per_share"),
        ("standalone_revenue", "consolidated_revenue"),
    ],
)
def test_two_measures_of_one_thing_are_never_the_same_relation(left, right):
    assert contrastive(left, right)


@pytest.mark.parametrize(
    "left, right",
    [
        # Only one side carries a modifier: `total` restates rather than
        # contrasts, and this shape is most of the corpus's real aliases.
        ("total_revenue_from_operations", "revenue_from_operations"),
        # Same modifier on both sides, different wording after it.
        ("net_cash_used_in_financing", "net_cash_generated_from_financing"),
        # One word stripped, not a greedy run: the heads stay different.
        ("consolidated_net_assets", "total_consolidated_net_assets"),
        # No modifier anywhere.
        ("expected_growth", "projected_growth_rate"),
    ],
)
def test_names_that_are_not_in_contrast_stay_mergeable(left, right):
    assert contrastive(left, right) is None


def test_a_measure_in_contrast_is_refused_without_asking_the_model():
    """The model answers this class inconsistently, so code answers it."""
    judge = Judge(same=True)
    reg = registry(client=judge, threshold=0.5)
    reg.observe("gross_inflows", "d1")
    event = reg.observe("total_inflows", "d2")

    assert event.event_type == "alias_rejected"
    assert event.decided_by == "deterministic"
    assert judge.prompts == []
    assert reg.calls == 0
    assert len(reg) == 2
    assert reg.aliases() == {}


def test_nothing_is_merged_without_a_client_to_confirm_it():
    """An unconfirmed alias must not merge two predicates: merging unlike
    things makes them look like contradictions, which is the worse error."""
    reg = registry(client=None, threshold=0.5)
    reg.observe("revenue_from_services", "d1")
    event = reg.observe("revenue_from_operations", "d2")

    assert event.event_type == "alias_rejected"
    assert len(reg) == 2


def test_a_model_that_errors_does_not_merge_anything():
    class Broken:
        def complete(self, *a, **k):
            raise RuntimeError("upstream is down")

    reg = registry(client=Broken(), threshold=0.5)
    reg.observe("revenue_from_services", "d1")
    assert reg.observe("revenue_from_operations", "d2").event_type == "alias_rejected"


def test_only_pairs_above_the_threshold_are_ever_asked_about():
    reg = registry(client=Judge(same=True), threshold=0.99)
    reg.observe("revenue_from_services", "d1")
    event = reg.observe("board_meeting_attendance", "d2")

    assert reg.calls == 0
    assert event.event_type == "predicate_registered"


# --- the timeline -----------------------------------------------------------

def test_every_decision_is_recorded_including_the_uneventful_ones():
    reg = registry(client=Judge(same=True), threshold=0.5)
    reg.observe("revenue_from_services", "d1")
    reg.observe("revenue_from_services", "d2")
    reg.observe("revenue_from_operations", "d2")

    assert [e.event_type for e in reg.events] == [
        "predicate_registered",
        "predicate_seen",
        "alias_confirmed",
    ]
    assert [e.doc_id for e in reg.events] == ["d1", "d2", "d2"]
    assert reg.events[-1].similarity > 0.5


def test_the_registry_is_independent_of_extraction_order():
    """The first spelling becomes canonical, so a run must be reproducible."""
    facts = [
        make_fact(fact_id="f_1", predicate="total_assets"),
        make_fact(fact_id="f_2", predicate="board_size"),
        make_fact(fact_id="f_3", predicate="revenue_from_services"),
    ]
    first = registry(client=Judge(same=False))
    second = registry(client=Judge(same=False))
    first.observe_all(facts, "d1")
    second.observe_all(list(reversed(facts)), "d1")

    assert sorted(first.entries) == sorted(second.entries)
    assert [e.subject for e in first.events] == [e.subject for e in second.events]


# --- the seam with comparison ----------------------------------------------

def test_the_alias_map_is_what_the_comparison_engine_consumes():
    from concord.compare.engine import compare

    reg = registry(client=Judge(same=True), threshold=0.5)
    facts = [
        make_fact(fact_id="f_a", predicate="revenue_from_services", raw="8,142 Cr",
                  doc="d1", period="FY 2023-24"),
        make_fact(fact_id="f_b", predicate="revenue_from_operations", raw="81,415.38 mn",
                  doc="d2", period="FY 2023-24"),
    ]
    reg.observe_all(facts[:1], "d1")
    reg.observe_all(facts[1:], "d2")

    assert compare(facts).verdicts["unrelated"] == 1
    assert compare(facts, aliases=reg.aliases()).relations[0].verdict == "corroborates"


def test_a_registry_loaded_from_stored_entries_keeps_working():
    stored = [Entry(canonical="revenue_from_services", aliases=["turnover"], count=4)]
    reg = PredicateRegistry(stored, encoder=WordOverlap(), client=Judge(same=False))

    assert reg.lookup("turnover").canonical == "revenue_from_services"
    assert reg.observe("turnover", "d3").event_type == "predicate_seen"
    assert reg.entries["revenue_from_services"].count == 5


@pytest.mark.parametrize("blank", ["", "   ", "!!!"])
def test_an_empty_predicate_is_recorded_but_never_registered(blank):
    reg = registry(client=Judge())
    assert reg.observe(blank, "d1").event_type == "predicate_seen"
    assert len(reg) == 0


def test_entries_embedded_by_another_model_are_skipped_not_fatal():
    """A stored registry outlives the encoder that built it."""
    stale = Entry(canonical="revenue_from_services", embedding=np.ones(7, dtype=np.float32))
    reg = PredicateRegistry([stale], encoder=WordOverlap(), client=Judge(same=True))

    event = reg.observe("revenue_from_operations", "d2")
    assert event.event_type == "predicate_registered"
    assert reg.calls == 0
