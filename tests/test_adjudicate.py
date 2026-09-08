"""Phase 5: what the judge is allowed to decide, and what it is not.

Every test here runs against a fake client. The point of the phase is the
guards, and a guard is only worth anything if it fires on a model that is
wrong - which is far easier to arrange with a fake than with a real one.
"""

import pytest
from factories import make_fact

from concord.compare.adjudicate import adjudicate, validate
from concord.compare.engine import Relation, compare
from concord.compare.judge import SYSTEM, VerdictOut, build, render_qualifiers
from concord.llm.cache import MissingFromCache
from concord.llm.client import LLMError


def pair(**overrides):
    """A conflict the deterministic layer would refer: same context, disjoint."""
    common = dict(period="FY 2023-24", consolidation="consolidated")
    a = make_fact(fact_id="f_a", raw="8,142 Cr", doc="deck", **common)
    b = make_fact(fact_id="f_b", raw="7,224 Cr", doc="report", **(common | overrides))
    return a, b


def relation(verdict="contradicts", rule="same_context_disjoint_intervals"):
    return Relation(
        fact_a="f_a",
        fact_b="f_b",
        verdict=verdict,
        rule_fired=rule,
        explanation="The values disagree: the intervals are disjoint.",
        needs_llm=True,
    )


class FakeClient:
    """Answers in sequence, so both presentation orders can be steered."""

    def __init__(self, *answers, raises=None):
        self.answers = list(answers)
        self.raises = raises
        self.prompts = []

    def complete(self, prompt, schema, system=None, temperature=0.0):
        if self.raises:
            raise self.raises
        self.prompts.append(prompt)
        return self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)]


def verdict(kind="contradicts", key=None, text="Because they cannot both hold."):
    return VerdictOut(verdict=kind, qualifier_key=key, explanation=text, confidence=0.8)


# --- the model may decide, when it agrees with itself -----------------------

def test_a_consistent_contradiction_is_upheld():
    a, b = pair()
    rel = relation()
    client = FakeClient(verdict("contradicts"), verdict("contradicts"))

    run = adjudicate([rel], [a, b], client)

    assert rel.verdict == "contradicts"
    assert rel.decided_by == "llm"
    assert rel.self_consistent is True
    assert rel.rule_fired == "llm_adjudicated"
    assert not rel.needs_llm
    assert run.calls == 2 and run.upheld == 1
    assert run.self_consistency_rate == 1.0


def test_the_model_may_reconcile_a_pair_the_table_referred():
    """The asymmetric prompt exists to make this outcome reachable."""
    a, b = pair()
    rel = relation()
    answer = verdict("reconciled_by_context", key="consolidation", text="Different bases.")
    run = adjudicate([rel], [a, b], FakeClient(answer, answer))

    assert rel.verdict == "reconciled_by_context"
    assert rel.qualifier_key == "consolidation"
    assert rel.explanation == "Different bases."
    assert run.downgraded == 0


# --- guard 1: the named qualifier must exist -------------------------------

def test_a_verdict_naming_a_qualifier_neither_fact_records_is_rejected():
    a, b = pair()
    rel = relation()
    invented = verdict("reconciled_by_context", key="restatement_basis")

    run = adjudicate([rel], [a, b], FakeClient(invented, invented))

    assert rel.verdict == "insufficient_context"
    assert rel.rule_fired == "hallucinated_qualifier_rejected"
    assert rel.decided_by == "deterministic"
    assert rel.qualifier_key is None
    assert run.rejected_keys == 2
    assert run.rejection_rate == 1.0
    assert "restatement_basis" in run.rejections[0]


def test_reconciliation_must_say_what_reconciles():
    a, b = pair()
    rel = relation()
    unnamed = verdict("reconciled_by_context", key=None)

    run = adjudicate([rel], [a, b], FakeClient(unnamed, unnamed))

    assert rel.verdict == "insufficient_context"
    assert run.rejected_keys == 2


def test_a_contradiction_naming_no_qualifier_is_admissible():
    """Null is the correct answer for `contradicts`, not a missing field."""
    assert validate(verdict("contradicts", key=None), *pair()) is None


def test_validation_accepts_a_key_present_on_either_side():
    a, b = pair(segment="express")
    assert validate(verdict("reconciled_by_context", key="segment"), a, b) is None


# --- guard 2: self-consistency ---------------------------------------------

def test_an_order_dependent_answer_establishes_nothing():
    a, b = pair()
    rel = relation()
    client = FakeClient(
        verdict("contradicts"),
        verdict("reconciled_by_context", key="consolidation"),
    )

    run = adjudicate([rel], [a, b], client)

    assert rel.verdict == "insufficient_context"
    assert rel.rule_fired == "judge_inconsistent"
    assert rel.self_consistent is False
    assert run.inconsistent == 1
    assert run.self_consistency_rate == 0.0
    assert "contradicts against reconciled_by_context" in rel.explanation


def test_both_presentation_orders_are_actually_sent():
    a, b = pair()
    client = FakeClient(verdict(), verdict())
    adjudicate([relation()], [a, b], client)

    first, second = client.prompts
    assert first != second
    assert a.claim_text in first and b.claim_text in first
    assert first.index(a.claim_text) < first.index(b.claim_text)
    assert second.index(b.claim_text) < second.index(a.claim_text)


# --- guard 3: a failed call is not a verdict --------------------------------

def test_offline_with_no_cached_answer_leaves_the_pair_untouched():
    a, b = pair()
    rel = relation()
    before = rel.explanation

    run = adjudicate([rel], [a, b], FakeClient(raises=MissingFromCache("no entry")))

    assert rel.verdict == "contradicts"
    assert rel.explanation == before
    assert rel.decided_by == "deterministic"
    assert run.unadjudicated == 1 and run.calls == 0


def test_an_api_error_is_not_a_verdict_either():
    a, b = pair()
    rel = relation()
    run = adjudicate([rel], [a, b], FakeClient(raises=LLMError("429")))
    assert rel.verdict == "contradicts"
    assert run.unadjudicated == 1


# --- reconciled candidates: prose only --------------------------------------

def test_a_reconciled_candidate_costs_one_call_and_keeps_its_verdict():
    a, b = pair(period="FY 2022-23")
    rel = relation(verdict="reconciled_by_context", rule="discriminating_qualifier_differs")
    rel.qualifier_key = "period"
    client = FakeClient(verdict("contradicts", key="period", text="They cover different years."))

    run = adjudicate([rel], [a, b], client)

    assert rel.verdict == "reconciled_by_context"  # not the model's to change
    assert rel.rule_fired == "discriminating_qualifier_differs"
    assert rel.decided_by == "deterministic"
    assert rel.explanation == "They cover different years."
    assert run.calls == 1 and run.prose_only == 1


def test_prose_that_invents_a_qualifier_falls_back_to_the_deterministic_sentence():
    a, b = pair(period="FY 2022-23")
    rel = relation(verdict="reconciled_by_context", rule="discriminating_qualifier_differs")
    before = rel.explanation

    run = adjudicate([rel], [a, b], FakeClient(verdict("reconciled_by_context", key="vintage")))

    assert rel.explanation == before
    assert run.rejected_keys == 1


# --- the prompt itself ------------------------------------------------------

def test_the_prompt_lists_only_keys_the_model_is_allowed_to_name():
    a, b = pair(segment="express")
    rendered = render_qualifiers(b)
    assert "segment: express" in rendered
    assert "consolidation: consolidated" in rendered
    assert "[stated]" in rendered


def test_a_fact_with_no_qualifiers_still_renders():
    assert "(none recorded)" in render_qualifiers(make_fact())


def test_the_prompt_names_no_subject_area():
    """Domain vocabulary in what *we* write is a generalisation failure.

    The rendered facts carry whatever words the document used; the template
    around them must not add any, or the judge is tuned to one document family.
    """
    from concord.compare.judge import PAIR

    template = (SYSTEM + PAIR).lower()
    for word in ("revenue", "financial", "company", "fiscal", "annual report", "currency"):
        assert word not in template


def test_the_prompt_forbids_arithmetic_and_carries_the_finding():
    a, b = pair()
    prompt = build(a, b, "8,142 Cr and 7,224 Cr are disjoint")
    assert "8,142 Cr and 7,224 Cr are disjoint" in prompt
    assert "recalculate" in SYSTEM and "Never invent a qualifier key" in SYSTEM


# --- the seam with Phase 4 --------------------------------------------------

def test_only_the_residue_is_ever_offered_to_the_judge():
    facts = [
        make_fact(fact_id="f_1", doc="deck", raw="8,142 Cr", period="FY 2023-24"),
        make_fact(fact_id="f_2", doc="report", raw="81,415.38 mn", period="FY 2023-24"),
        make_fact(fact_id="f_3", doc="filing", raw="7,224 Cr", period="FY 2023-24"),
    ]
    run = compare(facts)
    client = FakeClient(verdict(), verdict())
    adjudicate(run.llm_queue, facts, client)

    corroboration = next(r for r in run.relations if r.verdict == "corroborates")
    assert corroboration.decided_by == "deterministic"
    assert client.prompts  # the conflict was referred
    assert len(client.prompts) == 2 * len(
        [r for r in run.relations if r.rule_fired == "llm_adjudicated"]
    )


def test_a_limit_caps_what_a_run_can_spend():
    facts = [make_fact(fact_id=f"f_{i}", raw=f"{800 + i} Cr", doc=f"d{i}") for i in range(4)]
    run = compare(facts)
    assert len(run.llm_queue) > 1

    client = FakeClient(verdict(), verdict())
    spent = adjudicate(run.llm_queue, facts, client, limit=1)
    assert spent.pairs == 1 and spent.calls == 2


def test_the_run_reports_the_rates_the_readme_needs():
    a, b = pair()
    run = adjudicate([relation()], [a, b], FakeClient(verdict(), verdict()))
    summary = run.summary()
    assert "self-consistency" in summary
    assert "hallucinated-qualifier rejection" in summary


@pytest.mark.parametrize("rate", ["self_consistency_rate", "rejection_rate"])
def test_rates_are_zero_rather_than_undefined_on_an_empty_run(rate):
    from concord.compare.adjudicate import AdjudicationRun

    assert getattr(AdjudicationRun(), rate) == 0.0


def test_the_ledger_is_counted_after_adjudication_not_before():
    """The deterministic tally is a record of that pass, not the live state."""
    facts = [
        make_fact(fact_id="f_1", doc="deck", raw="8,142 Cr", period="FY 2023-24"),
        make_fact(fact_id="f_2", doc="report", raw="7,224 Cr", period="FY 2023-24"),
    ]
    run = compare(facts)
    assert run.verdicts["contradicts"] == 1

    reconciled = verdict("reconciled_by_context", key="period", text="Different years.")
    adjudicate(run.llm_queue, facts, FakeClient(reconciled, reconciled))

    assert run.verdicts["contradicts"] == 1  # unchanged: it records the pass
    assert run.final_verdicts()["contradicts"] == 0
    assert run.final_verdicts()["reconciled_by_context"] == 1


def test_a_rejected_verdict_is_not_counted_as_an_inconsistency():
    """It never reached the mirror comparison, so it cannot have failed it."""
    a, b = pair()
    invented = verdict("reconciled_by_context", key="restatement_basis")
    run = adjudicate([relation()], [a, b], FakeClient(invented, invented))

    assert run.consistency_checked == 0
    assert run.self_consistency_rate == 0.0
    assert run.rejected_keys == 2


def test_consistency_is_measured_over_pairs_that_reached_the_check():
    a, b = pair()
    consistent = adjudicate([relation()], [a, b], FakeClient(verdict(), verdict()))
    assert consistent.consistency_checked == 1
    assert consistent.self_consistency_rate == 1.0

    a2, b2 = pair()
    split = adjudicate(
        [relation()],
        [a2, b2],
        FakeClient(verdict("contradicts"), verdict("reconciled_by_context", key="consolidation")),
    )
    assert split.consistency_checked == 1
    assert split.self_consistency_rate == 0.0
