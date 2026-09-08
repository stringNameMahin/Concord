"""Adjudicating the residue, and refusing to take the model's word for it.

Only two kinds of pair reach this module, and they are treated differently:

`reconciled_by_context` candidates - the deterministic layer already found the
qualifier that differs, so the verdict is settled and the model is asked for
one thing: prose a human can read. One call, and the verdict is not the
model's to change.

`contradicts` candidates - a genuine value conflict under identical stated
context. Here the model may decide, and three guards stand between its answer
and the ledger:

1. **Named-qualifier guard.** A verdict must name the condition that drove it,
   and that key must exist on one of the two facts. A key that appears in
   neither record is a fabrication, and the verdict is rejected rather than
   argued with. A `reconciled_by_context` answer that names nothing is rejected
   for the same reason: it claims a reconciliation without saying what
   reconciles.
2. **Self-consistency.** The same pair is judged as (A,B) and as (B,A). A model
   whose answer depends on presentation order has not established anything, so
   disagreement downgrades to `insufficient_context`.
3. **Failure is not a verdict.** If the call cannot be made - offline with no
   cached response, or an API error - the relation keeps its deterministic
   verdict and explanation and is counted as unadjudicated. Nothing is guessed.

Every downgrade is counted. The rejection and self-consistency rates are
reportable numbers, not claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from concord.compare import judge
from concord.compare.engine import Relation
from concord.compare.judge import VerdictOut
from concord.facts import Fact
from concord.llm.cache import MissingFromCache
from concord.llm.client import LLMError


@dataclass
class AdjudicationRun:
    pairs: int = 0
    calls: int = 0
    prose_only: int = 0
    adjudicated: int = 0
    consistency_checked: int = 0
    consistent: int = 0
    inconsistent: int = 0
    rejected_keys: int = 0
    verdicts_checked: int = 0
    unadjudicated: int = 0
    upheld: int = 0
    downgraded: int = 0
    rejections: list[str] = field(default_factory=list)

    @property
    def self_consistency_rate(self) -> float:
        """Over pairs that actually reached the comparison.

        A verdict thrown out for naming a qualifier that does not exist never
        got as far as being checked against its mirror image, so counting it as
        an inconsistency would blame the wrong guard and understate this rate.
        """
        return self.consistent / self.consistency_checked if self.consistency_checked else 0.0

    @property
    def rejection_rate(self) -> float:
        """Share of model verdicts naming a qualifier that does not exist."""
        return self.rejected_keys / self.verdicts_checked if self.verdicts_checked else 0.0

    def summary(self) -> str:
        return (
            f"{self.pairs} pairs, {self.calls} calls; "
            f"{self.prose_only} prose-only, {self.adjudicated} adjudicated "
            f"({self.upheld} upheld, {self.downgraded} downgraded); "
            f"self-consistency {self.self_consistency_rate:.1%} "
            f"of {self.consistency_checked} checked, "
            f"hallucinated-qualifier rejection {self.rejection_rate:.1%}; "
            f"{self.unadjudicated} unadjudicated"
        )


def validate(verdict: VerdictOut, a: Fact, b: Fact) -> str | None:
    """Return the reason this verdict is inadmissible, or None if it stands."""
    available = a.qualifier_keys() | b.qualifier_keys()

    if verdict.qualifier_key and verdict.qualifier_key not in available:
        return (
            f"names qualifier {verdict.qualifier_key!r}, which appears on neither fact "
            f"(available: {', '.join(sorted(available)) or 'none'})"
        )
    if verdict.verdict == "reconciled_by_context" and not verdict.qualifier_key:
        return "claims the pair is reconciled by context without naming the condition"
    return None


def _ask(client, a: Fact, b: Fact, finding: str) -> VerdictOut:
    return client.complete(judge.build(a, b, finding), VerdictOut, system=judge.SYSTEM)


def adjudicate(
    relations: list[Relation],
    facts: list[Fact] | dict[str, Fact],
    client,
    limit: int | None = None,
) -> AdjudicationRun:
    """Finish the pairs a deterministic rule could not, updating them in place.

    `relations` is normally `ComparisonRun.llm_queue`. Relations are mutated
    rather than copied, so the comparison run they belong to reflects the
    outcome without a second join.
    """
    by_id = facts if isinstance(facts, dict) else {fact.fact_id: fact for fact in facts}
    run = AdjudicationRun()

    for relation in relations[:limit] if limit else relations:
        a, b = by_id[relation.fact_a], by_id[relation.fact_b]
        run.pairs += 1

        if relation.verdict == "reconciled_by_context":
            _write_prose(relation, a, b, client, run)
        else:
            _adjudicate_one(relation, a, b, client, run)

    return run


def _write_prose(relation: Relation, a: Fact, b: Fact, client, run: AdjudicationRun) -> None:
    """The verdict is already decided; only the sentence is the model's."""
    try:
        answer = _ask(client, a, b, relation.explanation)
        run.calls += 1
    except (MissingFromCache, LLMError):
        run.unadjudicated += 1
        return

    run.prose_only += 1
    run.verdicts_checked += 1

    reason = validate(answer, a, b)
    if reason:
        run.rejected_keys += 1
        run.rejections.append(f"{relation.relation_id}: {reason}")
        return  # keep the deterministic explanation

    relation.explanation = answer.explanation
    if answer.qualifier_key:
        relation.qualifier_key = answer.qualifier_key


def _adjudicate_one(relation: Relation, a: Fact, b: Fact, client, run: AdjudicationRun) -> None:
    finding = relation.explanation
    try:
        forward = _ask(client, a, b, finding)
        reverse = _ask(client, b, a, finding)
        run.calls += 2
    except (MissingFromCache, LLMError):
        run.unadjudicated += 1
        return

    run.adjudicated += 1
    run.verdicts_checked += 2

    reasons = [
        reason
        for reason in (validate(forward, a, b), validate(reverse, b, a))
        if reason is not None
    ]
    if reasons:
        run.rejected_keys += len(reasons)
        run.rejections.extend(f"{relation.relation_id}: {reason}" for reason in reasons)
        _downgrade(
            relation,
            run,
            "hallucinated_qualifier_rejected",
            "The pair was referred for adjudication, but the answer rested on a condition "
            "that neither document records, so it was rejected. "
            f"{finding} No admissible explanation was established.",
        )
        return

    run.consistency_checked += 1
    if forward.verdict != reverse.verdict:
        run.inconsistent += 1
        _downgrade(
            relation,
            run,
            "judge_inconsistent",
            "The pair was judged in both presentation orders and the answers differed "
            f"({forward.verdict} against {reverse.verdict}), so neither was accepted. "
            f"{finding}",
        )
        return

    run.consistent += 1
    run.upheld += 1
    relation.verdict = forward.verdict
    relation.rule_fired = "llm_adjudicated"
    relation.explanation = forward.explanation
    relation.qualifier_key = forward.qualifier_key
    relation.decided_by = "llm"
    relation.self_consistent = True
    relation.needs_llm = False


def _downgrade(relation: Relation, run: AdjudicationRun, rule: str, explanation: str) -> None:
    """A guard fired. The downgrade is ours, so it is recorded as ours."""
    run.downgraded += 1
    relation.verdict = "insufficient_context"
    relation.rule_fired = rule
    relation.explanation = explanation
    relation.qualifier_key = None
    relation.decided_by = "deterministic"
    relation.self_consistent = False
    relation.needs_llm = False
