"""Blocking, then the decision table, then a count of what is left for the LLM.

The instrumented counts this produces are the answer to the obvious question
about an O(n^2) problem: how many pairs existed, how many survived blocking,
how the verdicts came out, and how few pairs actually reach a model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from concord.compare.block import BUCKET_WIDTH, BUCKET_WINDOW, TOP_K, BlockingStats, block
from concord.compare.decide import DISCRIMINATING, Decision, decide
from concord.facts import Fact


@dataclass
class Relation:
    """One adjudicated pair. This is the graded object of the whole system."""

    fact_a: str
    fact_b: str
    verdict: str
    rule_fired: str
    explanation: str
    decided_by: str = "deterministic"
    qualifier_key: str | None = None
    blocked_by: list[str] = field(default_factory=list)
    cross_document: bool = False
    needs_llm: bool = False
    self_consistent: bool | None = None

    @property
    def relation_id(self) -> str:
        return f"r_{self.fact_a}_{self.fact_b}"


@dataclass
class ComparisonRun:
    relations: list[Relation] = field(default_factory=list)
    blocking: BlockingStats = field(default_factory=BlockingStats)
    verdicts: Counter = field(default_factory=Counter)
    dropped_unrelated: int = 0

    @property
    def llm_queue(self) -> list[Relation]:
        """The residue: pairs a deterministic rule could not finish."""
        return [relation for relation in self.relations if relation.needs_llm]

    def by_verdict(self, verdict: str) -> list[Relation]:
        return [relation for relation in self.relations if relation.verdict == verdict]

    def summary(self) -> str:
        counts = ", ".join(f"{name}={count}" for name, count in sorted(self.verdicts.items()))
        return (
            f"{self.blocking.summary()}; verdicts [{counts}]; "
            f"{len(self.llm_queue)} pairs to the LLM"
        )


def compare(
    facts: list[Fact],
    encoder=None,
    aliases: dict[str, str] | None = None,
    discriminating: frozenset[str] = DISCRIMINATING,
    keep_unrelated: bool = False,
    k: int = TOP_K,
    width: float = BUCKET_WIDTH,
    window: int = BUCKET_WINDOW,
) -> ComparisonRun:
    """Judge every candidate pair deterministically.

    `unrelated` verdicts are counted but not kept by default: blocking is tuned
    for recall, so most candidates are pairs the table then rejects, and
    storing them would bury the interesting rows in noise.
    """
    by_id = {fact.fact_id: fact for fact in facts}
    candidates, stats = block(facts, encoder=encoder, k=k, width=width, window=window)

    run = ComparisonRun(blocking=stats)
    for (left, right), strategies in candidates.items():
        a, b = by_id[left], by_id[right]
        decision = decide(a, b, aliases=aliases, discriminating=discriminating)
        run.verdicts[decision.verdict] += 1

        if decision.verdict == "unrelated" and not keep_unrelated:
            run.dropped_unrelated += 1
            continue

        run.relations.append(_relation(a, b, decision, strategies))

    run.relations.sort(key=lambda relation: (relation.verdict, relation.fact_a, relation.fact_b))
    return run


def _relation(a: Fact, b: Fact, decision: Decision, strategies: list[str]) -> Relation:
    return Relation(
        fact_a=a.fact_id,
        fact_b=b.fact_id,
        verdict=decision.verdict,
        rule_fired=decision.rule_fired,
        explanation=decision.explanation,
        qualifier_key=decision.qualifier_key,
        blocked_by=sorted(strategies),
        cross_document=a.doc_id != b.doc_id,
        needs_llm=decision.needs_llm,
    )

