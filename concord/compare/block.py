"""Candidate selection: which pairs are worth judging at all.

Zero LLM calls happen in this module. Its whole job is to turn an O(n^2) space
into a few thousand pairs cheaply, and to be honest about how many it dropped.
Recall matters more than precision here, because the decision table downstream
throws out everything that does not hold up - a pair that never enters the
candidate set, by contrast, can never be judged at all.

Three strategies, unioned, each catching what the others miss:

  comparison key  exact, free, catches the majority of true pairs
  semantic        catches predicate paraphrases the canonicaliser missed
  value-anchored  catches the same number written at different scales

The third is the one that makes cross-scale corroboration land. `Rs 8,142 Cr`
and `Rs 81,415.38 million` share almost no characters and are only mildly
similar as sentences, but after normalisation they are the same number, so
they collide in a log-magnitude bucket regardless of wording.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field

from concord.facts import Fact

# Bucket width in decades. 0.002 decades is a 0.46% band, so a figure and its
# rounded restatement land in the same bucket or the neighbouring one; the
# window of 1 then covers about 0.9% of relative difference. Wider than that
# and large buckets start pairing genuinely different figures.
BUCKET_WIDTH = 0.002
BUCKET_WINDOW = 1
TOP_K = 10

PairKey = tuple[str, str]


def pair_key(a: Fact, b: Fact) -> PairKey:
    """Order a pair by fact id so a pair is the same pair either way round."""
    return (a.fact_id, b.fact_id) if a.fact_id <= b.fact_id else (b.fact_id, a.fact_id)


@dataclass
class BlockingStats:
    n_facts: int = 0
    theoretical_pairs: int = 0
    by_strategy: dict[str, int] = field(default_factory=dict)
    candidates: int = 0
    duplicates_dropped: int = 0

    @property
    def reduction(self) -> float:
        if not self.theoretical_pairs:
            return 0.0
        return 1.0 - self.candidates / self.theoretical_pairs

    def summary(self) -> str:
        strategies = ", ".join(f"{name}={count}" for name, count in sorted(self.by_strategy.items()))
        return (
            f"{self.n_facts} facts, {self.theoretical_pairs} theoretical pairs, "
            f"{self.candidates} candidates ({self.reduction:.4%} reduction) "
            f"[{strategies}], {self.duplicates_dropped} duplicate spans dropped"
        )


def _same_span(a: Fact, b: Fact) -> bool:
    """The same claim extracted twice from one span is not a pair to judge."""
    return a.evidence.span == b.evidence.span


def comparison_key_pairs(facts: list[Fact]) -> set[PairKey]:
    """Group on the exact comparison key and pair within each group."""
    groups: dict[str, list[Fact]] = defaultdict(list)
    for fact in facts:
        groups[fact.comparison_key].append(fact)

    pairs: set[PairKey] = set()
    for group in groups.values():
        for a, b in itertools.combinations(group, 2):
            pairs.add(pair_key(a, b))
    return pairs


def _bucket(fact: Fact, width: float) -> tuple[str, int] | None:
    """Place a figure on a log-magnitude axis, in base units.

    Percentages get their own axis: 7.81% and 7.81 million are the same digits
    and must never collide. Sign is part of the axis for the same reason.
    Zero has no logarithm and gets a bucket of its own.
    """
    quantity = fact.quantity
    if quantity is None:
        return None

    value = quantity.normalized
    if not math.isfinite(value):
        return None

    axis = "pct" if quantity.is_percent else "abs"
    if value == 0.0:
        return (f"{axis}:zero", 0)

    sign = "-" if value < 0 else "+"
    return (f"{axis}:{sign}", int(math.floor(math.log10(abs(value)) / width)))


def value_pairs(
    facts: list[Fact], width: float = BUCKET_WIDTH, window: int = BUCKET_WINDOW
) -> set[PairKey]:
    """Pair facts whose normalised magnitudes are within a bucket window."""
    buckets: dict[tuple[str, int], list[Fact]] = defaultdict(list)
    for fact in facts:
        slot = _bucket(fact, width)
        if slot is not None:
            buckets[slot].append(fact)

    pairs: set[PairKey] = set()
    for (axis, index), members in buckets.items():
        for a, b in itertools.combinations(members, 2):
            pairs.add(pair_key(a, b))
        for offset in range(1, window + 1):
            for other in buckets.get((axis, index + offset), ()):
                for member in members:
                    pairs.add(pair_key(member, other))
    return pairs


def semantic_pairs(facts: list[Fact], encoder, k: int = TOP_K) -> set[PairKey]:
    """Top-k cosine neighbours over claim text, both directions unioned."""
    from concord.compare.embed import top_k

    if len(facts) < 2 or encoder is None:
        return set()

    matrix = encoder.encode([fact.embed_text for fact in facts])
    pairs: set[PairKey] = set()
    for row, neighbours in enumerate(top_k(matrix, k)):
        for column in neighbours:
            pairs.add(pair_key(facts[row], facts[column]))
    return pairs


def block(
    facts: list[Fact],
    encoder=None,
    k: int = TOP_K,
    width: float = BUCKET_WIDTH,
    window: int = BUCKET_WINDOW,
) -> tuple[dict[PairKey, list[str]], BlockingStats]:
    """Run all three strategies and union them, keeping which one fired.

    `blocked_by` is stored on the relation so the README can report what each
    strategy actually bought, rather than asserting that all three were needed.
    """
    by_id = {fact.fact_id: fact for fact in facts}
    stats = BlockingStats(
        n_facts=len(facts), theoretical_pairs=len(facts) * (len(facts) - 1) // 2
    )

    produced = {
        "comparison_key": comparison_key_pairs(facts),
        "value": value_pairs(facts, width, window),
        "semantic": semantic_pairs(facts, encoder, k) if encoder is not None else set(),
    }

    candidates: dict[PairKey, list[str]] = defaultdict(list)
    for name, pairs in produced.items():
        stats.by_strategy[name] = len(pairs)
        for left, right in pairs:
            if _same_span(by_id[left], by_id[right]):
                continue
            candidates[(left, right)].append(name)

    stats.duplicates_dropped = len(set().union(*produced.values())) - len(candidates)
    stats.candidates = len(candidates)
    return dict(candidates), stats
