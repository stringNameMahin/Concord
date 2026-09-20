"""Candidate selection: which pairs are worth judging at all.

Zero LLM calls happen in this module. Its whole job is to turn an O(n^2) space
into a few thousand pairs cheaply, and to be honest about how many it dropped.
Recall matters more than precision here, because the decision table downstream
throws out everything that does not hold up - a pair that never enters the
candidate set, by contrast, can never be judged at all.

Three strategies, unioned, each catching what the others miss:

  comparison key  exact, free, and complete over what the ledger can store
  semantic        catches predicate paraphrases the registry has not linked
  value-anchored  catches the same number written at different scales

The first is the load-bearing one, and it is worth being precise about why. A
pair is only ever stored if its comparison keys match, so once this strategy
resolves the key the way the decision table does - through the predicate
registry's aliases and through the subject identities `same_entity` accepts -
it proposes every pair that could become a relation. The other two propose
pairs the table then rejects. They stay because they cost milliseconds and
because they are the only recall a *new* vocabulary has before the registry has
learned it; they are no longer what carries the ledger.

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

from concord.facts import Fact, subject_identities

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


def comparison_key_pairs(
    facts: list[Fact], aliases: dict[str, str] | None = None
) -> set[PairKey]:
    """Group on the comparison key as the decision table reads it, and pair within.

    This used to group on `Fact.comparison_key` verbatim, which is the key as
    *written* rather than the key as *resolved* - and the decision table
    resolves it twice over before deciding anything. It asks the predicate
    registry whether two names are one relation, and it asks `subjects_match`
    whether two surfaces are one entity up to a trailing legal form. Neither
    question reached this function, so the one exact, free strategy could not
    see the equivalences the rest of the system had already established:

      - `revenue_from_operations` and `total_revenue_from_operations`, an alias
        the registry confirmed, sat in two buckets;
      - a fact carrying a CIN and a fact carrying only `Delhivery Limited` sat
        in two buckets, because the written key is `subject_key or surface`;
      - `Delhivery` and `Delhivery Limited` sat in two buckets.

    Those pairs could then only be found by the semantic block, whose top-k
    budget is contested by every fact added to the corpus - so a relation the
    ledger held could be crowded out and, because a whole-corpus run replaces
    what it no longer keeps, deleted. Two were. Resolving the key here puts
    them back on a strategy that has no rank budget to be crowded out of.

    A fact is filed under every identity it answers to, so one fact can be in
    several buckets. That is deliberately a superset of what `subjects_match`
    accepts: two facts with different hard keys and one surface will meet here
    and be thrown out there. Blocking decides what is compared; the table
    decides what is related.
    """
    resolve = aliases or {}
    groups: dict[tuple[str, str], list[Fact]] = defaultdict(list)
    for fact in facts:
        predicate = resolve.get(fact.predicate_canonical, fact.predicate_canonical)
        for identity in subject_identities(fact):
            groups[(identity, predicate)].append(fact)

    pairs: set[PairKey] = set()
    for group in groups.values():
        for a, b in itertools.combinations(group, 2):
            if a.fact_id != b.fact_id:
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


def only_touching(pairs: set[PairKey], fresh: frozenset[str] | None) -> set[PairKey]:
    """Drop pairs where neither fact is new.

    Ingesting a fourth document must not re-judge the first three against each
    other. Those verdicts are already in the ledger and nothing about them has
    changed, so recomputing them is work whose only possible outcome is the
    answer already stored.
    """
    if fresh is None:
        return pairs
    return {(a, b) for a, b in pairs if a in fresh or b in fresh}


def encode(facts: list[Fact], encoder, vectors: dict | None = None):
    """The (n, d) matrix for these facts, encoding only what is not already known.

    `vectors` maps fact id to its embedding and is filled in as new ones are
    computed, so a caller that persists it pays the encoder once per fact for
    the life of the ledger instead of once per ingest. A fact id is
    content-addressed over its span and predicate, and re-extracting a document
    replaces its rows outright, so a cached vector cannot outlive the text it
    was made from.
    """
    import numpy as np

    texts = [fact.embed_text for fact in facts]
    if vectors is None:
        return encoder.encode(texts)

    missing = [i for i, fact in enumerate(facts) if fact.fact_id not in vectors]
    if missing:
        fresh_vectors = encoder.encode([texts[i] for i in missing])
        for position, index in enumerate(missing):
            vectors[facts[index].fact_id] = np.asarray(fresh_vectors[position], dtype=np.float32)
    return np.vstack([vectors[fact.fact_id] for fact in facts])


def semantic_pairs(
    facts: list[Fact],
    encoder,
    k: int = TOP_K,
    fresh: frozenset[str] | None = None,
    vectors: dict | None = None,
) -> set[PairKey]:
    """Top-k cosine neighbours over claim text, both directions unioned.

    With `fresh`, neighbours are read only for the new facts' rows. Every fact
    stays in the search space - a new fact must be able to find an old one -
    but nothing is asked about which old fact is near which other old fact.
    """
    from concord.compare.embed import top_k

    if len(facts) < 2 or encoder is None:
        return set()

    matrix = encode(facts, encoder, vectors)
    rows = (
        list(range(len(facts)))
        if fresh is None
        else [i for i, fact in enumerate(facts) if fact.fact_id in fresh]
    )

    pairs: set[PairKey] = set()
    for row, neighbours in zip(rows, top_k(matrix, k, rows=rows)):
        for column in neighbours:
            pairs.add(pair_key(facts[row], facts[column]))
    return pairs


def block(
    facts: list[Fact],
    encoder=None,
    k: int = TOP_K,
    width: float = BUCKET_WIDTH,
    window: int = BUCKET_WINDOW,
    fresh: frozenset[str] | None = None,
    vectors: dict | None = None,
    aliases: dict[str, str] | None = None,
) -> tuple[dict[PairKey, list[str]], BlockingStats]:
    """Run all three strategies and union them, keeping which one fired.

    `blocked_by` is stored on the relation so the README can report what each
    strategy actually bought, rather than asserting that all three were needed.

    `fresh` restricts the result to pairs touching those fact ids, which is what
    makes an incremental ingest incremental. The theoretical count drops to
    match, so the reduction figure stays honest about the work actually faced.

    `aliases` is the predicate registry's map, and it is what makes the exact
    strategy complete: a pair can only be stored if its comparison keys match,
    and with the alias map and the subject identities in hand this strategy
    enumerates exactly those pairs. The other two are recall insurance over a
    vocabulary the registry has not linked yet, and cost 3 ms.
    """
    by_id = {fact.fact_id: fact for fact in facts}
    n = len(facts)
    stats = BlockingStats(n_facts=n, theoretical_pairs=n * (n - 1) // 2)
    if fresh is not None:
        settled = n - len(fresh)
        stats.theoretical_pairs -= settled * (settled - 1) // 2

    produced = {
        "comparison_key": only_touching(comparison_key_pairs(facts, aliases), fresh),
        "value": only_touching(value_pairs(facts, width, window), fresh),
        "semantic": (
            semantic_pairs(facts, encoder, k, fresh, vectors)
            if encoder is not None
            else set()
        ),
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
