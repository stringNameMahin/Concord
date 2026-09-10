"""Complementary categories: facts that divide a whole rather than disagree.

A breakdown table states one predicate over several mutually exclusive
categories - workforce by region, revenue by segment, shareholding by class.
Every row disagrees with every other row by construction, and every row is
correct. Run through the decision table those pairs come out
`reconciled_by_context`, which is mechanically right and substantively wrong:
nothing was reconciled, because nothing ever looked like a conflict. On the
shipped ledger one three-region breakdown produced three such rows, and the
shape scales quadratically with the number of categories.

The test is arithmetic, not semantic. Take the facts that share a subject, a
predicate and every qualifier but one; if the values under the remaining
qualifier are percentages that add up to a whole, that qualifier's values are
the parts of one distribution. Pairs drawn from inside such a group are then
`unrelated` - not compared, not reconciled, not adjudicated.

"Add up to a whole" is decided by the precision intervals the figures were
written to, the same way `compare_values` decides agreement. Three integer
percentages of 63, 15 and 22 span [98.5, 101.5], which contains 100, so they
are a partition; four training-coverage percentages of 97.67, 100.00, 96.32
and 100.00 span nowhere near it, so they are four separate measurements that
happen to share a predicate.

Only percentages are covered. A breakdown in absolute units is the same shape,
but recognising it needs the total, and a total that is not in the ledger
cannot be checked - guessing at one would be exactly the kind of unfalsifiable
inference the rest of the system refuses. Where the total *is* extracted, the
parts corroborate against it through the ordinary path.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from concord.facts import Fact

WHOLE = 100.0

# A share of a whole is a share: not negative, not more than the whole. Both
# bounds are needed - without the ceiling, a group of growth rates could sum
# through 100 by accident.
MIN_SHARE = 0.0
MAX_SHARE = 100.0

# Two categories can add to a whole by coincidence far more easily than five
# can. The floor is still two, because a genuine two-way split (`Domestic` and
# `Overseas`) is a partition and the commonest one there is; the coincidence
# risk is bounded instead by requiring the sum to be a whole under the stated
# precision, which two arbitrary percentages rarely are.
MIN_CATEGORIES = 2

_WORD = re.compile(r"[^a-z0-9]+")


def _value_key(text: str) -> str:
    return _WORD.sub(" ", (text or "").lower()).strip()


def _shareable(fact: Fact) -> bool:
    quantity = fact.quantity
    return (
        quantity is not None
        and quantity.is_percent
        and not quantity.is_bounded
        and MIN_SHARE <= quantity.normalized <= MAX_SHARE
    )


def _sums_to_a_whole(facts: list[Fact]) -> bool:
    """Does one figure per category add up to the whole, given their precision?"""
    low = sum(fact.quantity.interval[0] for fact in facts)
    high = sum(fact.quantity.interval[1] for fact in facts)
    return low <= WHOLE <= high


@dataclass
class PartitionIndex:
    """Which facts are parts of one distribution, and on which qualifier.

    `groups` maps a fact id to the (group id, qualifier key) pairs it belongs
    to. A fact can be a part of more than one distribution - a figure broken
    down by region in one table and by segment in another - so membership is a
    set rather than a single label.
    """

    groups: dict[str, set[tuple[int, str]]] = field(default_factory=lambda: defaultdict(set))

    def __len__(self) -> int:
        return len({group for memberships in self.groups.values() for group in memberships})

    def key_for(self, a: Fact, b: Fact) -> str | None:
        """The qualifier whose values make these two facts complementary.

        Both facts must sit in the same distribution and hold different
        categories of it. Two rows of one breakdown reporting the same category
        are an ordinary pair and are judged as one.
        """
        shared = self.groups.get(a.fact_id, set()) & self.groups.get(b.fact_id, set())
        for _, key in sorted(shared):
            if _value_key(a.qualifier(key).value) != _value_key(b.qualifier(key).value):
                return key
        return None


def _context_signature(fact: Fact, without: str) -> tuple[tuple[str, str], ...]:
    """The fact's qualifier bag with one key held out, as a comparable key."""
    return tuple(
        sorted(
            (key, _value_key(fact.qualifier(key).value))
            for key in fact.qualifier_keys()
            if key != without
        )
    )


def index(facts: list[Fact], aliases: dict[str, str] | None = None) -> PartitionIndex:
    """Find every group of facts that divides one whole between its categories."""
    resolve = aliases or {}
    found = PartitionIndex()
    counter = 0

    by_measure: dict[tuple[str, str], list[Fact]] = defaultdict(list)
    for fact in facts:
        if _shareable(fact):
            canonical = resolve.get(fact.predicate_canonical, fact.predicate_canonical)
            by_measure[(fact.subject_id, canonical)].append(fact)

    for measure in sorted(by_measure):
        members = by_measure[measure]
        keys = {key for fact in members for key in fact.qualifier_keys()}
        for key in sorted(keys):
            candidates: dict[tuple, list[Fact]] = defaultdict(list)
            for fact in members:
                if fact.qualifier(key).known:
                    candidates[_context_signature(fact, key)].append(fact)

            for signature in sorted(candidates, key=str):
                group = candidates[signature]
                by_category: dict[str, Fact] = {}
                for fact in group:
                    by_category.setdefault(_value_key(fact.qualifier(key).value), fact)

                if len(by_category) < MIN_CATEGORIES:
                    continue
                if not _sums_to_a_whole(list(by_category.values())):
                    continue

                counter += 1
                for fact in group:
                    found.groups[fact.fact_id].add((counter, key))

    return found
