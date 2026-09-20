"""The decision table. Deterministic, no LLM, one rule per row.

Every verdict in the system is produced here. The LLM is never asked whether
two numbers are equal, whether two dates are the same, or whether a unit
converts; those are Python, and they are the operations language models are
measurably worst at. What survives this table is a small residue of genuine
value conflicts, and only that residue is worth an LLM call.

    condition                                              verdict
    ----------------------------------------------------  --------------------
    comparison keys differ                                 unrelated
    keys match, facts are two categories of one            unrelated
      distribution                                           (complementary)
    keys match, bags compatible, intervals overlap         corroborates
    keys match, intervals overlap only because one figure   insufficient_context
      is written to a single digit                            (imprecise)
    keys match, discriminating qualifier differs, values   unrelated
      agree                                                  (different conditions)
    keys match, values disagree, discriminating qualifier   insufficient_context
      absent on one side
    keys match, values disagree, the only condition that    insufficient_context
      differs is one dimension spelled two ways, neither      (unestablished)
      of which resolves
    keys match, values disagree, bags differ on a           reconciled_by_context
      discriminating key                                     (LLM writes prose)
    keys match, values disagree, bags equal                 contradicts
                                                             (LLM adjudicates)

Complementary-category guard - two rows of one breakdown are parts of a
whole, not rival claims about it, so they never reach the value comparison at
all. `concord.compare.partition` decides which facts those are, arithmetically
rather than semantically; this module only reads the answer.

Two further guards are enforced here in code rather than asked of a prompt:

Missing-qualifier guard - a pair cannot be called `contradicts` when a
discriminating qualifier is present on one side and absent on the other. The
ceiling is `insufficient_context` and the response names the missing key. The
guard binds on disagreement, which is the direction that produces false
contradictions; when the values agree, a qualifier the other side never stated
cannot turn agreement into a conflict, so the pair still corroborates and the
missing key is named in the explanation.

Precision-interval comparison - values are never compared as points. Each
figure carries the interval implied by the digits it committed to, and overlap
decides agreement. That is what lets `8,142 Cr` corroborate `81,415.38 mn`
without a tolerance constant anyone has to defend.

Single-digit guard - the same rule degenerates at one written digit. `1
billion` spans half its own value either way, so it overlaps most of its
decade, and an overlap the coarse figure alone produces is consistency rather
than confirmation. Those pairs abstain and say which figure is carrying the
overlap. See `overlap_rests_on_imprecision`.

Bags are diffed by *dimension*, not by literal key name. An open vocabulary
spells one condition several ways - `period`, `as_at`, `time_period` and
`financial_year` all answer "when?" - and comparing the names meant two facts
that both stated the time each read as stating a condition the other lacked, so
the missing-qualifier guard abstained and said so in a sentence that was not
true. `concord.facts.dimension_of` holds the one dimension that is declared and
the moments deliberately excluded from it; everything else is its own
dimension, which is what the key-by-key diff already did.

`DISCRIMINATING` is an allowlist, and it binds in exactly one branch. The
qualifier bag is an open vocabulary: on the starter corpus the extractor coined
`service`, `category`, `condition` and `auditor`, none of which any fixed list
would contain, and every one of them was load-bearing. So:

- values disagree, any stated condition differs -> `reconciled_by_context`.
  Two facts holding under different stated conditions are not a conflict,
  whatever the key is called.
- values disagree, any condition is stated on one side and absent on the other
  -> `insufficient_context`, naming the key. We cannot know whether a key we
  do not recognise is the one that would settle the question.
- values agree, a *recognised* condition differs -> `unrelated`. This is the
  one place the allowlist binds, because an incidental key differing between
  two identical figures does not make them separate facts.

So a pair is only ever called `contradicts` when both documents state the same
conditions and still disagree. Measured on the Delhivery trio, consulting the
allowlist in the first two branches turned nine of twelve contradictions into
false positives; every one had its reconciling qualifier sitting in the bag
under a key the list did not know.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from concord.compare.partition import PartitionIndex
from concord.facts import (
    TIME_DIMENSION,
    Fact,
    Qualifier,
    comparison_keys_match,
    dimension_of,
    key_matches,
)
from concord.normalize.numbers import intervals_overlap, overlap_rests_on_imprecision
from concord.normalize.periods import bare_date, same_interval, years_compatible

VERDICTS = (
    "corroborates",
    "contradicts",
    "reconciled_by_context",
    "unrelated",
    "insufficient_context",
)

# Conditions under which a claim can be true and its neighbour false without
# either being wrong. These are dimensions of a document, not a vocabulary of
# one subject area: when a fact holds, as of when, over what scope, on what
# basis, for which segment or place. The bag itself stays open - this set only
# decides which differences are allowed to change a verdict.
DISCRIMINATING = frozenset(
    {
        "period",
        "as_of",
        "as_at",
        "consolidation",
        "basis",
        "scope",
        "segment",
        "geography",
    }
)

# Conditions that name two views of one subject rather than two different
# states of affairs, and so cannot turn agreement into `unrelated`.
#
# Consolidated and standalone statements report the same entity over the same
# year twice. When their figures *disagree* that is the whole explanation, and
# `consolidation` earns its place in `DISCRIMINATING` for exactly that. When
# they *agree* the agreement is the interesting part: an address, a CIN or an
# incorporation date restated in both sets of statements is one fact stated
# twice, and a subsidiary contribution of nil is a real corroboration too.
# Calling those `unrelated` would lose three correct verdicts on the shipped
# ledger to buy nothing.
#
# This binds only in the values-agree branch. A consolidation difference still
# reconciles a disagreement, and still forces `insufficient_context` when only
# one side states it.
NON_SEPARATING = frozenset({"consolidation"})

AGREE, DISAGREE, INCOMPARABLE = "agree", "disagree", "incomparable"


def is_discriminating(key: str, keys: frozenset[str] = DISCRIMINATING) -> bool:
    """Match on the key or on a run of words inside it.

    `reporting_period`, `as_of_date` and `geography_region` are the same
    conditioning dimensions under longer names, and an open vocabulary
    produces all three. Matching on word runs rather than single words is what
    lets `as_of_date` match `as_of` without `date` matching on its own.
    """
    return key_matches(key, keys)


@dataclass(frozen=True)
class Decision:
    verdict: str
    rule_fired: str
    explanation: str
    qualifier_key: str | None = None
    needs_llm: bool = False
    missing_keys: tuple[str, ...] = ()

    @property
    def candidate(self) -> bool:
        """A candidate verdict is one Phase 5 may still downgrade."""
        return self.needs_llm


@dataclass
class ContextDiff:
    conflicting: list[str] = field(default_factory=list)
    missing: list[tuple[str, str]] = field(default_factory=list)  # (key, side present)
    agreeing: list[str] = field(default_factory=list)
    known_conflicting: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    # Conditions that differ only in the sense that the two documents reached
    # for different key names *and* neither label resolves, so nothing about
    # the values was established. Comparing dimensions rather than key names
    # created this case: `period="year ended March 31, 2026"` against
    # `date="March 31, 2026"` is one moment written twice, and before the
    # dimensions met, the pair abstained because each key was missing from the
    # other side. It must not now be reconciled *by* that difference.
    unestablished: list[str] = field(default_factory=list)
    # How each compared condition is spelled on the two sides. Entries are
    # keyed by the left-hand spelling, which is what the lists above carry, so
    # everything reading them keeps working; this is what lets an explanation
    # quote the right value when the two documents named one condition two
    # ways. See `concord.facts.dimension_of`.
    spellings: dict[str, tuple[str, str]] = field(default_factory=dict)

    @property
    def compatible(self) -> bool:
        return not self.conflicting

    @property
    def proven_conflicting(self) -> list[str]:
        """Recognised conditions we established differ, rather than assumed."""
        return [key for key in self.known_conflicting if key not in self.unreadable]

    def spelled(self, key: str) -> tuple[str, str]:
        return self.spellings.get(key, (key, key))

    def sides(self, a: Fact, b: Fact, key: str) -> tuple[Qualifier, Qualifier]:
        """The two qualifiers behind one entry, under each side's own name."""
        left, right = self.spelled(key)
        return a.qualifier(left), b.qualifier(right)

    def name(self, key: str) -> str:
        """The condition as a reader should see it, both spellings if they differ."""
        left, right = self.spelled(key)
        return left if left == right else f"{left}/{right}"

    def names(self, keys) -> str:
        return ", ".join(self.name(key) for key in keys)


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def compare_qualifier(a: Qualifier, b: Qualifier) -> str:
    """Compare one qualifier on both sides, by interval where it is one.

    Periods compare as intervals: only equal intervals are the same condition.
    Overlapping-but-unequal intervals are a difference, not a match, because a
    nine-month figure and a full-year figure are genuinely different facts even
    though one contains the other.
    """
    if a.period and b.period:
        return AGREE if same_interval(a.period, b.period) else DISAGREE

    left, right = bare_date(a.value), bare_date(b.value)
    if left and right:
        return AGREE if left == right else DISAGREE

    return AGREE if _normalize_text(a.value) == _normalize_text(b.value) else DISAGREE


def unreadable_period(left: Qualifier, right: Qualifier) -> bool:
    """A period key whose difference we compared as text, not as intervals.

    At least one label did not parse, so the comparison fell through to string
    equality and the only evidence of a difference is that the two strings are
    spelled differently - and two spellings are often one period.
    In the shipped corpus "till FY26" against "inception till FY26" and
    "financial year under review" against "financial year ended March 31, 2026"
    are both the same period written twice. The years each label names are
    still checked, so "April-December 2024" against "April-December 2023" stays
    a difference.

    Asked of the dimension rather than of `PERIOD_KEYS`, because the pair being
    compared may now be `as_at` against `date`: the reason a wording difference
    is weak evidence does not depend on which of time's several spellings each
    document reached for. Widening this can only keep an agreement that would
    otherwise have been broken, which is the safe direction.
    """
    if TIME_DIMENSION not in (dimension_of(left.key), dimension_of(right.key)):
        return False
    if dimension_of(left.key) != dimension_of(right.key):
        return False
    if left.period and right.period:
        return False
    return years_compatible(left.value, right.value)


def _by_dimension(fact: Fact) -> dict[str, list[str]]:
    """This fact's known qualifier keys, grouped by the question they answer."""
    grouped: dict[str, list[str]] = {}
    for key in sorted(fact.qualifier_keys()):
        grouped.setdefault(dimension_of(key), []).append(key)
    return grouped


def _align(left_keys: list[str], right_keys: list[str]):
    """Pair up two sides' spellings of one dimension.

    A key spelled the same on both sides pairs with itself, which is every pair
    the old key-by-key diff already handled and is why this change moves
    nothing in the common case. Only what is left over may cross spellings, and
    only when there is exactly one candidate a side: two unmatched keys against
    one is an ambiguity, and guessing which of them answers the other would
    invent a reading. Those fall through as unpaired and are reported missing,
    which is what the old code did with all of them.
    """
    shared = [key for key in left_keys if key in right_keys]
    pairs = [(key, key) for key in shared]
    rest_left = [key for key in left_keys if key not in shared]
    rest_right = [key for key in right_keys if key not in shared]

    if len(rest_left) == 1 and len(rest_right) == 1:
        pairs.append((rest_left[0], rest_right[0]))
        return pairs, [], []
    return pairs, rest_left, rest_right


def compare_qualifiers(
    a: Fact, b: Fact, discriminating: frozenset[str] = DISCRIMINATING
) -> ContextDiff:
    """Set difference over two qualifier bags, by dimension rather than by name.

    The bags are still open vocabularies and nothing here decides what a key
    may be called. What it does decide is that two keys answering the same
    question are one condition to compare rather than two conditions each
    missing from the other side - see `concord.facts.dimension_of` for which
    keys those are and how narrowly the set is drawn.
    """
    diff = ContextDiff()
    left_bag, right_bag = _by_dimension(a), _by_dimension(b)

    for dimension in sorted(set(left_bag) | set(right_bag)):
        left_keys = left_bag.get(dimension, [])
        right_keys = right_bag.get(dimension, [])

        if not left_keys or not right_keys:
            holder = a.fact_id if left_keys else b.fact_id
            for key in left_keys or right_keys:
                diff.missing.append((key, holder))
            continue

        pairs, spare_left, spare_right = _align(left_keys, right_keys)
        for left_key, right_key in pairs:
            left, right = a.qualifier(left_key), b.qualifier(right_key)
            diff.spellings[left_key] = (left_key, right_key)

            if compare_qualifier(left, right) == AGREE:
                diff.agreeing.append(left_key)
                continue

            diff.conflicting.append(left_key)
            # Both spellings must be recognised before a difference is allowed
            # to break up two agreeing figures. `date` alone is deliberately
            # not a condition, and pairing it with `period` must not promote it
            # into one - that is the merge this change is most at risk of
            # making by accident, so it is refused outright.
            if (
                is_discriminating(left_key, discriminating)
                and is_discriminating(right_key, discriminating)
                and not key_matches(left_key, NON_SEPARATING)
            ):
                diff.known_conflicting.append(left_key)
            if unreadable_period(left, right):
                diff.unreadable.append(left_key)
                if left_key != right_key:
                    diff.unestablished.append(left_key)

        for key in spare_left:
            diff.missing.append((key, a.fact_id))
        for key in spare_right:
            diff.missing.append((key, b.fact_id))

    # Name a recognised condition first, so a pair differing on both `period`
    # and some incidental key reports the one a reader will recognise.
    diff.conflicting.sort(key=lambda key: key not in diff.known_conflicting)
    diff.missing.sort(key=lambda item: not is_discriminating(item[0], discriminating))
    return diff


def compare_values(a: Fact, b: Fact) -> tuple[str, str]:
    """Interval against interval, never point against point.

    Returns the outcome and the sentence fragment that justifies it, so the
    explanation quotes the same arithmetic the verdict used.
    """
    if a.value_kind == "quantity" and b.value_kind == "quantity":
        left, right = a.quantity, b.quantity
        if left is None or right is None:
            return INCOMPARABLE, "one figure could not be parsed into a number"
        if left.is_percent != right.is_percent:
            return INCOMPARABLE, "a percentage cannot be compared with an absolute quantity"
        if left.unit and right.unit and left.unit != right.unit:
            return (
                INCOMPARABLE,
                f"stated in {left.unit} and {right.unit}; no unit conversion is performed",
            )

        fragment = (
            f"{left.raw} spans [{left.interval[0]:.6g}, {left.interval[1]:.6g}] "
            f"and {right.raw} spans [{right.interval[0]:.6g}, {right.interval[1]:.6g}]"
        )
        if intervals_overlap(left.interval, right.interval):
            return AGREE, f"{fragment}, which overlap"
        return DISAGREE, f"{fragment}, which are disjoint"

    if a.value_kind != b.value_kind:
        return INCOMPARABLE, f"a {a.value_kind} value cannot be compared with a {b.value_kind} one"

    left_date, right_date = bare_date(a.value_raw), bare_date(b.value_raw)
    if left_date and right_date:
        if left_date == right_date:
            return AGREE, f"both resolve to {left_date.isoformat()}"
        return DISAGREE, f"{left_date.isoformat()} and {right_date.isoformat()} are different dates"

    if _normalize_text(a.value_raw) == _normalize_text(b.value_raw):
        return AGREE, f"both state {a.value_raw!r}"
    return DISAGREE, f"{a.value_raw!r} and {b.value_raw!r} differ"


def _name(keys) -> str:
    return ", ".join(keys)


def decide(
    a: Fact,
    b: Fact,
    aliases: dict[str, str] | None = None,
    discriminating: frozenset[str] = DISCRIMINATING,
    partitions: PartitionIndex | None = None,
) -> Decision:
    """Apply the decision table to one candidate pair."""
    if not comparison_keys_match(a, b, aliases):
        return Decision(
            verdict="unrelated",
            rule_fired="comparison_key_differs",
            explanation=(
                f"Different comparison keys: {a.comparison_key!r} and {b.comparison_key!r}. "
                "The two facts are not about the same thing."
            ),
        )

    # Before the values are looked at, because there is nothing to look at:
    # two categories of one distribution disagree by construction and neither
    # is evidence about the other.
    partition_key = partitions.key_for(a, b) if partitions is not None else None
    if partition_key:
        return Decision(
            verdict="unrelated",
            rule_fired="complementary_categories",
            explanation=(
                f"The two facts are different categories of one distribution: they share "
                f"a comparison key and differ only on {partition_key!r} "
                f"({a.qualifier(partition_key).value!r} against "
                f"{b.qualifier(partition_key).value!r}), whose values across the group add "
                "up to the whole. They divide a total rather than making rival claims "
                "about it, so there is no apparent conflict to reconcile."
            ),
            qualifier_key=partition_key,
        )

    outcome, evidence = compare_values(a, b)
    diff = compare_qualifiers(a, b, discriminating)
    missing_keys = tuple(key for key, _ in diff.missing)

    if outcome == INCOMPARABLE:
        return Decision(
            verdict="insufficient_context",
            rule_fired="incomparable_values",
            explanation=f"Same comparison key, but {evidence}.",
            missing_keys=missing_keys,
        )

    if outcome == AGREE:
        # Agreement is only broken up by a condition we recognise and can show
        # differs. An incidental key differing between two identical figures
        # does not make them separate facts, and neither does a period label
        # that only differs as text.
        proven = diff.proven_conflicting
        if proven:
            return Decision(
                verdict="unrelated",
                rule_fired="context_differs_values_agree",
                explanation=(
                    f"The values agree ({evidence}) but the facts hold under different "
                    f"conditions: {diff.names(proven)} differ. They describe separate "
                    "states of affairs rather than confirming one another."
                ),
                qualifier_key=proven[0],
            )

        # An overlap that exists only because one figure is written to a
        # single digit is consistency, not confirmation. See
        # `overlap_rests_on_imprecision`.
        coarse = (
            overlap_rests_on_imprecision(a.quantity, b.quantity)
            if a.quantity is not None and b.quantity is not None
            else None
        )
        if coarse is not None:
            return Decision(
                verdict="insufficient_context",
                rule_fired="agreement_rests_on_imprecision",
                explanation=(
                    f"The intervals overlap ({evidence}), but only because {coarse.raw!r} is "
                    f"written to a single digit and so stands for anything in "
                    f"[{coarse.interval[0]:.6g}, {coarse.interval[1]:.6g}]. Read at the "
                    "precision the other figure commits to, the two do not meet, so the "
                    "overlap is not evidence that the documents agree."
                ),
                missing_keys=missing_keys,
            )

        note = ""
        if missing_keys:
            note = (
                f" One side does not state {_name(missing_keys)}, so the agreement holds "
                "only under the conditions both documents do state."
            )
        if diff.unreadable:
            left, right = diff.sides(a, b, diff.unreadable[0])
            note += (
                f" The two sides spell {diff.names(diff.unreadable)} differently "
                f"({left.value!r} against {right.value!r}) and neither spelling "
                "resolves to an interval, so the wording is not evidence that the "
                "conditions differ."
            )
        return Decision(
            verdict="corroborates",
            rule_fired="keys_match_intervals_overlap",
            explanation=f"Same comparison key and compatible context; {evidence}.{note}",
            missing_keys=missing_keys,
        )

    # The values disagree. Everything below decides whether that disagreement
    # is a conflict, an artefact of context, or something we cannot tell.
    if diff.missing:
        key, present_on = diff.missing[0]
        holder = "the first" if present_on == a.fact_id else "the second"
        return Decision(
            verdict="insufficient_context",
            rule_fired="missing_qualifier_guard",
            explanation=(
                f"The values disagree ({evidence}), but only {holder} fact states "
                f"{key!r}. Without it the pair cannot be judged a contradiction."
            ),
            qualifier_key=key,
            missing_keys=missing_keys,
        )

    # A condition can only account for a difference in value if we established
    # that the condition differs. Where the two sides merely spelled one
    # dimension two ways and neither label resolved, we did not: those pairs
    # abstain and name the condition, rather than being handed an explanation
    # that rests on a difference nobody demonstrated. This binds only on the
    # cross-spelled case, which is the one comparing dimensions introduced;
    # a difference between two identical key names is left exactly as the table
    # already decided it - see `test_an_unreadable_period_does_not_rescue_a_disagreement`.
    established = [key for key in diff.conflicting if key not in diff.unestablished]
    if diff.conflicting and not established:
        key = diff.conflicting[0]
        left, right = diff.sides(a, b, key)
        return Decision(
            verdict="insufficient_context",
            rule_fired="unestablished_context_difference",
            explanation=(
                f"The values disagree ({evidence}). The two facts both state "
                f"{diff.name(key)}, under different names and in wordings neither of "
                f"which resolves to an interval ({left.value!r} against {right.value!r}), "
                "so it cannot be shown that the condition differs at all - and a "
                "difference that has not been shown cannot account for the figures."
            ),
            qualifier_key=key,
            missing_keys=missing_keys,
        )

    if established:
        key = established[0]
        left, right = diff.sides(a, b, key)
        return Decision(
            verdict="reconciled_by_context",
            rule_fired="discriminating_qualifier_differs",
            explanation=(
                f"The values disagree ({evidence}), and the facts differ on "
                f"{diff.names(diff.conflicting)}: "
                f"{left.value!r} against {right.value!r}. "
                "The difference in context accounts for the difference in value."
            ),
            qualifier_key=key,
            needs_llm=True,
        )

    return Decision(
        verdict="contradicts",
        rule_fired="same_context_disjoint_intervals",
        explanation=(
            f"Same comparison key and the same stated context "
            f"({diff.names(diff.agreeing) or 'no qualifiers on either side'}), yet {evidence}."
        ),
        needs_llm=True,
    )
