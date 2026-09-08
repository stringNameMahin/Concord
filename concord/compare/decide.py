"""The decision table. Deterministic, no LLM, one rule per row.

Every verdict in the system is produced here. The LLM is never asked whether
two numbers are equal, whether two dates are the same, or whether a unit
converts; those are Python, and they are the operations language models are
measurably worst at. What survives this table is a small residue of genuine
value conflicts, and only that residue is worth an LLM call.

    condition                                              verdict
    ----------------------------------------------------  --------------------
    comparison keys differ                                 unrelated
    keys match, bags compatible, intervals overlap         corroborates
    keys match, discriminating qualifier differs, values   unrelated
      agree                                                  (different conditions)
    keys match, values disagree, discriminating qualifier   insufficient_context
      absent on one side
    keys match, values disagree, bags differ on a           reconciled_by_context
      discriminating key                                     (LLM writes prose)
    keys match, values disagree, bags equal                 contradicts
                                                             (LLM adjudicates)

Two guards are enforced here in code rather than asked of a prompt:

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

from concord.facts import Fact, Qualifier, comparison_keys_match, key_matches
from concord.normalize.numbers import intervals_overlap
from concord.normalize.periods import bare_date, same_interval

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

    @property
    def compatible(self) -> bool:
        return not self.conflicting


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


def compare_qualifiers(
    a: Fact, b: Fact, discriminating: frozenset[str] = DISCRIMINATING
) -> ContextDiff:
    """Set difference over two qualifier bags. This is all case 3 is."""
    diff = ContextDiff()
    for key in sorted(a.qualifier_keys() | b.qualifier_keys()):
        left, right = a.qualifier(key), b.qualifier(key)

        if left.known and right.known:
            if compare_qualifier(left, right) == AGREE:
                diff.agreeing.append(key)
            else:
                diff.conflicting.append(key)
                if is_discriminating(key, discriminating):
                    diff.known_conflicting.append(key)
        else:
            diff.missing.append((key, a.fact_id if left.known else b.fact_id))

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
        # Agreement is only broken up by a condition we recognise. An
        # incidental key differing between two identical figures does not make
        # them separate facts.
        if diff.known_conflicting:
            return Decision(
                verdict="unrelated",
                rule_fired="context_differs_values_agree",
                explanation=(
                    f"The values agree ({evidence}) but the facts hold under different "
                    f"conditions: {_name(diff.known_conflicting)} differ. They describe separate "
                    "states of affairs rather than confirming one another."
                ),
                qualifier_key=diff.known_conflicting[0],
            )

        note = ""
        if missing_keys:
            note = (
                f" One side does not state {_name(missing_keys)}, so the agreement holds "
                "only under the conditions both documents do state."
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

    if diff.conflicting:
        key = diff.conflicting[0]
        return Decision(
            verdict="reconciled_by_context",
            rule_fired="discriminating_qualifier_differs",
            explanation=(
                f"The values disagree ({evidence}), and the facts differ on "
                f"{_name(diff.conflicting)}: "
                f"{a.qualifier(key).value!r} against {b.qualifier(key).value!r}. "
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
            f"({_name(diff.agreeing) or 'no qualifiers on either side'}), yet {evidence}."
        ),
        needs_llm=True,
    )
