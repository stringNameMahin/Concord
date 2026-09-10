"""The materialised fact record, and the two keys every comparison rests on.

`concord.extract.models.FactOut` is what the model is allowed to say. `Fact` is
what the system believes: the same claim with its figure parsed into base
units, its period labels resolved to intervals, and its qualifier bag carrying
provenance. Everything downstream of extraction reads `Fact` and never the raw
model output.

A fact splits into exactly two keys:

  comparison key = (subject key or normalised surface, canonical predicate)
                   "are these two facts even about the same thing?"
  context key    = the qualifier bag
                   "under what conditions?"

The verdict follows from those two keys, which is why both live here rather
than inside the comparison code.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from concord.normalize.numbers import NotANumber, Quantity, parse_quantity
from concord.normalize.periods import Period, bare_date, parse_period

# Qualifier keys whose value is a date interval rather than a string, so they
# are compared by interval and not by wording. Open like every other key: this
# only says how to compare the value, never which keys may exist.
PERIOD_KEYS = ("period", "as_of", "as_at")

# Trailing corporate legal forms, as whole phrases. Read only by
# `same_entity`, and only to strip one suffix off the end of a name that
# already matches word for word - see that function for why the list is safe
# where a general suffix-stripping rule would not be. Bare "as" (the
# Norwegian form) is deliberately absent: it is a common English word, and
# the only entry that could plausibly end a subject surface by accident.
LEGAL_FORMS = frozenset(
    {
        "limited",
        "ltd",
        "company",
        "company limited",
        "co",
        "co ltd",
        "corporation",
        "corp",
        "inc",
        "incorporated",
        "plc",
        "private limited",
        "private ltd",
        "pvt ltd",
        "pvt limited",
        "pte ltd",
        "pte limited",
        "llp",
        "llc",
        "lp",
        "limited liability partnership",
        "gmbh",
        "ag",
        "nv",
        "bv",
        "sa",
        "sas",
        "spa",
        "srl",
        "ab",
        "aps",
        "oy",
        "oyj",
        "kk",
        "sdn bhd",
        "bhd",
        "pjsc",
        "jsc",
    }
)

_WORD = re.compile(r"[^a-z0-9]+")


def key_matches(key: str, known: frozenset[str] | tuple[str, ...]) -> bool:
    """Does an open-vocabulary key name one of a set of known conditions?

    The extractor coins its own keys, so the same condition arrives as
    `as_of`, `as_of_date` and `reporting_period` across three documents. A
    known key matches when its words appear as a run inside the key's words,
    which recognises those without matching `date` on its own.
    """
    tokens = key.split("_")
    for candidate in known:
        parts = candidate.split("_")
        span = len(parts)
        if any(tokens[i : i + span] == parts for i in range(len(tokens) - span + 1)):
            return True
    return False


def canonical_predicate(text: str) -> str:
    """Fold a predicate to a comparable form without interpreting it.

    Case and punctuation only. Anything that needs to know that
    `revenue_from_services` and `revenue_from_operations` mean the same thing
    is the predicate registry's job, not a string rule's.
    """
    return _WORD.sub("_", (text or "").strip().lower()).strip("_")


def normalize_surface(text: str) -> str:
    """Normalise a subject surface form for use as a fallback identity."""
    lowered = (text or "").strip().lower()
    lowered = re.sub("['\u2019]s\\b", "", lowered)
    return _WORD.sub(" ", lowered).strip()


def surface_tokens(text: str) -> frozenset[str]:
    return frozenset(normalize_surface(text).split())


def subject_names_the_measurement(subject_surface: str, predicate: str) -> bool:
    """True when the subject is the row label instead of the thing measured.

    A model reading a table can hand back `Adjusted EBITDA / adjusted_ebitda`
    or `Total equity / total_equity`, naming the metric twice and never the
    organisation the table is about. Prompt rule 8 forbids it and a model still
    does it, so this is the deterministic check - the same shape of guard the
    quote gets from the aligner and the qualifier gets from the named-key test.

    The test is that every word of the subject already appears in the
    predicate. That is what a doubled row label looks like, and it stays inside
    what the rules already promise: rule 7 bars the entity from the predicate,
    so a subject wholly contained in one is either the metric restated or a
    predicate that broke rule 7. It needs no vocabulary and no subject area,
    which is the whole constraint on this system. Measured on the shipped
    ledger: 7 of 506 facts, every one of them a genuine instance, no false
    positives - see bug 23 in docs/status.md.
    """
    subject = surface_tokens(subject_surface)
    return bool(subject) and subject <= set(canonical_predicate(predicate).split("_"))


@dataclass(frozen=True)
class Qualifier:
    """One condition on a fact, with how we came to know it.

    `provenance` is the load-bearing field. `absent` is not an empty value; it
    is the record that the document never said, which is what lets the engine
    abstain instead of guessing.
    """

    key: str
    value: str
    provenance: str = "stated"  # stated | inherited | absent
    period: Period | None = None

    @property
    def known(self) -> bool:
        return self.provenance != "absent" and bool(self.value)


def absent(key: str) -> Qualifier:
    return Qualifier(key=key, value="", provenance="absent")


@dataclass(frozen=True)
class Evidence:
    """Where the claim lives in the parser's text, to the byte."""

    doc_id: str
    page: int
    char_start: int
    char_end: int
    quote: str
    align_status: str = "exact"
    block_type: str | None = None

    @property
    def span(self) -> tuple[str, int, int]:
        return (self.doc_id, self.char_start, self.char_end)


@dataclass
class Fact:
    fact_id: str
    doc_id: str
    claim_text: str
    subject_surface: str
    predicate: str
    value_kind: str
    value_raw: str
    evidence: Evidence
    subject_key: str | None = None
    subject_type: str | None = None
    quantity: Quantity | None = None
    qualifiers: dict[str, Qualifier] = field(default_factory=dict)
    predicate_canonical: str = ""
    confidence: float = 1.0
    flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.predicate_canonical:
            self.predicate_canonical = canonical_predicate(self.predicate)

    @property
    def subject_id(self) -> str:
        """The hard key when the document gave one, else the surface form."""
        return self.subject_key or normalize_surface(self.subject_surface)

    @property
    def comparison_key(self) -> str:
        return f"{self.subject_id}|{self.predicate_canonical}"

    @property
    def normalized(self) -> float | None:
        return self.quantity.normalized if self.quantity else None

    def qualifier(self, key: str) -> Qualifier:
        """Every key is answerable; unrecorded ones answer `absent`."""
        return self.qualifiers.get(key, absent(key))

    def qualifier_keys(self) -> frozenset[str]:
        return frozenset(k for k, q in self.qualifiers.items() if q.known)

    @property
    def embed_text(self) -> str:
        return self.claim_text or f"{self.subject_surface} {self.predicate} {self.value_raw}"


def same_entity(left: str, right: str) -> bool:
    """Is one surface form the other plus a trailing corporate legal form?

    This is the whole of the surface-form latitude the system allows, and it
    is deliberately one hop wide. The rule the corpus needs is `Delhivery` and
    `Delhivery Limited`; the rule it must not have is anything that lets a
    parent absorb a subsidiary. Two constraints together give that:

    - the shorter name must be an ordered **prefix** of the longer one, not a
      token subset. `Delhivery Limited` is not a prefix of `Delhivery Corp
      Limited`, so the UK subsidiary cannot reach the parent even though the
      only extra word is one this module recognises.
    - what remains must be exactly one legal form, spelled as a whole. `corp
      limited` is not a legal form, so a name cannot pick up two of them.

    `LEGAL_FORMS` is a vocabulary, and this is the one place the system keeps
    one. It is a naming convention rather than a subject area - it says how
    organisations write their own names, not what any of them do - and the
    prefix rule means an entry can only ever strip a suffix, never bridge two
    names whose content words differ. Measured over the 223 distinct subject
    surfaces in the shipped ledger, it joins exactly two pairs, both correct.
    """
    if left == right:
        return True
    if len(right) < len(left):
        left, right = right, left
    return right.startswith(left + " ") and right[len(left) + 1 :] in LEGAL_FORMS


def subjects_match(a: Fact, b: Fact) -> bool:
    """Decide whether two facts speak about the same subject.

    Two hard keys must be equal - a CIN or an ISIN is an identity, and two
    different ones are two different entities no matter how alike the names
    look. Without a hard key on both sides the surface forms have to be the
    same name, up to the trailing legal form `same_entity` allows.

    This used to accept one token set being a subset of the other. That rule
    matched `Delhivery` to `Delhivery Limited` as intended, and also matched
    the parent to `Delhivery Freight Services Private Limited`, `current other
    assets` to `current other financial assets`, and `cash and cash
    equivalents` to `bank balances other than cash and cash equivalents`. On
    the shipped ledger it produced twenty mismatched-subject pairs, including
    every one of the eight false contradictions. A subset of a name is a
    different name, not a shorter one.
    """
    if a.subject_key and b.subject_key:
        return a.subject_key == b.subject_key

    left, right = normalize_surface(a.subject_surface), normalize_surface(b.subject_surface)
    if not left or not right:
        return False
    return same_entity(left, right)


def predicates_match(a: Fact, b: Fact, aliases: dict[str, str] | None = None) -> bool:
    """Canonical equality, with the predicate registry consulted if present.

    `aliases` maps an alias to its canonical predicate. Phase 4 runs with an
    empty map; Phase 7 fills it as the registry learns, and nothing here
    changes when it does.
    """
    resolve = aliases or {}
    left = resolve.get(a.predicate_canonical, a.predicate_canonical)
    right = resolve.get(b.predicate_canonical, b.predicate_canonical)
    return bool(left) and left == right


def comparison_keys_match(a: Fact, b: Fact, aliases: dict[str, str] | None = None) -> bool:
    return subjects_match(a, b) and predicates_match(a, b, aliases)


def make_fact_id(doc_id: str, start: int, end: int, predicate: str) -> str:
    """Content-addressed, so re-ingesting a document yields the same ids.

    Phase 7 appends new documents to an existing ledger; a fact id that
    depends on run order would make that impossible to do incrementally.
    """
    digest = hashlib.sha1(
        f"{doc_id}:{start}:{end}:{canonical_predicate(predicate)}".encode()
    ).hexdigest()
    return f"f_{digest[:16]}"


def _field(item, name: str, default=None):
    """Read a field from either a pydantic model or a plain dict."""
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _as_interval(value: str, fy_end_month: int | None) -> Period | None:
    """Resolve a period qualifier to an interval, instants included.

    An `as_of` qualifier names a moment, not a span - "as of March 31, 2024"
    is a stock quantity's date. Representing it as a zero-length interval lets
    one comparison rule serve both: two instants are the same condition when
    they are the same day, exactly as two spans are when their bounds match.
    """
    period = parse_period(value, fy_end_month)
    if period is not None:
        return period

    moment = bare_date(value)
    if moment is None:
        return None
    return Period(moment, moment, value.strip(), "instant", "stated:date")


def build_qualifiers(
    pairs, fy_end_month: int | None = None, inherited: dict[str, str] | None = None
) -> dict[str, Qualifier]:
    """Turn the model's qualifier list into a bag, resolving period labels.

    Parsing happens here so that a period is compared as an interval by the
    decision table. If a label does not resolve, the qualifier survives as a
    string and is compared as one; an unparsed label must never silently
    become a missing qualifier.

    `inherited` carries conditions the document states structurally rather
    than in the sentence - the bullet heading a figure sits under. They fill
    keys the extractor left empty and never overwrite one it filled: a
    qualifier read off the claim itself is better evidence than one read off
    the layout, and where both exist the sentence wins.
    """
    bag: dict[str, Qualifier] = {}
    for item in pairs:
        key = canonical_predicate(_field(item, "key", "") or "")
        value = _field(item, "value", "") or ""
        provenance = _field(item, "provenance", "stated") or "stated"
        if not key or not str(value).strip():
            continue

        period = None
        if key_matches(key, PERIOD_KEYS):
            period = _as_interval(str(value), fy_end_month)

        bag[key] = Qualifier(
            key=key, value=str(value).strip(), provenance=provenance, period=period
        )

    for raw_key, raw_value in (inherited or {}).items():
        key = canonical_predicate(raw_key)
        value = str(raw_value or "").strip()
        if not key or not value or key in bag:
            continue
        period = _as_interval(value, fy_end_month) if key_matches(key, PERIOD_KEYS) else None
        bag[key] = Qualifier(key=key, value=value, provenance="inherited", period=period)
    return bag


def materialize(
    out,
    doc_id: str,
    alignment,
    page: int,
    fy_end_month: int | None = None,
    default_scale: str | None = None,
    default_unit: str | None = None,
    inherited: dict[str, str] | None = None,
) -> Fact:
    """Build a `Fact` from one grounded extraction.

    The figure is parsed here rather than at extraction time because the scale
    and currency may come from the context stack (a table header) instead of
    the cell, and only the caller knows the stack. A figure that will not parse
    is not an error: the fact keeps its text value, is flagged, and simply
    never takes part in an interval comparison.
    """
    quantity = None
    flags: list[str] = []
    kind = out.value.kind

    if kind == "quantity":
        try:
            quantity = parse_quantity(
                out.value.raw,
                default_scale=out.value.scale or default_scale,
                default_unit=out.value.unit or default_unit,
            )
        except NotANumber:
            flags.append("unparsed_quantity")

    if subject_names_the_measurement(out.subject_surface, out.predicate):
        flags.append("subject_names_the_measurement")

    evidence = Evidence(
        doc_id=doc_id,
        page=page,
        char_start=alignment.start,
        char_end=alignment.end,
        quote=alignment.text,
        align_status=alignment.status,
    )

    return Fact(
        fact_id=make_fact_id(doc_id, alignment.start, alignment.end, out.predicate),
        doc_id=doc_id,
        claim_text=out.claim_text,
        subject_surface=out.subject_surface,
        subject_key=out.subject_key,
        subject_type=out.subject_type,
        predicate=out.predicate,
        value_kind=kind,
        value_raw=out.value.raw,
        quantity=quantity,
        qualifiers=build_qualifiers(out.qualifiers, fy_end_month, inherited),
        evidence=evidence,
        confidence=out.confidence,
        flags=flags,
    )
