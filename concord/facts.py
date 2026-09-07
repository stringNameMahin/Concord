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
from concord.normalize.periods import Period, parse_period

# Qualifier keys whose value is a date interval rather than a string, so they
# are compared by interval and not by wording. Open like every other key: this
# only says how to compare the value, never which keys may exist.
PERIOD_KEYS = ("period", "as_of", "as_at")

_WORD = re.compile(r"[^a-z0-9]+")


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
    lowered = re.sub(r"['’]s\b", "", lowered)
    return _WORD.sub(" ", lowered).strip()


def surface_tokens(text: str) -> frozenset[str]:
    return frozenset(normalize_surface(text).split())


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


def subjects_match(a: Fact, b: Fact) -> bool:
    """Decide whether two facts speak about the same subject.

    Two hard keys must be equal - a CIN or an ISIN is an identity, and two
    different ones are two different entities no matter how alike the names
    look. Without a hard key on both sides we fall back to the surface forms
    and accept one being a token subset of the other, so `Delhivery` matches
    `Delhivery Limited` without a list of corporate suffixes to maintain.
    """
    if a.subject_key and b.subject_key:
        return a.subject_key == b.subject_key

    left, right = surface_tokens(a.subject_surface), surface_tokens(b.subject_surface)
    if not left or not right:
        return False
    return left <= right or right <= left


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


def build_qualifiers(
    pairs, fy_end_month: int | None = None
) -> dict[str, Qualifier]:
    """Turn the model's qualifier list into a bag, resolving period labels.

    Parsing happens here so that a period is compared as an interval by the
    decision table. If a label does not resolve, the qualifier survives as a
    string and is compared as one; an unparsed label must never silently
    become a missing qualifier.
    """
    bag: dict[str, Qualifier] = {}
    for item in pairs:
        key = canonical_predicate(_field(item, "key", "") or "")
        value = _field(item, "value", "") or ""
        provenance = _field(item, "provenance", "stated") or "stated"
        if not key or not str(value).strip():
            continue

        period = None
        if key in PERIOD_KEYS:
            period = parse_period(str(value), fy_end_month)

        bag[key] = Qualifier(
            key=key, value=str(value).strip(), provenance=provenance, period=period
        )
    return bag


def materialize(
    out,
    doc_id: str,
    alignment,
    page: int,
    fy_end_month: int | None = None,
    default_scale: str | None = None,
    default_unit: str | None = None,
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
        qualifiers=build_qualifiers(out.qualifiers, fy_end_month),
        evidence=evidence,
        confidence=out.confidence,
        flags=flags,
    )
