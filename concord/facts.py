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

from concord.normalize.numbers import (
    NotANumber,
    Quantity,
    is_scale_word,
    parse_quantity,
    scale_after_figure,
    truncates,
)
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


# --- qualifier dimensions ---------------------------------------------------
#
# The bag is an open vocabulary and always will be, but an open vocabulary
# means the same condition arrives under several names. `period`, `as_at`,
# `as_of`, `time_period`, `financial_year` and `for_the_year_ended` all answer
# "when does this figure hold?", and the decision table used to diff the bags
# by literal key name - so two facts that both stated the time, spelled
# differently, each read as "states a condition the other does not" and the
# missing-qualifier guard abstained. Measured on the shipped ledger: 18 of 46
# `missing_qualifier_guard` relations, whose explanations then said one side
# did not state a period two words away from where it did.
#
# A dimension is a set of key spellings that answer one question. Everything
# not named here is its own dimension, which is exactly today's behaviour, so
# the coined keys the corpus invents (`service`, `auditor`, `category`) are
# untouched.
#
# Only one dimension is declared, because only one is measurably needed. This
# is a naming convention - how a document spells "when" - and not knowledge of
# any subject area, the same commitment `LEGAL_FORMS` and `CONSOLIDATION_WORDS`
# already make.
REPORTING_TIME = PERIOD_KEYS + ("as_on", "year", "quarter", "month", "date", "time")

# ...minus the moments that belong to an event inside the claim rather than to
# the measurement. A contract's effective date and the period a figure covers
# are two conditions, not two spellings of one, and merging them would be the
# false merge this whole change is trying not to make. Matched as whole words,
# so `for_the_year_ended` keeps its place in the dimension while `end_date`
# leaves it.
EVENT_TIME = (
    "effective",
    "acquisition",
    "meeting",
    "target",
    "start",
    "end",
    "maturity",
    "commencement",
    "expiry",
    "vesting",
    "grant",
    "inception",
)

TIME_DIMENSION = "period"


def dimension_of(key: str) -> str:
    """Which question this qualifier key answers.

    Returns the dimension's name, or the key itself when it names a dimension
    of its own - which is the answer for every key the corpus coins.
    """
    if key_matches(key, REPORTING_TIME) and not key_matches(key, EVENT_TIME):
        return TIME_DIMENSION
    return key


def canonical_predicate(text: str) -> str:
    """Fold a predicate to a comparable form without interpreting it.

    Case and punctuation only. Anything that needs to know that
    `revenue_from_services` and `revenue_from_operations` mean the same thing
    is the predicate registry's job, not a string rule's.
    """
    return _WORD.sub("_", (text or "").strip().lower()).strip("_")


# The two bases a filer states twice over the same year. Named here as well as
# in `parse.structure` because they arrive by two routes: from the page, where
# the structure layer promotes them, and from inside the predicate, where the
# model folds them into the name.
CONSOLIDATION_WORDS = ("consolidated", "standalone")


def split_consolidation(predicate: str) -> tuple[str, str | None]:
    """A predicate as (measure, basis), where the model folded the basis in.

    Asked for the consolidated revenue line the model does not always coin
    `revenue_from_operations` and qualify it; sometimes it coins
    `consolidated_revenue_from_operations`. That puts the basis where it
    fragments the comparison key instead of conditioning it, and the two
    figures then never meet at all:

        delhivery limited|standalone_revenue_from_operations    74,540.82
        delhivery limited|consolidated_revenue_from_operations  81,415.38
        delhivery limited|revenue_from_operations               81,355.38

    Three keys, so every pair is `unrelated` and drops out before any guard
    runs, with no trace. Ten facts in the shipped ledger carry a basis-bearing
    predicate and five of the eight such predicates have a de-basis twin
    already in the vocabulary. Lifting the word into the `consolidation`
    qualifier puts those figures back on one key, where the decision table can
    say what it is for.

    Only these two words, and only as the leading token. This is the same
    commitment `parse.structure` already makes and not a licence to strip
    modifiers generally: `real` against nominal growth is a genuine predicate
    distinction with no qualifier key waiting for it, and folding that one out
    would merge two different measures.
    """
    canonical = canonical_predicate(predicate)
    head, _, rest = canonical.partition("_")
    if head in CONSOLIDATION_WORDS and rest:
        return rest, head
    return canonical, None


# Leading words that belong to a name's grammar rather than to the name. Both
# lists strip only from the front and only one word, which is strictly
# narrower than the trailing-legal-form rule in `same_entity` and cannot bridge
# two names whose content words differ. Measured over the 223 distinct subject
# surfaces in the shipped ledger, this joins `the group` to `group` and
# `mr sahil barua` to `sahil barua`, and nothing else.
ARTICLES = frozenset({"the", "a", "an"})
HONORIFICS = frozenset(
    {"mr", "mrs", "ms", "miss", "dr", "prof", "shri", "smt", "sri", "sh", "mx"}
)


def normalize_surface(text: str) -> str:
    """Normalise a subject surface form for use as a fallback identity."""
    lowered = (text or "").strip().lower()
    lowered = re.sub("['\u2019]s\\b", "", lowered)
    words = _WORD.sub(" ", lowered).split()
    if len(words) > 1 and (words[0] in ARTICLES or words[0] in HONORIFICS):
        words = words[1:]
    return " ".join(words)


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


# The longest legal form in the table, in words, so the stripper knows how far
# back to look. Derived rather than restated, so adding a form cannot make the
# two disagree.
_LONGEST_LEGAL_FORM = max(len(form.split()) for form in LEGAL_FORMS)


def surface_identities(surface: str) -> frozenset[str]:
    """Every normalised name `same_entity` would accept this surface as.

    `same_entity` is a test over two names; blocking needs the same rule as a
    *group key*, so that two facts about one entity land in one bucket without
    comparing every surface against every other. A surface answers to its own
    normalised form and to that form with one trailing legal form removed - the
    only latitude `same_entity` allows - so `Delhivery Limited` answers to both
    `delhivery limited` and `delhivery`, and meets a fact whose surface is just
    `Delhivery` in the second.

    Every matching trailing run is stripped, not only the longest, because
    `LEGAL_FORMS` holds `pvt limited` and `limited` both and `same_entity`
    accepts either reading. Nothing is stripped that would leave the name
    empty, so a subject that is *only* a legal form keeps it.
    """
    normalised = normalize_surface(surface)
    if not normalised:
        return frozenset()

    identities = {normalised}
    words = normalised.split()
    for size in range(1, min(_LONGEST_LEGAL_FORM, len(words) - 1) + 1):
        if " ".join(words[-size:]) in LEGAL_FORMS:
            identities.add(" ".join(words[:-size]))
    return frozenset(identities)


def subject_identities(fact: "Fact") -> frozenset[str]:
    """The identities a fact answers to, for grouping - never for deciding.

    A hard key is an identity, so a fact that carries one answers to it. It
    answers to its surface forms as well, because `subjects_match` falls back
    to the surface whenever *either* side lacks a key - so a fact with a CIN
    and a fact without one, both naming the same company, have to meet
    somewhere. They never used to: `comparison_key` is `subject_key or
    surface`, so the two sat in different buckets and only the semantic block
    could bring them together.

    This is deliberately a superset of `subjects_match`. Grouping decides what
    is *compared*; `subjects_match` still decides what is *related*, and throws
    out the extra pairs this admits.
    """
    identities = set(surface_identities(fact.subject_surface))
    if fact.subject_key:
        identities.add(fact.subject_key)
    return frozenset(identities)


# Words a document uses to mean "the entity this document is about". They name
# no entity on their own, so two documents that both use one share a comparison
# key while meaning two different organisations - which is a real collision
# waiting for the third corporate report, not a hypothetical one: `Company`,
# `Group` and `the Group` are 50 of the 773 facts in the shipped ledger and all
# from one document, which is luck rather than design.
#
# Deliberately short. `Bank` is absent although the RBI report uses it that
# way, because a bank is also a thing a document can be about from the outside.
ANAPHORS = frozenset(
    {"company", "group", "issuer", "firm", "entity", "corporation",
     "organisation", "organization"}
)


def resolve_anaphora(facts: list[Fact]) -> int:
    """Point `the Company` at the entity its document names elsewhere.

    Prompt rule 8 asks the extractor to name the subject even where the
    document states it only in a heading or a page header. It complied for one
    filer (113 of 136 facts say the full name) and not for another, which is
    the same shape as every other finding here: a prompt rule with no
    deterministic backstop.

    The backstop is the document's own dominant subject - the most common
    surface that is not itself an anaphor - and it is applied only when that is
    unambiguous. A tie means the document has not settled who it is about, and
    inventing a winner would attach facts to the wrong entity, which is worse
    than leaving them orphaned. Rewritten facts carry a flag, because a subject
    the layer supplied and one the sentence stated are different evidence.

    The identifier is untouched: a hard key is an identity and nothing here
    invents one.
    """
    counts: dict[str, int] = {}
    surfaces: dict[str, str] = {}
    for fact in facts:
        surface = normalize_surface(fact.subject_surface)
        if not surface or surface in ANAPHORS:
            continue
        if "subject_names_the_measurement" in fact.flags:
            continue
        counts[surface] = counts.get(surface, 0) + 1
        surfaces.setdefault(surface, fact.subject_surface)

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if not ranked or ranked[0][1] < 2:
        return 0
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return 0

    dominant = surfaces[ranked[0][0]]
    resolved = 0
    for fact in facts:
        if normalize_surface(fact.subject_surface) not in ANAPHORS:
            continue
        fact.subject_surface = dominant
        fact.flags.append("subject_resolved_from_anaphor")
        resolved += 1
    return resolved


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

    The magnitude is recovered from the document rather than trusted to the
    model. A model handed a sentence reading "81,415.38 million" routinely
    returns `raw="81,415.38"` with `scale` left null, and the resulting record
    is wrong by a factor of a million while looking entirely ordinary - the
    quote is right, the value shown is right, only `normalized` is wrong, and
    `normalized` is what every interval comparison uses. Measured over the
    seven-document ledger before this guard existed: 133 of 728 quantity facts,
    touching 36 of 171 relations. So the order is

        the figure itself  ->  what the model said  ->  the located bytes
                           ->  the sentence         ->  the caller's context

    and only the first two are the model's. Reading the scale back out of the
    span the aligner already located is the same deterministic guard the quote
    gets from the aligner and the subject gets from
    `subject_names_the_measurement`; the magnitude was the last load-bearing
    field without one.
    """
    quantity = None
    flags: list[str] = []
    kind = out.value.kind

    if kind == "quantity":
        recovered = scale_after_figure(out.value.raw, alignment.text) or scale_after_figure(
            out.value.raw, out.claim_text
        )
        # Both of the model's units fields are free text and it fills them the
        # wrong way round often enough to matter, so neither is trusted by
        # position. A `scale` that names no known magnitude is not a scale and
        # must not shadow what the document itself says; a `unit` that names a
        # magnitude is a scale and must not become a bogus unit that then
        # refuses every comparison.
        stated = next(
            (s for s in (out.value.scale, out.value.unit) if is_scale_word(s)), None
        )
        # One exception to the model winning: when what it said is only the
        # first half of what the document wrote. `lakh` and `lakh crore` are
        # different magnitudes by a factor of ten million, and a model that
        # returns `scale="lakh"` for "Rs 1.1 lakh crore" has not disagreed with
        # the page, it has stopped reading early. Preferring the longer phrase
        # the document actually contains is not second-guessing a complete
        # answer - `million` against bytes reading `Cr` is still the model's
        # call, because neither extends the other.
        if stated and truncates(stated, recovered):
            stated = recovered
        scale = next(
            (s for s in (stated, recovered, default_scale) if is_scale_word(s)), None
        )
        unit = next(
            (u for u in (out.value.unit, default_unit) if u and not is_scale_word(u)), None
        )
        if scale is not None and scale == recovered and stated is None:
            flags.append("scale_recovered_from_source")
        try:
            quantity = parse_quantity(
                out.value.raw, default_scale=scale, default_unit=unit
            )
        except NotANumber:
            flags.append("unparsed_quantity")

    # A basis the model folded into the predicate is a condition, not part of
    # the measure's name. See `split_consolidation`.
    canonical, basis = split_consolidation(out.predicate)
    qualifiers = build_qualifiers(out.qualifiers, fy_end_month, inherited)
    if basis:
        stated_basis = qualifiers.get("consolidation")
        # The name is nearer the claim than the page frame is, so it beats an
        # inherited basis - and loses to one the extractor stated outright.
        if stated_basis is None or stated_basis.provenance != "stated":
            qualifiers["consolidation"] = Qualifier(
                key="consolidation", value=basis, provenance="stated"
            )

    if subject_names_the_measurement(out.subject_surface, canonical):
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
        fact_id=make_fact_id(doc_id, alignment.start, alignment.end, canonical),
        doc_id=doc_id,
        claim_text=out.claim_text,
        subject_surface=out.subject_surface,
        subject_key=out.subject_key,
        subject_type=out.subject_type,
        # The model's own spelling is kept for display - a reader comparing the
        # record against the quote should see what was read - while the
        # canonical form, which is half the comparison key, carries the basis
        # in the qualifier bag instead of in the name.
        predicate=out.predicate,
        predicate_canonical=canonical,
        value_kind=kind,
        value_raw=out.value.raw,
        quantity=quantity,
        qualifiers=qualifiers,
        evidence=evidence,
        confidence=out.confidence,
        flags=flags,
    )
