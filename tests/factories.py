"""Fact builders for the comparison tests.

Every value here is invented. Nothing in the test suite may encode a fact,
figure or filename from the starter corpus, because the reviewers test with
unseen PDFs and a fixture that knows the answers would prove nothing.
"""

from itertools import count

from concord.facts import Evidence, Fact, Qualifier
from concord.normalize.numbers import NotANumber, parse_quantity
from concord.normalize.periods import parse_period

_spans = count(1000, 100)

PERIOD_KEYS = ("period", "as_of", "as_at")


def qualifiers(**pairs) -> dict[str, Qualifier]:
    """`period="FY24", consolidation="consolidated"` into a qualifier bag.

    Pass a (value, provenance) tuple where the tier matters.
    """
    bag = {}
    for key, raw in pairs.items():
        value, provenance = raw if isinstance(raw, tuple) else (raw, "stated")
        period = parse_period(value) if key in PERIOD_KEYS else None
        bag[key] = Qualifier(key=key, value=value, provenance=provenance, period=period)
    return bag


def make_fact(
    raw="8,142 Cr",
    predicate="revenue_from_services",
    subject="Acme Logistics Limited",
    subject_key=None,
    doc="d1",
    kind="quantity",
    claim=None,
    scale=None,
    unit=None,
    fact_id=None,
    span=None,
    **quals,
) -> Fact:
    quantity = None
    if kind == "quantity":
        try:
            quantity = parse_quantity(raw, default_scale=scale, default_unit=unit)
        except NotANumber:
            quantity = None

    start = span if span is not None else next(_spans)
    return Fact(
        fact_id=fact_id or f"f_{start:06d}",
        doc_id=doc,
        claim_text=claim or f"{subject} {predicate.replace('_', ' ')} was {raw}.",
        subject_surface=subject,
        subject_key=subject_key,
        predicate=predicate,
        value_kind=kind,
        value_raw=raw,
        quantity=quantity,
        qualifiers=qualifiers(**quals),
        evidence=Evidence(
            doc_id=doc,
            page=1,
            char_start=start,
            char_end=start + len(raw),
            quote=raw,
        ),
    )
