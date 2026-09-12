"""Properties the shipped ledger must satisfy, checked against the real rows.

Every other test in this suite is synthetic, and deliberately so - see
`tests/factories.py`. These are the exception: they assert *properties* over
whatever the ledger happens to hold, never values from it. No corpus figure,
predicate, document name or count appears here, so the file stays true if the
ledger is rebuilt, extended or replaced.

They exist because the defects that have cost this project most were not
reachable from synthetic fixtures. F1 (a magnitude dropped from 133 facts), F20
(a percentage stored as six billion) and F21 (`lakh crore` read as `lakh`) were
each a single wrong number in a ledger that looked entirely ordinary, and each
would have been caught on the first run by one of the checks below. Line
coverage was 94% throughout.

The ledger is committed, so these run with no key and no network. If it is
missing the file skips rather than fails: a contributor who has cleared
`data/` has not broken anything.
"""

import json

import pytest

from concord import config
from concord.normalize.numbers import (
    SCALES,
    WHOLE_SCALE_WORD,
    _scale_key,
    scale_after_figure,
)
from concord.store.db import connect

pytestmark = pytest.mark.skipif(
    not config.DB_PATH.exists(), reason="no ledger in data/db to check"
)


@pytest.fixture(scope="module")
def ledger():
    conn = connect(config.DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module")
def quantities(ledger):
    """Every quantity fact, with the bytes it was located in."""
    rows = []
    for row in ledger.execute(
        """SELECT predicate_canonical, claim_text, value_json,
                  json_extract(evidence_json, '$.quote') AS quote
             FROM facts WHERE value_kind = 'quantity'"""
    ):
        value = json.loads(row["value_json"])
        if value.get("quantity"):
            rows.append((row["predicate_canonical"], value["raw"],
                         value["quantity"], row["quote"], row["claim_text"]))
    return rows


def test_the_ledger_is_not_empty(ledger):
    """Guards the rest of the file: every check below is vacuous on no rows."""
    assert ledger.execute("SELECT COUNT(*) FROM facts").fetchone()[0] > 0
    assert ledger.execute("SELECT COUNT(*) FROM relations").fetchone()[0] > 0


def test_every_grounded_fact_still_cuts_its_quote_from_source(ledger):
    """The headline claim, asserted rather than described.

    A stored quote that no longer matches `text[char_start:char_end]` means the
    evidence link is broken, which is the one failure this system cannot survive.
    """
    texts = {}
    for row in ledger.execute("SELECT doc_id, text_path FROM documents"):
        with open(row["text_path"], encoding="utf-8") as handle:
            texts[row["doc_id"]] = handle.read()

    broken = []
    for row in ledger.execute(
        "SELECT fact_id, doc_id, evidence_json FROM facts WHERE align_status != 'unlocated'"
    ):
        span = json.loads(row["evidence_json"])
        text = texts.get(row["doc_id"])
        if text is None:
            broken.append((row["fact_id"], "no source text on disk"))
        elif text[span["char_start"]:span["char_end"]] != span["quote"]:
            broken.append((row["fact_id"], "offsets no longer cut the quote"))
    assert broken == []


def test_a_scale_the_document_wrote_is_the_scale_that_was_applied(quantities):
    """F1, F21 and the compound-scale family, as one property.

    When the located bytes put a scale word directly behind the figure, the
    stored magnitude must be that scale's. Spelling may differ - `crore` and
    `crores` are one magnitude - so the comparison is on the factor, not the
    word. Everything this has caught was silent: the quote right, the displayed
    value right, and only `normalized` wrong by a power of ten.
    """
    wrong = []
    for predicate, raw, quantity, quote, claim in quantities:
        written = scale_after_figure(raw, quote or "") or scale_after_figure(raw, claim or "")
        if not written:
            continue
        applied = SCALES.get(_scale_key(quantity.get("scale") or ""), 1.0)
        if applied != SCALES[_scale_key(written)]:
            wrong.append((predicate, raw, written, quantity.get("scale")))
    assert wrong == []


def test_no_percentage_carries_a_magnitude(quantities):
    """F20. A ratio has no scale, so a scaled percentage is always a bug - and
    it reads as one on sight: six per cent with an interval half a billion wide.
    """
    scaled = [
        (predicate, raw, quantity["scale"])
        for predicate, raw, quantity, _, _ in quantities
        if quantity.get("is_percent") and quantity.get("scale_factor", 1.0) != 1.0
    ]
    assert scaled == []


def test_no_currency_unit_smuggles_a_magnitude_into_its_identity(quantities):
    """F22. The magnitude belongs on `scale`, where it is applied once.

    Left in the unit as well it becomes part of the identity, and
    `compare_values` then refuses `INR` against `INR crore` - two spellings of
    one currency, and on the shipped ledger the same figure against itself.
    Units that are not currencies may keep a magnitude that is part of their
    meaning (`per million person-hours`), so this binds on currencies only.
    """
    offenders = []
    for predicate, raw, quantity, _, _ in quantities:
        unit = quantity.get("unit")
        if not unit or not WHOLE_SCALE_WORD.search(unit):
            continue
        # A currency is what `normalize_currency` already resolved to a code:
        # short, all upper case, nothing else left once the magnitude is out.
        stripped = WHOLE_SCALE_WORD.sub(" ", unit).strip()
        if stripped.isupper() and stripped.isalpha() and len(stripped) <= 3:
            offenders.append((predicate, raw, unit))
    assert offenders == []


def test_every_relation_points_at_facts_that_exist(ledger):
    """`save_facts` deletes a document's rows before rewriting them, and the
    cascade is what keeps relations from outliving their facts. A dangling
    `fact_a` renders as an empty row in the UI."""
    dangling = ledger.execute(
        """SELECT COUNT(*) FROM relations r
             WHERE NOT EXISTS (SELECT 1 FROM facts f WHERE f.fact_id = r.fact_a)
                OR NOT EXISTS (SELECT 1 FROM facts f WHERE f.fact_id = r.fact_b)"""
    ).fetchone()[0]
    assert dangling == 0


def test_no_relation_is_stored_against_a_quarantined_fact(ledger):
    """Quarantined facts never enter adjudication, so a relation touching one
    means the ungrounded path leaked into the comparison layer."""
    leaked = ledger.execute(
        """SELECT COUNT(*) FROM relations r JOIN facts f
             ON f.fact_id IN (r.fact_a, r.fact_b)
            WHERE f.align_status = 'unlocated'"""
    ).fetchone()[0]
    assert leaked == 0


def test_a_contradiction_names_no_qualifier_the_facts_do_not_carry(ledger):
    """The named-qualifier guard, asserted over the stored rows rather than the
    adjudicator's counters. A `qualifier_key` on neither fact is a fabrication
    that survived into the ledger."""
    fabricated = []
    for row in ledger.execute(
        """SELECT r.relation_id, r.qualifier_key, a.qualifiers_json AS qa,
                  b.qualifiers_json AS qb
             FROM relations r
             JOIN facts a ON a.fact_id = r.fact_a
             JOIN facts b ON b.fact_id = r.fact_b
            WHERE r.qualifier_key IS NOT NULL"""
    ):
        keys = set(json.loads(row["qa"] or "{}")) | set(json.loads(row["qb"] or "{}"))
        if row["qualifier_key"] not in keys:
            fabricated.append((row["relation_id"], row["qualifier_key"], sorted(keys)))
    assert fabricated == []
