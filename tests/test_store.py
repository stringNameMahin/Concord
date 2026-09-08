"""The ledger must return exactly what it was given.

A store that quietly changes a fact is worse than no store: the verdicts stay
plausible and stop being true. So the test that matters here is not "does a row
come back" but "does the engine reach the same conclusions after a round-trip".
"""

import pytest
from factories import make_fact

from concord.compare.engine import compare
from concord.store import repo
from concord.store.db import connect


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr("concord.config.WORK_DIR", tmp_path / "work")
    connection = connect(tmp_path / "ledger.sqlite")
    yield connection
    connection.close()


def store(conn, facts, doc_id="d1"):
    conn.execute(
        """INSERT INTO documents (doc_id, filename, sha256, status, ingested_at, text_path)
           VALUES (?, ?, ?, 'ingested', '2026-01-01', '')""",
        (doc_id, f"{doc_id}.pdf", doc_id),
    )
    repo.save_facts(conn, facts, doc_id)
    return repo.load_facts(conn, doc_id)


def test_a_quantity_survives_the_round_trip(conn):
    fact = make_fact(raw="Rs 8,142 Cr", period="FY 2023-24", consolidation="consolidated")
    back = store(conn, [fact])[0]

    assert back.fact_id == fact.fact_id
    assert back.value_raw == "Rs 8,142 Cr"
    assert back.quantity.normalized == fact.quantity.normalized
    assert back.quantity.interval == fact.quantity.interval
    assert back.quantity.unit == "INR"
    assert back.comparison_key == fact.comparison_key


def test_a_text_value_survives_the_round_trip(conn):
    """Regression: text facts came back with an empty value, so any two of them
    compared as equal and two real contradictions read as corroborations."""
    fact = make_fact(
        raw="L63090DL2011PLC221234", kind="text", predicate="corporate_identity_number"
    )
    back = store(conn, [fact])[0]

    assert back.value_raw == "L63090DL2011PLC221234"
    assert back.value_kind == "text"
    assert back.quantity is None


def test_qualifiers_keep_their_provenance_and_interval(conn):
    fact = make_fact(period="FY 2023-24", consolidation=("consolidated", "inherited"))
    back = store(conn, [fact])[0]

    assert back.qualifier("consolidation").provenance == "inherited"
    assert back.qualifier("period").period.start.isoformat() == "2023-04-01"
    assert back.qualifier_keys() == fact.qualifier_keys()


def test_evidence_offsets_survive_exactly(conn):
    fact = make_fact(span=48120)
    back = store(conn, [fact])[0]
    assert back.evidence.char_start == 48120
    assert back.evidence.quote == fact.evidence.quote
    assert back.evidence.align_status == "exact"


def test_the_engine_reaches_the_same_verdicts_after_a_round_trip(conn):
    """The guard that would have caught the value-persistence bug."""
    facts = [
        make_fact(fact_id="f_1", raw="8,142 Cr", doc="d1", period="FY 2023-24"),
        make_fact(fact_id="f_2", raw="81,415.38 mn", doc="d1", period="FY 2023-24"),
        make_fact(fact_id="f_3", raw="7,224 Cr", doc="d1", period="FY 2023-24"),
        make_fact(fact_id="f_4", raw="L6309", kind="text", predicate="cin", doc="d1"),
        make_fact(fact_id="f_5", raw="U6309", kind="text", predicate="cin", doc="d1"),
    ]
    before = compare(facts)
    after = compare(store(conn, facts))

    assert dict(before.verdicts) == dict(after.verdicts)
    assert [(r.fact_a, r.fact_b, r.verdict) for r in before.relations] == [
        (r.fact_a, r.fact_b, r.verdict) for r in after.relations
    ]


def test_reingesting_a_document_replaces_rather_than_duplicates(conn):
    facts = [make_fact(fact_id="f_1", span=100), make_fact(fact_id="f_2", span=200)]
    store(conn, facts)
    again = store_again = repo.save_facts(conn, facts, "d1")
    del again, store_again

    assert len(repo.load_facts(conn, "d1")) == 2
    assert repo.counts(conn)["facts"] == 2


def test_counts_separate_grounded_from_quarantined(conn):
    store(conn, [make_fact(fact_id="f_1")])
    assert repo.counts(conn) == {
        "documents": 1,
        "facts": 1,
        "quarantined": 0,
        "relations": 0,
    }
