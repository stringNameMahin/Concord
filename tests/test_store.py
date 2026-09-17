"""The ledger must return exactly what it was given.

A store that quietly changes a fact is worse than no store: the verdicts stay
plausible and stop being true. So the test that matters here is not "does a row
come back" but "does the engine reach the same conclusions after a round-trip".
"""

from pathlib import Path

import pytest
from factories import make_fact

from concord.compare.engine import compare
from concord.store import repo
from concord.store.db import connect

ROOT = Path(__file__).resolve().parent.parent


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


# --- relations: the judged flag, and a full run replacing what it dropped ---

def relation(a, b, **kwargs):
    from concord.compare.engine import Relation

    fields = {
        "verdict": "reconciled_by_context",
        "rule_fired": "discriminating_qualifier_differs",
        "explanation": "the period differs",
        "qualifier_key": "period",
        "blocked_by": ["semantic"],
    }
    fields.update(kwargs)
    return Relation(fact_a=a, fact_b=b, **fields)


def test_a_relation_round_trips_with_the_record_of_having_been_judged(conn):
    """`judged` is what stops a whole-corpus run re-paying for settled pairs,
    so it has to survive the ledger, not just the process that set it."""
    store(conn, [make_fact(fact_id="f_1"), make_fact(fact_id="f_2", span=900)])
    repo.save_relations(conn, [relation("f_1", "f_2", judged=True, decided_by="llm")])

    back = repo.load_relations(conn)[("f_1", "f_2")]
    assert back.judged is True
    assert back.decided_by == "llm"
    assert back.qualifier_key == "period"
    assert back.blocked_by == ["semantic"]


def test_an_unjudged_relation_reads_back_unjudged(conn):
    store(conn, [make_fact(fact_id="f_1"), make_fact(fact_id="f_2", span=900)])
    repo.save_relations(conn, [relation("f_1", "f_2")])
    assert repo.load_relations(conn)[("f_1", "f_2")].judged is False


def test_a_whole_corpus_run_removes_a_pair_it_no_longer_keeps(conn):
    """A pair that has become `unrelated` is dropped by `compare`, so a run
    that judged everything must take its stored row with it. Otherwise the UI
    shows a verdict the current code does not produce."""
    store(
        conn,
        [
            make_fact(fact_id="f_1"),
            make_fact(fact_id="f_2", span=900),
            make_fact(fact_id="f_3", span=1900),
        ],
    )
    repo.save_relations(conn, [relation("f_1", "f_2"), relation("f_1", "f_3")])
    assert len(repo.load_relations(conn)) == 2

    repo.save_relations(conn, [relation("f_1", "f_3")], replace=True)
    assert set(repo.load_relations(conn)) == {("f_1", "f_3")}


def test_an_incremental_run_leaves_the_rest_of_the_ledger_alone(conn):
    """The default must stay additive: a run that only looked at part of the
    space cannot be allowed to delete the part it never judged."""
    store(conn, [make_fact(fact_id="f_1"), make_fact(fact_id="f_2", span=900),
                 make_fact(fact_id="f_3", span=1900)])
    repo.save_relations(conn, [relation("f_1", "f_2")])
    repo.save_relations(conn, [relation("f_1", "f_3")])
    assert len(repo.load_relations(conn)) == 2


# --- embeddings: computed once per fact, not once per ingest ---------------

def test_embeddings_round_trip_for_the_model_that_made_them(conn):
    import numpy as np

    store(conn, [make_fact(fact_id="f_1"), make_fact(fact_id="f_2", span=900)])
    vectors = {
        "f_1": np.array([0.1, 0.2, 0.3], dtype="float32"),
        "f_2": np.array([0.4, 0.5, 0.6], dtype="float32"),
    }
    assert repo.save_embeddings(conn, vectors, model="test-model") == 2

    back = repo.load_embeddings(conn, model="test-model")
    assert set(back) == {"f_1", "f_2"}
    assert np.allclose(back["f_1"], vectors["f_1"])


def test_another_models_vectors_are_not_served_to_this_one(conn):
    """Two encoders' vectors are not comparable even at the same width, and a
    stale one would distort every neighbour list without failing anywhere."""
    import numpy as np

    store(conn, [make_fact(fact_id="f_1")])
    repo.save_embeddings(conn, {"f_1": np.zeros(3, dtype="float32")}, model="old-model")
    assert repo.load_embeddings(conn, model="new-model") == {}
    assert set(repo.load_embeddings(conn, model="old-model")) == {"f_1"}


def test_re_extracting_a_document_drops_its_stale_vectors(conn):
    """`save_facts` replaces the document's rows, so the embedding goes with
    the text it was made from rather than outliving it."""
    import numpy as np

    store(conn, [make_fact(fact_id="f_1")])
    repo.save_embeddings(conn, {"f_1": np.zeros(3, dtype="float32")}, model="m")
    repo.save_facts(conn, [make_fact(fact_id="f_1")], "d1")
    assert repo.load_embeddings(conn, model="m") == {}


# --- a quarantined row keeps its id across processes ------------------------

def test_a_quarantine_id_does_not_move_between_processes():
    """F15, as its own reproduction rather than as a restatement of the code.

    The id was `abs(hash(quote))`, and Python randomises string hashing per
    interpreter, so the same unlocated quote got a different id in every run:
    three processes, three ids. `save_quarantine` writes `INSERT OR REPLACE`
    as though the id were stable, and `/evidence/{id}` hands a reviewer a link
    that breaks on the next ingest.
    """
    import os
    import subprocess
    import sys

    script = (
        "from concord.store.repo import quarantine_id;"
        "print(quarantine_id('d1', 3, 'a quote that never located'))"
    )
    ids = set()
    for seed in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT)}
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True
        )
        ids.add(out.stdout.strip())
    assert len(ids) == 1
    assert ids.pop().startswith("q_d1_3_")


def test_a_quarantined_row_keeps_its_id_when_the_document_is_re_ingested(conn):
    """The link in the quarantine bin has to survive the next upload."""
    run = _run_with_quarantine("a quote that never located")
    store(conn, [make_fact(fact_id="f_1")])
    repo.save_quarantine(conn, run, "d1")
    first = [row["fact_id"] for row in conn.execute(
        "SELECT fact_id FROM facts WHERE align_status = 'unlocated'"
    )]

    repo.save_quarantine(conn, run, "d1")
    again = [row["fact_id"] for row in conn.execute(
        "SELECT fact_id FROM facts WHERE align_status = 'unlocated'"
    )]
    assert first == again and len(again) == 1


def _run_with_quarantine(quote):
    from types import SimpleNamespace

    fact = SimpleNamespace(
        claim_text="a claim",
        subject_surface="Acme",
        predicate="revenue",
        value=SimpleNamespace(raw="1", kind="quantity"),
        quote=quote,
        confidence=0.5,
    )
    chunk = SimpleNamespace(index=3, pages=[1])
    return SimpleNamespace(quarantined=[SimpleNamespace(fact=fact, chunk=chunk)])
