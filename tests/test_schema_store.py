"""Persisting the emergent schema, and the incremental path it enables.

The registry has to survive a restart with its embeddings intact, or every
ingest would re-learn the vocabulary from scratch and the timeline would be a
fiction.
"""

import numpy as np
import pytest
from factories import make_fact
from test_registry import Judge, WordOverlap

from concord.registry import PredicateRegistry
from concord.store import repo
from concord.store.db import connect


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr("concord.config.WORK_DIR", tmp_path / "work")
    connection = connect(tmp_path / "ledger.sqlite")
    yield connection
    connection.close()


def test_the_registry_survives_a_restart_with_its_embeddings(conn):
    first = PredicateRegistry(encoder=WordOverlap(), client=Judge(same=False))
    first.observe("revenue_from_services", "d1")
    first.observe("board_meeting_attendance", "d1")
    repo.save_registry(conn, first)

    second = repo.load_registry(conn, encoder=WordOverlap(), client=Judge(same=False))
    assert len(second) == 2
    stored = second.entries["revenue_from_services"]
    assert stored.first_seen_doc == "d1"
    assert isinstance(stored.embedding, np.ndarray)
    assert stored.embedding.dtype == np.float32


def test_a_reloaded_registry_recognises_what_it_already_knew(conn):
    """Without this, every ingest re-learns the vocabulary and re-asks the LLM."""
    first = PredicateRegistry(encoder=WordOverlap(), client=Judge(same=True), threshold=0.5)
    first.observe("revenue_from_services", "d1")
    first.observe("revenue_from_operations", "d1")
    repo.save_registry(conn, first)
    assert first.calls == 1

    second = repo.load_registry(conn, encoder=WordOverlap(), client=Judge(same=True))
    assert second.observe("revenue_from_operations", "d2").event_type == "predicate_seen"
    assert second.calls == 0
    assert second.aliases() == {"revenue_from_operations": "revenue_from_services"}


def test_saving_the_registry_twice_updates_rather_than_duplicates(conn):
    registry = PredicateRegistry(encoder=WordOverlap(), client=Judge(same=False))
    registry.observe("total_assets", "d1")
    repo.save_registry(conn, registry)
    registry.observe("total_assets", "d2")
    repo.save_registry(conn, registry)

    assert len(repo.load_registry(conn)) == 1
    assert repo.load_registry(conn).entries["total_assets"].count == 2


def test_the_timeline_records_what_happened_and_when(conn):
    registry = PredicateRegistry(encoder=WordOverlap(), client=Judge(same=True), threshold=0.5)
    events = [
        registry.observe("revenue_from_services", "d1"),
        registry.observe("revenue_from_operations", "d2"),
    ]
    repo.log_schema_events(conn, events)

    timeline = repo.schema_timeline(conn)
    assert [e["event_type"] for e in timeline] == ["alias_confirmed", "predicate_registered"]
    confirmed = timeline[0]
    assert confirmed["target"] == "revenue_from_services"
    assert confirmed["decided_by"] == "llm"
    assert confirmed["similarity"] > 0.5
    assert confirmed["doc_id"] == "d2"

    assert len(repo.schema_timeline(conn, doc_id="d1")) == 1


def test_an_empty_registry_loads_as_an_empty_one(conn):
    assert len(repo.load_registry(conn)) == 0
    assert repo.schema_timeline(conn) == []


def test_a_ledger_written_before_the_new_columns_catches_up_in_place(tmp_path):
    """`CREATE TABLE IF NOT EXISTS` cannot widen a table that already exists.

    The committed ledger predates both `facts.embedding_model` and
    `relations.judged`, so without a migration every read of either column
    fails against the one file the project ships.
    """
    import sqlite3

    from concord.store.db import LATER_COLUMNS, connect, migrate

    from concord.store.db import SCHEMA

    # The previous shape, built from the current one by removing exactly the
    # columns that were added - so this stays a test of the migration rather
    # than of a hand-copied snapshot that will drift.
    previous = "\n".join(
        line
        for line in SCHEMA.read_text(encoding="utf-8").splitlines()
        if not any(f" {column} " in line for _, column, _ in LATER_COLUMNS)
    )
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.row_factory = sqlite3.Row
    old.executescript(previous)
    old.commit()
    assert "embedding_model" not in {
        row["name"] for row in old.execute("PRAGMA table_info(facts)")
    }
    assert migrate(old) == [f"{table}.{column}" for table, column, _ in LATER_COLUMNS]
    assert migrate(old) == []  # idempotent
    old.close()

    conn = connect(path)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(facts)")}
    assert "embedding_model" in columns
    conn.close()
