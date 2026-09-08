"""Reading and writing the ledger.

Two rules shape this module.

The canonical document text is written to a file and the row keeps its path,
not its contents. Every offset in the system points into that text, so
`/evidence` and `/verify` re-slice the same bytes the aligner used. Storing the
text inline would work too; keeping it beside the database means a reviewer can
diff it, and it keeps the SQLite file small enough to commit.

Writes are idempotent. Fact ids are content-addressed, so re-ingesting a
document replaces its rows rather than duplicating them, which is what lets
Phase 7 append a fourth document without rebuilding the first three.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from sqlite3 import Connection

from concord import config
from concord.compare.engine import Relation
from concord.facts import Evidence, Fact, Qualifier
from concord.normalize.numbers import Quantity
from concord.normalize.periods import Period


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def text_path(doc_id: str) -> Path:
    return Path(config.WORK_DIR) / f"{doc_id}.txt"


def _value_json(fact: Fact) -> str:
    """Store the written value alongside the parsed one.

    A text or date fact has no `Quantity`, so persisting only the quantity
    loses the value entirely and every such fact reads back with an empty
    value - which makes any two of them compare as equal. That turned two
    genuine contradictions into corroborations on the first round-trip.
    """
    quantity = None
    if fact.quantity is not None:
        quantity = asdict(fact.quantity)
        quantity["interval"] = list(fact.quantity.interval)
    return json.dumps({"raw": fact.value_raw, "kind": fact.value_kind, "quantity": quantity})


def _qualifiers_json(qualifiers: dict[str, Qualifier]) -> str:
    return json.dumps(
        {
            key: {
                "value": q.value,
                "provenance": q.provenance,
                "period": (
                    {
                        "start": q.period.start.isoformat(),
                        "end": q.period.end.isoformat(),
                        "granularity": q.period.granularity,
                        "fiscal_basis": q.period.fiscal_basis,
                    }
                    if q.period
                    else None
                ),
            }
            for key, q in qualifiers.items()
        }
    )


def document_id_for_hash(conn: Connection, sha256: str) -> str | None:
    """The id this content already has, if the ledger has seen these bytes.

    Resolved before extraction so facts are materialised under the id they will
    be stored with. Fact ids are content-addressed from the document id, so
    deciding identity afterwards would give the same bytes two sets of ids.
    """
    row = conn.execute(
        "SELECT doc_id FROM documents WHERE sha256 = ?", (sha256,)
    ).fetchone()
    return row["doc_id"] if row else None


def save_document(conn: Connection, ingested, doc_id: str | None = None) -> str:
    """Persist the document row and write its canonical text beside it.

    A document's identity is its content hash, not its filename. Uploading the
    same bytes under a new name re-ingests the document already stored rather
    than failing on the unique hash - which is what a reviewer trying the same
    PDF twice actually means.
    """
    identifier = doc_id or ingested.doc_id
    existing = conn.execute(
        "SELECT doc_id FROM documents WHERE sha256 = ?", (ingested.doc.sha256,)
    ).fetchone()
    if existing is not None:
        identifier = existing["doc_id"]
    path = text_path(identifier)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ingested.doc.text, encoding="utf-8")

    conn.execute(
        """
        INSERT INTO documents (doc_id, filename, sha256, n_pages, parser,
                               parser_version, text_path, doc_context, status, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            filename = excluded.filename,
            sha256 = excluded.sha256,
            n_pages = excluded.n_pages,
            text_path = excluded.text_path,
            doc_context = excluded.doc_context,
            status = excluded.status,
            ingested_at = excluded.ingested_at
        """,
        (
            identifier,
            ingested.doc.filename,
            ingested.doc.sha256,
            ingested.doc.n_pages,
            ingested.doc.parser,
            ingested.doc.parser_version,
            str(path),
            json.dumps({"fy_end_month": ingested.fy_end_month}),
            "ingested",
            _now(),
        ),
    )
    return identifier


def save_facts(conn: Connection, facts: list[Fact], doc_id: str) -> int:
    """Replace this document's facts. Ids are content-addressed, so a re-ingest
    of unchanged text produces the same rows."""
    conn.execute("DELETE FROM facts WHERE doc_id = ?", (doc_id,))
    conn.executemany(
        """
        INSERT INTO facts (fact_id, doc_id, claim_text, subject_surface, subject_key,
                           subject_type, predicate, predicate_canonical, comparison_key,
                           value_json, value_kind, value_normalized, qualifiers_json,
                           evidence_json, align_status, page, confidence, flags_json,
                           created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                fact.fact_id,
                fact.doc_id,
                fact.claim_text,
                fact.subject_surface,
                fact.subject_key,
                fact.subject_type,
                fact.predicate,
                fact.predicate_canonical,
                fact.comparison_key,
                _value_json(fact),
                fact.value_kind,
                fact.normalized,
                _qualifiers_json(fact.qualifiers),
                json.dumps(asdict(fact.evidence)),
                fact.evidence.align_status,
                fact.evidence.page,
                fact.confidence,
                json.dumps(fact.flags),
                _now(),
            )
            for fact in facts
        ],
    )
    return len(facts)


def save_quarantine(conn: Connection, run, doc_id: str) -> int:
    """Store facts whose quote could not be located.

    They are kept deliberately. A grounding guarantee nobody can see the
    exceptions to is a claim rather than a guarantee, so the rejected bin is
    part of the interface and these rows feed it.
    """
    rows = []
    for record in run.quarantined:
        fact = record.fact
        rows.append(
            (
                f"q_{doc_id}_{record.chunk.index}_{abs(hash(fact.quote)) % 10**10}",
                doc_id,
                fact.claim_text,
                fact.subject_surface,
                None,
                None,
                fact.predicate,
                None,
                None,
                json.dumps({"raw": fact.value.raw, "kind": fact.value.kind, "quantity": None}),
                fact.value.kind,
                None,
                "{}",
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "page": record.chunk.pages[0] if record.chunk.pages else 0,
                        "char_start": -1,
                        "char_end": -1,
                        "quote": fact.quote,
                        "align_status": "unlocated",
                    }
                ),
                "unlocated",
                record.chunk.pages[0] if record.chunk.pages else 0,
                fact.confidence,
                json.dumps(["unlocated"]),
                _now(),
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO facts (fact_id, doc_id, claim_text, subject_surface,
            subject_key, subject_type, predicate, predicate_canonical, comparison_key,
            value_json, value_kind, value_normalized, qualifiers_json, evidence_json,
            align_status, page, confidence, flags_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def save_relations(conn: Connection, relations: list[Relation]) -> int:
    conn.executemany(
        """
        INSERT INTO relations (relation_id, fact_a, fact_b, verdict, rule_fired,
                               qualifier_key, explanation, decided_by, self_consistent,
                               cross_document, blocked_by, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fact_a, fact_b) DO UPDATE SET
            verdict = excluded.verdict,
            rule_fired = excluded.rule_fired,
            qualifier_key = excluded.qualifier_key,
            explanation = excluded.explanation,
            decided_by = excluded.decided_by,
            self_consistent = excluded.self_consistent,
            blocked_by = excluded.blocked_by
        """,
        [
            (
                relation.relation_id,
                relation.fact_a,
                relation.fact_b,
                relation.verdict,
                relation.rule_fired,
                relation.qualifier_key,
                relation.explanation,
                relation.decided_by,
                None if relation.self_consistent is None else int(relation.self_consistent),
                int(relation.cross_document),
                json.dumps(relation.blocked_by),
                _now(),
            )
            for relation in relations
        ],
    )
    return len(relations)


def load_facts(conn: Connection, doc_id: str | None = None) -> list[Fact]:
    """Rebuild `Fact` objects from rows, so comparison can run over the ledger.

    Phase 7 blocks a new document's facts against everything already stored,
    which means the stored form has to round-trip back into the same objects
    the engine works on.
    """
    sql = "SELECT * FROM facts WHERE align_status != 'unlocated'"
    params: tuple = ()
    if doc_id:
        sql += " AND doc_id = ?"
        params = (doc_id,)

    facts = []
    for row in conn.execute(sql, params):
        facts.append(_fact_from_row(row))
    return facts


def _fact_from_row(row) -> Fact:
    evidence = json.loads(row["evidence_json"])
    value = json.loads(row["value_json"]) or {}
    quantity = None
    payload = value.get("quantity")
    if payload:
        payload["interval"] = tuple(payload["interval"])
        quantity = Quantity(**payload)

    qualifiers = {}
    for key, payload in json.loads(row["qualifiers_json"]).items():
        period = None
        if payload.get("period"):
            spell = payload["period"]
            period = Period(
                start=_date(spell["start"]),
                end=_date(spell["end"]),
                label=payload["value"],
                granularity=spell["granularity"],
                fiscal_basis=spell["fiscal_basis"],
            )
        qualifiers[key] = Qualifier(
            key=key,
            value=payload["value"],
            provenance=payload["provenance"],
            period=period,
        )

    return Fact(
        fact_id=row["fact_id"],
        doc_id=row["doc_id"],
        claim_text=row["claim_text"],
        subject_surface=row["subject_surface"] or "",
        subject_key=row["subject_key"],
        subject_type=row["subject_type"],
        predicate=row["predicate"],
        predicate_canonical=row["predicate_canonical"] or "",
        value_kind=row["value_kind"],
        value_raw=value.get("raw", ""),
        quantity=quantity,
        qualifiers=qualifiers,
        evidence=Evidence(**evidence),
        confidence=row["confidence"] or 1.0,
        flags=json.loads(row["flags_json"]),
    )


def _date(value: str):
    from datetime import date

    return date.fromisoformat(value)


def save_registry(conn: Connection, registry) -> int:
    """Persist the predicate vocabulary the corpus has taught us so far.

    Embeddings are stored as raw float32 so a later ingest can search the
    registry without re-encoding every predicate it has ever seen.
    """
    conn.executemany(
        """
        INSERT INTO predicate_registry (canonical, aliases_json, embedding,
                                        first_seen_doc, count, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(canonical) DO UPDATE SET
            aliases_json = excluded.aliases_json,
            embedding = excluded.embedding,
            count = excluded.count
        """,
        [
            (
                entry.canonical,
                json.dumps(entry.aliases),
                None if entry.embedding is None else entry.embedding.astype("float32").tobytes(),
                entry.first_seen_doc,
                entry.count,
                _now(),
            )
            for entry in registry.entries.values()
        ],
    )
    return len(registry.entries)


def load_registry(conn: Connection, encoder=None, client=None):
    """Rebuild the registry, so a new document meets everything already known."""
    import numpy as np

    from concord.registry import Entry, PredicateRegistry

    entries = []
    for row in conn.execute("SELECT * FROM predicate_registry ORDER BY canonical"):
        blob = row["embedding"]
        entries.append(
            Entry(
                canonical=row["canonical"],
                aliases=json.loads(row["aliases_json"]),
                embedding=np.frombuffer(blob, dtype="float32") if blob else None,
                first_seen_doc=row["first_seen_doc"],
                count=row["count"],
            )
        )
    return PredicateRegistry(entries, encoder=encoder, client=client)


def log_schema_events(conn: Connection, events, doc_id: str | None = None) -> int:
    """Append the timeline. Uneventful decisions are recorded too.

    `predicate_seen` rows are what make the interesting rows legible: without
    them there is no denominator, and "14 new predicates" means nothing.
    """
    conn.executemany(
        """
        INSERT INTO schema_events (doc_id, event_type, subject, target,
                                   similarity, decided_by, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                event.doc_id or doc_id,
                event.event_type,
                event.subject,
                event.target,
                event.similarity,
                event.decided_by,
                _now(),
            )
            for event in events
        ],
    )
    return len(events)


def schema_timeline(conn: Connection, doc_id: str | None = None, limit: int = 500) -> list[dict]:
    sql = "SELECT * FROM schema_events"
    params: tuple = ()
    if doc_id:
        sql += " WHERE doc_id = ?"
        params = (doc_id,)
    sql += " ORDER BY event_id DESC LIMIT ?"
    return [dict(row) for row in conn.execute(sql, (*params, limit))]


def document_text(conn: Connection, doc_id: str) -> str:
    row = conn.execute(
        "SELECT text_path FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    if row is None:
        raise KeyError(doc_id)
    return Path(row["text_path"]).read_text(encoding="utf-8")


def counts(conn: Connection) -> dict[str, int]:
    """The integrity numbers the UI shows in its header."""
    grounded = conn.execute(
        "SELECT COUNT(*) FROM facts WHERE align_status != 'unlocated'"
    ).fetchone()[0]
    quarantined = conn.execute(
        "SELECT COUNT(*) FROM facts WHERE align_status = 'unlocated'"
    ).fetchone()[0]
    return {
        "documents": conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
        "facts": grounded,
        "quarantined": quarantined,
        "relations": conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0],
    }
