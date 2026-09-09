"""The HTTP surface: upload a PDF, inspect what came out, check it is real.

`/verify/{fact_id}` is the endpoint worth reading first. It re-slices the
parser's canonical text at the offsets stored on the fact and compares the
result to the stored quote. Every fact in the ledger claims to round-trip to
source bytes; this is the endpoint that lets a reviewer falsify that claim one
fact at a time, rather than taking the claim on trust.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from concord import config
from concord.compare.adjudicate import adjudicate
from concord.compare.engine import compare
from concord.llm.client import LLMClient
from concord.parse.pdf import UnreadablePDF, file_sha256
from concord.pipeline import ingest
from concord.store import repo
from concord.store.db import session

STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="Concord",
    description="A fact knowledge layer: grounded facts with cross-document reconciliation",
)

_encoder = None


def encoder():
    """Loaded once, on first use. Importing torch at startup would make the
    API slow to boot for requests that never need an embedding."""
    global _encoder
    if _encoder is None:
        from concord.compare.embed import LocalEncoder

        _encoder = LocalEncoder()
    return _encoder


def _finite(obj):
    """Encode an infinite interval bound as null.

    A bounded figure - `over 33,200` - normalises to a half-open precision
    interval whose open end is infinity. That is the correct comparison
    semantics and it round-trips through SQLite, because Python's json
    writes and reads the non-standard token `Infinity`. Strict JSON has no
    such value, so Starlette refuses to serialise it and the whole page
    500s - which took out the ledger and relations views on any page that
    happened to contain one. `null` is the honest encoding of an unbounded
    end, and the interval stays infinite everywhere it is actually used.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


def _fact_row(row) -> dict:
    evidence = json.loads(row["evidence_json"])
    return {
        "fact_id": row["fact_id"],
        "doc_id": row["doc_id"],
        "claim_text": row["claim_text"],
        "subject": {"surface": row["subject_surface"], "key": row["subject_key"]},
        "predicate": row["predicate"],
        "comparison_key": row["comparison_key"],
        "value": _finite(json.loads(row["value_json"])),
        "value_kind": row["value_kind"],
        "normalized": _finite(row["value_normalized"]),
        "qualifiers": _finite(json.loads(row["qualifiers_json"])),
        "evidence": evidence,
        "align_status": row["align_status"],
        "page": row["page"],
        "confidence": _finite(row["confidence"]),
        "flags": json.loads(row["flags_json"]),
    }


def _relation_row(row) -> dict:
    return {
        "relation_id": row["relation_id"],
        "fact_a": row["fact_a"],
        "fact_b": row["fact_b"],
        "verdict": row["verdict"],
        "rule_fired": row["rule_fired"],
        "qualifier_key": row["qualifier_key"],
        "explanation": row["explanation"],
        "decided_by": row["decided_by"],
        "self_consistent": row["self_consistent"],
        "cross_document": bool(row["cross_document"]),
        "blocked_by": json.loads(row["blocked_by"]),
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    with session() as conn:
        return {"status": "ok", "offline": config.OFFLINE, **repo.counts(conn)}


@app.post("/ingest")
async def ingest_document(file: UploadFile, adjudicate_residue: bool = Query(True)):
    """Parse, extract, ground, compare and adjudicate one uploaded PDF.

    The new document's facts are blocked against everything already in the
    ledger, so relations are cross-document from the second upload onward.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "a PDF is required")

    suffix = Path(file.filename).name
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
        target = Path(work) / suffix
        with target.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)

        # Identity is the content hash. The same bytes re-uploaded under a new
        # name are the same document, and must reach the same ids.
        with session() as conn:
            known = repo.document_id_for_hash(conn, file_sha256(target))

        client = LLMClient()
        try:
            result = ingest(target, client, doc_id=known or Path(suffix).stem)
        except UnreadablePDF as exc:
            # Named like a PDF, but nothing in it parses. That is the upload's
            # fault, not an upstream failure, so it gets the same 400 as a file
            # that is not named like one - and a reason rather than a trace.
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # extraction is the one step that can fail hard
            raise HTTPException(
                502, f"extraction failed: {type(exc).__name__}: {exc}"
            ) from exc

    # Every request failed, so nothing was extracted. Say why, and say what to
    # do about it: a reviewer who uploads a PDF and is handed "0 facts" has no
    # way to tell a broken install from a document that states nothing.
    if result.extraction.total_failure:
        detail = (
            "Extraction produced nothing: all "
            f"{result.extraction.requests} requests failed. "
        )
        if not config.LLM_API_KEYS:
            detail += (
                "No API key is configured. Ingesting a new PDF needs one - set "
                "GEMINI_API_KEY (the free tier is enough) or OPENROUTER_API_KEY "
                "and restart. The documents already in the ledger can be browsed "
                "and verified without any key."
            )
        else:
            detail += f"First failure: {result.extraction.failures[0]}"
        raise HTTPException(503, detail)

    with session() as conn:
        doc_id = repo.save_document(conn, result)
        repo.save_facts(conn, result.facts, doc_id)
        repo.save_quarantine(conn, result.extraction, doc_id)
        everything = repo.load_facts(conn)

        # The schema learns before the comparison runs, so this document's
        # predicates can already be aliased onto the vocabulary it is about to
        # be compared against.
        registry = repo.load_registry(conn, encoder=encoder(), client=client)
        events = registry.observe_all(result.facts, doc_id)
        repo.save_registry(conn, registry)
        repo.log_schema_events(conn, events, doc_id)

        # Only pairs touching this document are judged. The rest are already in
        # the ledger and nothing about them has changed.
        fresh = frozenset(fact.fact_id for fact in result.facts)
        run = compare(
            everything, encoder=encoder(), aliases=registry.aliases(), fresh=fresh
        )
        adjudication = None
        if adjudicate_residue and run.llm_queue:
            try:
                adjudication = adjudicate(run.llm_queue, everything, client)
            except Exception:
                adjudication = None  # deterministic verdicts stand on their own
        repo.save_relations(conn, run.relations)

        payload = {
            "doc_id": doc_id,
            "pages": result.doc.n_pages,
            "chunks": len(result.chunks),
            "facts": len(result.facts),
            "quarantined": len(result.extraction.quarantined),
            "quarantine_rate": round(result.extraction.quarantine_rate, 4),
            "failed_batches": result.extraction.failed_batches,
            "duplicates_collapsed": result.duplicates,
            "requests": result.extraction.requests,
            "blocking": {
                "theoretical_pairs": run.blocking.theoretical_pairs,
                "candidates": run.blocking.candidates,
                "reduction": round(run.blocking.reduction, 6),
                "by_strategy": run.blocking.by_strategy,
            },
            "verdicts": dict(run.final_verdicts()),
            "llm_pairs": len(run.llm_queue),
            "schema": {
                "predicates_known": len(registry),
                "new_predicates": sum(
                    1 for e in events if e.event_type == "predicate_registered"
                ),
                "aliases_confirmed": sum(
                    1 for e in events if e.event_type == "alias_confirmed"
                ),
                "alias_questions": registry.calls,
            },
        }
        if adjudication:
            payload["adjudication"] = {
                "calls": adjudication.calls,
                "self_consistency_rate": round(adjudication.self_consistency_rate, 4),
                "rejection_rate": round(adjudication.rejection_rate, 4),
                "unadjudicated": adjudication.unadjudicated,
            }
        return payload


@app.get("/facts")
def list_facts(
    doc_id: str | None = None,
    predicate: str | None = None,
    q: str | None = None,
    quarantined: bool = False,
    limit: int = Query(200, le=2000),
    offset: int = 0,
):
    clauses = ["align_status = 'unlocated'" if quarantined else "align_status != 'unlocated'"]
    params: list = []
    if doc_id:
        clauses.append("doc_id = ?")
        params.append(doc_id)
    if predicate:
        clauses.append("predicate_canonical = ?")
        params.append(predicate)
    if q:
        clauses.append("(claim_text LIKE ? OR subject_surface LIKE ? OR predicate LIKE ?)")
        params.extend([f"%{q}%"] * 3)

    where = " AND ".join(clauses)
    with session() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM facts WHERE {where}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM facts WHERE {where} ORDER BY doc_id, page, fact_id LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        return {"total": total, "facts": [_fact_row(row) for row in rows]}


@app.get("/relations")
def list_relations(
    verdict: str | None = None,
    cross_document: bool | None = None,
    limit: int = Query(200, le=2000),
    offset: int = 0,
):
    """Everything except `unrelated`, which blocking produces by the thousand."""
    clauses = ["verdict != 'unrelated'"]
    params: list = []
    if verdict:
        clauses[0] = "verdict = ?"
        params.append(verdict)
    if cross_document:
        clauses.append("cross_document = 1")

    where = " AND ".join(clauses)
    with session() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM relations WHERE {where}", params).fetchone()[0]
        rows = conn.execute(
            f"""SELECT * FROM relations WHERE {where}
                ORDER BY cross_document DESC, verdict, relation_id LIMIT ? OFFSET ?""",
            [*params, limit, offset],
        ).fetchall()

        relations = []
        for row in rows:
            payload = _relation_row(row)
            for side in ("fact_a", "fact_b"):
                fact = conn.execute(
                    "SELECT * FROM facts WHERE fact_id = ?", (payload[side],)
                ).fetchone()
                payload[side.replace("fact_", "") + "_fact"] = _fact_row(fact) if fact else None
            relations.append(payload)
        return {"total": total, "relations": relations}


@app.get("/evidence/{fact_id}")
def evidence(fact_id: str, window: int = Query(600, le=4000)):
    """The quote in its surrounding text, with offsets for highlighting."""
    with session() as conn:
        row = conn.execute("SELECT * FROM facts WHERE fact_id = ?", (fact_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "no such fact")

        fact = _fact_row(row)
        span = fact["evidence"]
        if span["char_start"] < 0:
            return {"fact": fact, "located": False, "context": None}

        text = repo.document_text(conn, row["doc_id"])
        start = max(0, span["char_start"] - window)
        end = min(len(text), span["char_end"] + window)
        return {
            "fact": fact,
            "located": True,
            "context": text[start:end],
            "context_start": start,
            "highlight": [span["char_start"] - start, span["char_end"] - start],
        }


@app.get("/verify/{fact_id}")
def verify(fact_id: str):
    """Re-cut the source at the stored offsets and compare to the stored quote.

    This is the grounding guarantee made falsifiable. It reads the document
    text fresh rather than trusting the row, so a parser change or a corrupted
    offset shows up here as `verified: false` instead of going unnoticed.
    """
    with session() as conn:
        row = conn.execute("SELECT * FROM facts WHERE fact_id = ?", (fact_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "no such fact")

        span = json.loads(row["evidence_json"])
        if span["char_start"] < 0:
            return {
                "fact_id": fact_id,
                "verified": False,
                "reason": "quarantined: the quote was never located in the source",
                "align_status": row["align_status"],
            }

        text = repo.document_text(conn, row["doc_id"])
        actual = text[span["char_start"] : span["char_end"]]
        return {
            "fact_id": fact_id,
            "verified": actual == span["quote"],
            "doc_id": row["doc_id"],
            "page": row["page"],
            "char_start": span["char_start"],
            "char_end": span["char_end"],
            "stored_quote": span["quote"],
            "text_at_offsets": actual,
            "align_status": row["align_status"],
        }


@app.get("/schema")
def schema(doc_id: str | None = None, limit: int = Query(300, le=2000)):
    """The vocabulary the corpus taught the layer, and when it learned each part.

    This is the emergent-schema view: no ontology was declared up front, so the
    only honest description of the schema is the record of how it grew.
    """
    with session() as conn:
        registry = repo.load_registry(conn)
        events = repo.schema_timeline(conn, doc_id, limit)
        counts: dict[str, int] = {}
        for event in events:
            counts[event["event_type"]] = counts.get(event["event_type"], 0) + 1

        return {
            "predicates": len(registry),
            "aliases": sum(len(e.aliases) for e in registry.entries.values()),
            "event_counts": counts,
            "entries": sorted(
                (
                    {
                        "canonical": e.canonical,
                        "aliases": e.aliases,
                        "count": e.count,
                        "first_seen_doc": e.first_seen_doc,
                    }
                    for e in registry.entries.values()
                ),
                key=lambda e: (-e["count"], e["canonical"]),
            ),
            "timeline": events,
        }


@app.get("/stats")
def stats():
    """What the ledger contains, for the header and the README."""
    with session() as conn:
        base = repo.counts(conn)
        verdicts = {
            row["verdict"]: row["n"]
            for row in conn.execute(
                "SELECT verdict, COUNT(*) AS n FROM relations GROUP BY verdict"
            )
        }
        by_document = [
            dict(row)
            for row in conn.execute(
                """SELECT d.doc_id, d.filename, d.n_pages,
                          (SELECT COUNT(*) FROM facts f
                            WHERE f.doc_id = d.doc_id
                              AND f.align_status != 'unlocated') AS facts,
                          (SELECT COUNT(*) FROM facts f
                            WHERE f.doc_id = d.doc_id
                              AND f.align_status = 'unlocated') AS quarantined
                     FROM documents d ORDER BY d.doc_id"""
            )
        ]
        total = base["facts"] + base["quarantined"]
        return {
            **base,
            "quarantine_rate": round(base["quarantined"] / total, 4) if total else 0.0,
            "verdicts": verdicts,
            "by_document": by_document,
        }


@app.exception_handler(KeyError)
def missing(_request, exc):
    return JSONResponse({"detail": f"not found: {exc}"}, status_code=404)
