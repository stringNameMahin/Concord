"""The HTTP surface.

`/verify` gets the most attention here. It is the endpoint that makes the
grounding guarantee falsifiable, so the test that matters is not that it
returns 200 - it is that it returns `verified: false` when the source and the
stored quote genuinely disagree.
"""

import json

import pytest
from factories import make_fact
from fastapi.testclient import TestClient

from concord.compare.engine import compare
from concord.store import repo
from concord.store.db import connect

TEXT = (
    "Delhivery Limited\n"
    "PADDING " * 40 + "\nRevenue from services 81,415.38 for the year ended March 31, 2024.\n"
    + "PADDING " * 40
)
QUOTE = "Revenue from services 81,415.38"
START = TEXT.index(QUOTE)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("concord.config.DB_PATH", tmp_path / "ledger.sqlite")
    monkeypatch.setattr("concord.config.WORK_DIR", tmp_path / "work")

    text_file = tmp_path / "work" / "d1.txt"
    text_file.parent.mkdir(parents=True, exist_ok=True)
    text_file.write_text(TEXT, encoding="utf-8")

    conn = connect(tmp_path / "ledger.sqlite")
    conn.execute(
        """INSERT INTO documents (doc_id, filename, sha256, n_pages, status, ingested_at, text_path)
           VALUES ('d1', 'd1.pdf', 'abc', 4, 'ingested', '2026-01-01', ?)""",
        (str(text_file),),
    )

    grounded = make_fact(
        fact_id="f_good",
        doc="d1",
        raw="81,415.38 mn",
        span=START,
        period="FY 2023-24",
    )
    object.__setattr__(grounded.evidence, "char_end", START + len(QUOTE))
    object.__setattr__(grounded.evidence, "quote", QUOTE)

    other = make_fact(fact_id="f_other", doc="d1", raw="7,224 Cr", period="FY 2023-24")
    repo.save_facts(conn, [grounded, other], "d1")
    repo.save_relations(conn, compare([grounded, other]).relations)
    conn.commit()
    conn.close()

    from concord.api.app import app

    return TestClient(app)


def test_health_reports_what_the_ledger_holds(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["facts"] == 2


def test_the_ledger_lists_grounded_facts(client):
    body = client.get("/facts").json()
    assert body["total"] == 2
    assert {f["fact_id"] for f in body["facts"]} == {"f_good", "f_other"}


def test_facts_can_be_searched_and_filtered(client):
    assert client.get("/facts?q=zzzznothing").json()["total"] == 0
    assert client.get("/facts?doc_id=d1").json()["total"] == 2
    assert client.get("/facts?doc_id=nosuchdoc").json()["total"] == 0


def test_relations_hide_the_unrelated_and_carry_both_facts(client):
    body = client.get("/relations").json()
    assert all(r["verdict"] != "unrelated" for r in body["relations"])
    relation = body["relations"][0]
    assert relation["a_fact"]["fact_id"]
    assert relation["b_fact"]["fact_id"]
    assert relation["explanation"]


def test_evidence_highlight_brackets_the_quote_exactly(client):
    """The offsets the UI highlights must cut out the stored quote."""
    body = client.get("/evidence/f_good").json()
    start, end = body["highlight"]
    assert body["located"]
    assert body["context"][start:end] == QUOTE


def test_verify_confirms_the_round_trip_to_source_bytes(client):
    body = client.get("/verify/f_good").json()
    assert body["verified"] is True
    assert body["text_at_offsets"] == body["stored_quote"] == QUOTE


def test_verify_fails_loudly_when_the_offsets_stop_matching(client, tmp_path):
    """A parser change or a corrupted offset must surface here, not go unseen."""
    conn = connect(tmp_path / "ledger.sqlite")
    row = conn.execute("SELECT evidence_json FROM facts WHERE fact_id='f_good'").fetchone()
    evidence = json.loads(row["evidence_json"])
    evidence["char_start"] += 7  # as if the text were re-parsed and shifted
    conn.execute(
        "UPDATE facts SET evidence_json = ? WHERE fact_id='f_good'", (json.dumps(evidence),)
    )
    conn.commit()
    conn.close()

    body = client.get("/verify/f_good").json()
    assert body["verified"] is False
    assert body["text_at_offsets"] != body["stored_quote"]


def test_a_quarantined_fact_reports_why_it_cannot_verify(client, tmp_path):
    conn = connect(tmp_path / "ledger.sqlite")
    conn.execute(
        """INSERT INTO facts (fact_id, doc_id, claim_text, predicate, value_json, value_kind,
                              evidence_json, align_status, page, flags_json, created_at)
           VALUES ('f_lost', 'd1', 'c', 'p', '{"raw":"1","kind":"quantity","quantity":null}',
                   'quantity', ?, 'unlocated', 1, '["unlocated"]', '2026-01-01')""",
        (json.dumps({"doc_id": "d1", "page": 1, "char_start": -1, "char_end": -1,
                     "quote": "never found", "align_status": "unlocated"}),),
    )
    conn.commit()
    conn.close()

    assert client.get("/facts").json()["total"] == 2
    quarantined = client.get("/facts?quarantined=true").json()
    assert quarantined["total"] == 1

    body = client.get("/verify/f_lost").json()
    assert body["verified"] is False
    assert "quarantined" in body["reason"]
    assert client.get("/evidence/f_lost").json()["located"] is False


def test_stats_carry_the_integrity_numbers(client):
    """`documents` is a count and `by_document` is the breakdown. They collided
    once, and the count silently became a list the header could not render."""
    body = client.get("/stats").json()
    assert body["documents"] == 1
    assert body["quarantine_rate"] == 0.0
    assert [d["doc_id"] for d in body["by_document"]] == ["d1"]
    assert body["by_document"][0]["facts"] == 2


def test_unknown_ids_are_404_not_500(client):
    assert client.get("/verify/nope").status_code == 404
    assert client.get("/evidence/nope").status_code == 404


def test_the_default_page_of_every_listing_serves(client):
    """The endpoints a reviewer hits first, with no query string at all.

    Both defaulted to `limit=200` and nothing exercised that: every example in
    the manual guide passes a small limit, and no fixture built a bounded
    figure. The shipped ledger had four, and both listings 500ed.
    """
    for path in ("/facts", "/relations", "/schema", "/stats", "/health"):
        assert client.get(path).status_code == 200, path


def test_an_unbounded_figure_serialises_with_a_null_open_end(client, tmp_path):
    """`over 33,200` is a half-open interval, and JSON has no infinity.

    Python's json writes and reads the non-standard token `Infinity`, so the
    value round-trips through SQLite and only dies at the HTTP boundary, where
    Starlette serialises with allow_nan=False. The open end has to reach the
    client as null; the bound field is what carries the semantics.
    """
    from concord.store.db import connect

    conn = connect(tmp_path / "ledger.sqlite")
    bounded = make_fact(fact_id="f_bounded", doc="d1", raw="over 4,100")
    assert bounded.quantity.interval[1] == float("inf")  # the fixture is the case
    repo.save_facts(conn, [bounded], "d1")
    conn.commit()
    conn.close()

    response = client.get("/facts?limit=300")
    assert response.status_code == 200

    fact = next(f for f in response.json()["facts"] if f["fact_id"] == "f_bounded")
    quantity = fact["value"]["quantity"]
    assert quantity["interval"] == [4100.0, None]
    assert quantity["bound"] == "greater_than"


def test_ingest_refuses_anything_that_is_not_a_pdf(client):
    response = client.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_a_corrupt_pdf_is_a_400_with_a_reason_not_a_500(client):
    """Named like a PDF, but nothing in it parses.

    This was a 500 on Windows: the 502 raised for the parse failure was
    discarded when the temp directory could not be cleaned up, and the
    reviewer got a stack trace instead of a reason on their first action with
    an unusual file. The suite only ever fed this endpoint real PDFs, which is
    why nothing caught it.
    """
    response = client.post(
        "/ingest", files={"file": ("broken.pdf", b"%PDF-1.4 garbage", "application/pdf")}
    )
    assert response.status_code == 400

    detail = response.json()["detail"]
    assert "broken.pdf" in detail
    assert "not a readable PDF" in detail


def test_the_page_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Concord" in response.text


def test_the_schema_view_reports_a_vocabulary_nobody_declared(client, tmp_path):
    """Phase 7's UI surface: what the layer learned, and when."""
    from test_registry import Judge, WordOverlap

    from concord.registry import PredicateRegistry
    from concord.store.db import connect

    conn = connect(tmp_path / "ledger.sqlite")
    registry = PredicateRegistry(encoder=WordOverlap(), client=Judge(same=True), threshold=0.5)
    events = [
        registry.observe("revenue_from_services", "d1"),
        registry.observe("revenue_from_operations", "d1"),
        registry.observe("board_meeting_attendance", "d1"),
    ]
    repo.save_registry(conn, registry)
    repo.log_schema_events(conn, events)
    conn.commit()
    conn.close()

    body = client.get("/schema").json()
    assert body["predicates"] == 2
    assert body["aliases"] == 1
    assert body["event_counts"]["alias_confirmed"] == 1
    assert body["event_counts"]["predicate_registered"] == 2

    canonical = {e["canonical"]: e for e in body["entries"]}
    assert canonical["revenue_from_services"]["aliases"] == ["revenue_from_operations"]
    assert [e["event_type"] for e in body["timeline"]][0] == "predicate_registered"


def test_the_schema_view_is_empty_and_calm_before_any_ingest(client):
    body = client.get("/schema").json()
    assert body == {
        "predicates": 0,
        "aliases": 0,
        "event_counts": {},
        "entries": [],
        "timeline": [],
    }


def test_the_same_bytes_uploaded_twice_are_one_document(client, tmp_path):
    """A reviewer re-uploading a PDF, or uploading a renamed copy, must not
    crash on the unique content hash."""
    from concord.store.db import connect

    conn = connect(tmp_path / "ledger.sqlite")
    assert repo.document_id_for_hash(conn, "abc") == "d1"
    assert repo.document_id_for_hash(conn, "never-seen") is None
    conn.close()


def test_uploading_without_a_key_says_so_instead_of_returning_nothing(client, monkeypatch):
    """The gap a reviewer would hit first: upload an unseen PDF with no
    credentials. Returning `0 facts` cannot be told apart from a document that
    genuinely states nothing."""
    from concord.extract.runner import ExtractionRun

    import concord.api.app as app_module

    class Empty:
        doc = type("D", (), {"n_pages": 1, "sha256": "zzz", "text": "x", "filename": "u.pdf",
                             "parser": "p", "parser_version": "1"})()
        structure = None
        chunks = []
        facts = []
        duplicates = 0
        fy_end_month = None
        extraction = ExtractionRun(requests=3, failed_batches=3,
                                   failures=["LLMError: no API key; set GEMINI_API_KEY"])

    monkeypatch.setattr(app_module, "ingest", lambda *a, **k: Empty())
    monkeypatch.setattr(app_module.config, "LLM_API_KEYS", [])

    response = client.post("/ingest", files={"file": ("new.pdf", b"%PDF-1.4", "application/pdf")})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "all 3 requests failed" in detail
    assert "GEMINI_API_KEY" in detail
    assert "without any key" in detail  # tells them what still works


def test_a_failure_with_a_key_present_reports_the_actual_cause(client, monkeypatch):
    from concord.extract.runner import ExtractionRun

    import concord.api.app as app_module

    class Empty:
        doc = type("D", (), {"n_pages": 1, "sha256": "zzz", "text": "x", "filename": "u.pdf",
                             "parser": "p", "parser_version": "1"})()
        structure = None
        chunks = []
        facts = []
        duplicates = 0
        fy_end_month = None
        extraction = ExtractionRun(requests=2, failed_batches=2,
                                   failures=["LLMError: 429: quota exhausted"])

    monkeypatch.setattr(app_module, "ingest", lambda *a, **k: Empty())
    monkeypatch.setattr(app_module.config, "LLM_API_KEYS", ["a-key"])

    detail = client.post(
        "/ingest", files={"file": ("new.pdf", b"%PDF-1.4", "application/pdf")}
    ).json()["detail"]
    assert "quota exhausted" in detail
    assert "GEMINI_API_KEY" not in detail  # they have one; that is not the problem


def test_a_password_protected_pdf_is_a_400_and_not_a_502(client, tmp_path):
    """Bug 24 at the endpoint. 502 says the provider failed; nothing reached
    a provider. The caller sent a file nobody here can open, which is the same
    class of problem as a corrupt one and gets the same answer."""
    import pymupdf

    locked = tmp_path / "locked.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "confidential")
    doc.save(str(locked), encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    doc.close()

    response = client.post(
        "/ingest",
        files={"file": ("locked.pdf", locked.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 400
    assert "password" in response.json()["detail"]


def test_the_page_never_keys_an_element_id_on_a_fact_id(client):
    """Bug 22, as a shape rule rather than a browser test.

    One fact appears in the ledger and in every relation card it belongs to,
    so an element id built from its fact id is not unique on the page.
    `querySelector` then returns the first in document order, and a reviewer
    clicking the lower card was shown another occurrence's evidence under a
    "verified" tick - the one claim this whole system rests on. The evidence
    box is found by walking up from the clicked button instead.
    """
    page = client.get("/").text

    assert 'id="ev-' not in page
    assert 'showEvidence(\'${f.fact_id}\', this)' in page
    assert 'btn.closest(".side, .card").querySelector(".evidence")' in page
