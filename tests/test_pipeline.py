"""The ingest seam: what survives from one document into the ledger."""

import pytest
from factories import make_fact

from concord.parse.pdf import UnreadablePDF, parse
from concord.pipeline import dedupe


def test_the_same_claim_extracted_twice_collapses_to_one_record():
    """Observed on the real deck: identical facts on two extraction passes."""
    twice = [make_fact(fact_id="f_x", span=100), make_fact(fact_id="f_x", span=100)]
    kept, dropped = dedupe(twice)
    assert len(kept) == 1 and dropped == 1


def test_the_better_qualified_copy_wins():
    """The impoverished copy would drag pairs into insufficient_context."""
    bare = make_fact(fact_id="f_x", span=100)
    qualified = make_fact(fact_id="f_x", span=100, period="FY24", consolidation="consolidated")
    for order in ([bare, qualified], [qualified, bare]):
        kept, dropped = dedupe(order)
        assert dropped == 1
        assert kept[0].qualifier_keys() == {"period", "consolidation"}


def test_distinct_facts_are_left_alone():
    facts = [make_fact(fact_id="f_a"), make_fact(fact_id="f_b")]
    kept, dropped = dedupe(facts)
    assert len(kept) == 2 and dropped == 0


BROKEN = [
    ("garbage after the header", b"%PDF-1.4 garbage"),
    ("an empty file", b""),
    ("a truncated object", b"%PDF-1.7 1 0 obj << /Type /Catalog"),
]


@pytest.mark.parametrize("what,payload", BROKEN, ids=[w for w, _ in BROKEN])
def test_an_unreadable_pdf_is_refused_by_name(tmp_path, what, payload):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(payload)

    with pytest.raises(UnreadablePDF) as caught:
        parse(broken)

    # The reason has to name the file. A reviewer uploading three documents
    # needs to know which one was rejected.
    assert "broken.pdf" in str(caught.value)


def test_refusing_a_pdf_does_not_leave_it_open(tmp_path):
    """The 500-on-corrupt-PDF bug, at its source.

    PyMuPDF holds an OS file handle on a Document whose construction failed,
    and the traceback of the error it raises keeps that frame alive for as
    long as anything holds the exception - which `raise HTTPException(...)
    from exc` does. Cleaning up the temp directory then hit PermissionError on
    Windows, and that replaced the considered response with a stack trace.

    Holding `caught` across the unlink is what reproduces it, so this test
    fails on Windows against the old path-based open and passes on Linux
    either way.
    """
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4 garbage")

    caught = None
    try:
        parse(broken)
    except UnreadablePDF as exc:
        caught = exc

    assert caught is not None
    broken.unlink()  # PermissionError before the fix
    assert not broken.exists()


def test_a_password_protected_pdf_is_refused_like_any_unreadable_one(tmp_path):
    """Bug 24. An encrypted file opens cleanly and fails later.

    PyMuPDF returns a Document for it, so the guard on `open` never sees it
    and the failure surfaced deep in the page loop as an unclassified
    exception - which the endpoint could only answer 502 for, blaming the
    provider for the caller's file. `needs_pass` is knowable the moment the
    handle exists, so it is checked there.
    """
    import pymupdf

    locked = tmp_path / "locked.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "confidential")
    doc.save(str(locked), encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    doc.close()

    with pytest.raises(UnreadablePDF) as caught:
        parse(locked)

    assert "locked.pdf" in str(caught.value)
    assert "password" in str(caught.value)


# --- end to end: one synthetic PDF, one fake model, a real ledger row -------
#
# `ingest()` had no end-to-end test through two audits, and its absence is why
# F1 (a magnitude dropped from a fifth of quantity facts) was invisible: every
# comparison fixture built a fact whose scale was already inside `raw`, so the
# bug class "figure in the cell, magnitude in the surrounding bytes" could not
# be expressed. This builds the document, so the bytes are real.

# A running header on every page, the way a filing carries one, and the
# figures far enough down the page to stay out of the furniture band.
HEADER = "Financial Statements: Consolidated"
PAGES = [
    [
        "Statement of Profit and Loss for the year ended March 31, 2024",
        "Revenue from services 81,415.38 million for the year ended March 31, 2024.",
    ],
    ["Active customers stood at 33,200 as at March 31, 2024."],
    ["The board met 8 times during the year ended March 31, 2024."],
    ["Acme Logistics Limited is incorporated in India."],
]


def synthetic_pdf(path, pages=PAGES, header=HEADER):
    import pymupdf

    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        page.insert_text((72, 60), header, fontsize=9)
        for row, line in enumerate(lines):
            page.insert_text((72, 260 + row * 18), line, fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


class Extractor:
    """A model that reports the figures and leaves the magnitude in the page.

    This is the behaviour measured on the real one: handed
    "81,415.38 million" it returns `raw="81,415.38"` with `scale` null, and the
    record is then wrong by a factor of a million while looking ordinary.
    """

    def __init__(self, facts):
        self.facts = facts
        self.calls = 0

    def complete(self, prompt, schema, system=None, temperature=0.0):
        from concord.extract.models import ExtractionOut

        self.calls += 1
        return ExtractionOut(facts=[build() for build in self.facts])


def emitted(quote, predicate, raw, kind="quantity", quals=(), **value):
    from concord.extract.models import FactOut, QualifierOut, ValueOut

    def build():
        return FactOut(
            passage_id=0,
            claim_text=quote,
            subject_surface="Acme Logistics Limited",
            predicate=predicate,
            value=ValueOut(kind=kind, raw=raw, **value),
            qualifiers=[
                QualifierOut(key=k, value=v, provenance="stated") for k, v in quals
            ],
            quote=quote,
            confidence=0.9,
        )

    return build


def test_ingest_grounds_materialises_and_scales_one_document(tmp_path):
    from concord.pipeline import ingest

    path = synthetic_pdf(tmp_path / "acme.pdf")
    client = Extractor([
        emitted("Revenue from services 81,415.38 million", "revenue_from_services", "81,415.38"),
        emitted("Active customers stood at 33,200", "active_customers", "33,200"),
    ])
    result = ingest(path, client)

    assert client.calls == 1
    assert result.extraction.failed_batches == 0
    assert len(result.facts) == 2
    assert not result.extraction.quarantined

    by_predicate = {fact.predicate: fact for fact in result.facts}
    revenue = by_predicate["revenue_from_services"]

    # Grounding: the stored quote is the document's bytes, not the model's.
    assert result.doc.text[revenue.evidence.char_start : revenue.evidence.char_end] == (
        revenue.evidence.quote
    )
    # F1: the magnitude was in the page and not in `raw`, and it was recovered.
    assert revenue.quantity.scale == "million"
    assert revenue.quantity.normalized == 81415.38e6
    assert "scale_recovered_from_source" in revenue.flags
    # ...and a figure with no magnitude behind it does not acquire one.
    assert by_predicate["active_customers"].quantity.scale is None


def test_ingest_infers_the_fiscal_basis_and_keeps_the_evidence(tmp_path):
    from concord.pipeline import ingest

    path = synthetic_pdf(tmp_path / "acme.pdf")
    result = ingest(path, Extractor([
        emitted("Revenue from services 81,415.38 million", "revenue_from_services", "81,415.38"),
    ]))

    assert result.fy_end_month == 3
    assert sum(result.fy_evidence.values()) >= 1


def test_ingest_inherits_the_consolidation_basis_from_the_statement(tmp_path):
    """F18, end to end: the one qualifier that matters most in a financial
    document reaches the fact from the page rather than from the sentence."""
    from concord.pipeline import ingest

    path = synthetic_pdf(tmp_path / "acme.pdf")
    result = ingest(path, Extractor([
        emitted("Revenue from services 81,415.38 million", "revenue_from_services", "81,415.38"),
    ]))

    basis = result.facts[0].qualifier("consolidation")
    assert basis.value == "consolidated"
    assert basis.provenance == "inherited"


def test_an_invented_quote_never_reaches_the_ledger(tmp_path):
    from concord.pipeline import ingest

    path = synthetic_pdf(tmp_path / "acme.pdf")
    result = ingest(path, Extractor([
        emitted("Revenue from services 99,999.99 million", "revenue_from_services", "99,999.99"),
    ]))

    assert result.facts == []
    assert len(result.extraction.quarantined) == 1
    assert result.extraction.quarantine_rate == 1.0
