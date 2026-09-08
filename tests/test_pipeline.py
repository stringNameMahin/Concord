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
