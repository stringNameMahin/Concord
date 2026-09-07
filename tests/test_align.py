import random
from pathlib import Path

import pytest

from concord.extract.align import Aligner, normalize_ws
from concord.parse.pdf import parse

CORPUS = Path(__file__).resolve().parent.parent / "starter-datasets" / "starter-datasets"
PDFS = sorted(CORPUS.rglob("*.pdf"))

TEXT = "Notes\nRevenue from services*\n81,415.38\t72,236.47\nTotal   revenue\n"


def test_normalize_maps_every_char_back():
    norm, index = normalize_ws(TEXT)
    assert len(norm) == len(index)
    for pos, char in enumerate(norm):
        if char != " ":
            assert TEXT[index[pos]] == char


def test_exact_match_round_trips():
    a = Aligner(TEXT)
    found = a.locate("81,415.38")
    assert found.status == "exact"
    assert TEXT[found.start : found.end] == found.text == "81,415.38"


def test_quote_spanning_a_newline_is_located():
    a = Aligner(TEXT)
    found = a.locate("Revenue from services* 81,415.38")
    assert found.located
    assert TEXT[found.start : found.end] == found.text
    assert "81,415.38" in found.text


def test_collapsed_internal_whitespace_is_located():
    a = Aligner(TEXT)
    found = a.locate("Total revenue")
    assert found.located
    assert TEXT[found.start : found.end] == found.text


def test_near_miss_is_located_fuzzily():
    a = Aligner(TEXT)
    found = a.locate("Revenue from service 81,415.38")
    assert found.status in ("exact", "fuzzy")
    assert TEXT[found.start : found.end] == found.text


def test_absent_quote_is_unlocated_not_guessed():
    a = Aligner(TEXT)
    found = a.locate("EBITDA margin improved to 1.6 percent in the fourth quarter")
    assert found.status == "unlocated"
    assert found.start == -1


def test_empty_quote_is_unlocated():
    assert Aligner(TEXT).locate("   ").status == "unlocated"


@pytest.mark.skipif(not PDFS, reason="starter corpus not present")
@pytest.mark.parametrize("name", [p.name for p in PDFS])
def test_real_document_spans_round_trip(name):
    """Quotes lifted from a real document must relocate to the same bytes.

    Whitespace is deliberately mangled the way a model would rewrite a table
    row, so this exercises the normalized path rather than plain find().
    """
    doc = parse(next(p for p in PDFS if p.name == name))
    aligner = Aligner(doc.text)
    rng = random.Random(0)

    lines = [line for line in doc.lines if len(line.text) > 25]
    located = 0
    for line in rng.sample(lines, min(50, len(lines))):
        quote = " ".join(line.text.split())
        found = aligner.locate(quote, near=(line.start, line.end))
        if found.located:
            located += 1
            assert doc.text[found.start : found.end] == found.text

    assert located >= 48, f"only {located}/50 quotes relocated in {name}"
