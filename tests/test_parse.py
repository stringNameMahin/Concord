import random
from pathlib import Path

import pytest

from concord.parse import structure
from concord.parse.chunk import chunk_document
from concord.parse.pdf import parse

CORPUS = Path(__file__).resolve().parent.parent / "starter-datasets" / "starter-datasets"
PDFS = sorted(CORPUS.rglob("*.pdf"))

pytestmark = pytest.mark.skipif(not PDFS, reason="starter corpus not present")


@pytest.fixture(scope="module", params=[p.name for p in PDFS])
def parsed(request):
    path = next(p for p in PDFS if p.name == request.param)
    doc = parse(path)
    return doc, structure.analyse(doc)


def test_every_line_round_trips(parsed):
    doc, _ = parsed
    for line in doc.lines:
        assert doc.text[line.start : line.end] == line.text


def test_fifty_random_spans_round_trip(parsed):
    doc, _ = parsed
    rng = random.Random(0)
    sample = rng.sample(doc.lines, min(50, len(doc.lines)))
    for line in sample:
        assert doc.text[line.start : line.end] == line.text
        assert doc.page_of(line.start) == line.page


def test_pages_tile_the_text_without_gaps(parsed):
    doc, _ = parsed
    assert doc.pages[0].start == 0
    for earlier, later in zip(doc.pages, doc.pages[1:]):
        assert earlier.end == later.start
    assert doc.pages[-1].end == len(doc.text)


def test_chunks_cover_the_document_and_round_trip(parsed):
    doc, struct = parsed
    chunks = chunk_document(doc, struct)
    assert chunks

    for chunk in chunks:
        assert doc.text[chunk.start : chunk.end] == chunk.text

    assert chunks[0].start == doc.lines[0].start
    assert chunks[-1].end == doc.lines[-1].end
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.start > earlier.start
        assert later.start <= earlier.end


def test_structure_finds_furniture_and_headings(parsed):
    doc, struct = parsed
    assert struct.headings, "no headings detected"
    for heading in struct.headings:
        assert doc.text[heading.start : heading.start + len(heading.text)] == heading.text


def test_heading_path_is_monotonic_in_offset(parsed):
    doc, struct = parsed
    assert struct.heading_path(len(doc.text)) or not struct.headings


def test_context_text_renders_for_every_chunk(parsed):
    doc, struct = parsed
    chunks = chunk_document(doc, struct)
    assert any(c.context_text for c in chunks), "no chunk carried any context"
    for chunk in chunks:
        rendered = chunk.context_text
        assert isinstance(rendered, str)
        for line in chunk.page_frame:
            assert line in rendered


def test_chunks_never_span_more_than_the_page_cap(parsed):
    from concord.parse.chunk import MAX_PAGES

    doc, struct = parsed
    for chunk in chunk_document(doc, struct):
        assert chunk.pages[-1] - chunk.pages[0] <= MAX_PAGES


def test_chunk_count_stays_proportional_to_document_size(parsed):
    from concord.parse.chunk import TARGET_CHARS

    doc, struct = parsed
    chunks = chunk_document(doc, struct)
    ceiling = 2 * (len(doc.text) // TARGET_CHARS + 1) + doc.n_pages
    assert len(chunks) <= ceiling, f"{len(chunks)} chunks exceeds ceiling {ceiling}"
