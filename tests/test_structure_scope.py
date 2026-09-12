"""Bullet-scoped context: which list items condition the facts beneath them.

A bullet opens a scope, and every figure under it belongs to whatever the
bullet names. The context stack already knew where headings were; these tests
pin the part that decides which of them are labels rather than sentences, and
the part that turns a label into an inherited qualifier.

Every document here is synthetic. Nothing in the suite may encode a line from
the starter corpus - see tests/factories.py.
"""

import pytest

from concord.facts import build_qualifiers
from concord.parse.pdf import Line, Page, ParsedDoc
from concord.parse.structure import (
    Heading,
    Structure,
    analyse,
    bullet_starts,
    scope_label,
)


def document(lines, size=10.0, page_height=800.0):
    """A ParsedDoc from (text, bold, y) triples, laid out like a parsed page."""
    text_parts = []
    built = []
    cursor = 0
    for item in lines:
        body, bold, y = item
        built.append(
            Line(page=0, text=body, start=cursor, end=cursor + len(body),
                 size=size, bold=bold, y=y)
        )
        text_parts.append(body + "\n")
        cursor += len(body) + 1
    text = "".join(text_parts)
    return ParsedDoc(
        sha256="x",
        filename="synthetic.pdf",
        text=text,
        lines=built,
        pages=[Page(index=0, start=0, end=len(text), width=600.0, height=page_height)],
    )


# --- which bulleted lines name a scope ---------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Express Parcel",
        "Supply Chain Services",
        "Cross-Border Services",
        "Fleet Management System (\u201cFMS\u201d)",
    ],
)
def test_a_short_title_cased_phrase_names_a_scope(text):
    assert scope_label(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "17,000 freight units per hour",          # carries its own figure
        "Total headcount increased by 11%",       # a sentence with a number in it
        "Reducing food and cutlery waste",        # sentence case, not a label
        "Independent Audits: Verified through",   # a clause, not a name
        "A rather long bulleted clause about several different things at once",
    ],
)
def test_a_bulleted_sentence_does_not_name_a_scope(text):
    assert scope_label(text) is None


def test_a_leading_bullet_glyph_is_stripped_from_the_label():
    assert scope_label("\u2022 Express Parcel") == "Express Parcel"


def test_short_function_words_may_stay_lower_case():
    assert scope_label("Supply Chain and Logistics") == "Supply Chain and Logistics"


# --- finding the bullets -----------------------------------------------------

def test_a_bullet_on_its_own_line_opens_the_line_below_it():
    doc = document([("\u2022", False, 100.0), ("Express Parcel", True, 101.0)])
    assert doc.lines[1].start in bullet_starts(doc)


def test_a_bullet_sharing_its_line_opens_that_line():
    doc = document([("\u2022 Express Parcel", True, 100.0)])
    assert doc.lines[0].start in bullet_starts(doc)


def test_a_bullet_far_above_a_line_does_not_open_it():
    """A glyph and its text are one typographic line, so the gap is bounded."""
    doc = document([("\u2022", False, 100.0), ("Express Parcel", True, 400.0)])
    assert bullet_starts(doc) == set()


def test_a_lone_hyphen_is_a_table_cell_not_a_bullet():
    doc = document([("-", False, 100.0), ("Express Parcel", True, 101.0)])
    assert bullet_starts(doc) == set()


def test_a_symbol_font_bullet_counts():
    """Wingdings bullets reach the parser as private-use code points."""
    doc = document([("\uf09f", False, 100.0), ("Express Parcel", True, 101.0)])
    assert doc.lines[1].start in bullet_starts(doc)


# --- inheriting the scope ----------------------------------------------------

def scoped_document():
    return document(
        [
            ("\u2022", False, 100.0),
            ("Express Parcel", True, 101.0),
            ("We served nine hundred customers in the period under review.", False, 120.0),
            ("\u2022", False, 140.0),
            ("Supply Chain Services", True, 141.0),
            ("We served four hundred customers in the period under review.", False, 160.0),
        ]
    )


def test_a_fact_inherits_the_bullet_it_sits_under():
    doc = scoped_document()
    found = analyse(doc)
    assert found.inherited_qualifiers(doc.lines[2].start) == {"segment": "Express Parcel"}


def test_the_next_bullet_replaces_the_previous_one():
    """The scope is resolved at the offset, so it moves with the document."""
    doc = scoped_document()
    found = analyse(doc)
    assert found.inherited_qualifiers(doc.lines[5].start) == {
        "segment": "Supply Chain Services"
    }


def test_an_offset_under_no_bullet_inherits_nothing():
    doc = document([("Some ordinary prose with no list around it at all.", False, 100.0)])
    assert analyse(doc).inherited_qualifiers(doc.lines[0].start) == {}


def test_an_unbulleted_heading_is_not_promoted_to_a_qualifier():
    """A prose heading is as often a rhetorical label as it is a scope."""
    doc = document(
        [
            ("Strong Relationships With Customers", True, 100.0),
            ("We served nine hundred customers in the period under review.", False, 120.0),
        ]
    )
    assert analyse(doc).inherited_qualifiers(doc.lines[1].start) == {}


def test_the_innermost_bulleted_label_wins():
    structure = Structure(
        furniture=set(),
        page_frames={},
        headings=[
            Heading("Express Parcel", 0, 0, 10.0, 0, bulleted=True),
            Heading("Domestic Lane", 50, 0, 10.0, 1, bulleted=True),
        ],
    )
    assert structure.inherited_qualifiers(100) == {"segment": "Domestic Lane"}


def test_a_bulleted_sentence_falls_through_to_the_label_above_it():
    structure = Structure(
        furniture=set(),
        page_frames={},
        headings=[
            Heading("Express Parcel", 0, 0, 10.0, 0, bulleted=True),
            Heading("Volumes grew by 11% year on year", 50, 0, 10.0, 1, bulleted=True),
        ],
    )
    assert structure.inherited_qualifiers(100) == {"segment": "Express Parcel"}


# --- what an inherited qualifier may and may not do --------------------------

def test_an_inherited_qualifier_fills_a_key_the_extractor_left_empty():
    bag = build_qualifiers([{"key": "period", "value": "FY24"}],
                           inherited={"segment": "Express Parcel"})
    assert bag["segment"].value == "Express Parcel"
    assert bag["segment"].provenance == "inherited"
    assert bag["segment"].known


def test_the_sentence_beats_the_layout_where_both_state_a_key():
    bag = build_qualifiers(
        [{"key": "segment", "value": "Cross-Border Services"}],
        inherited={"segment": "Express Parcel"},
    )
    assert bag["segment"].value == "Cross-Border Services"
    assert bag["segment"].provenance == "stated"


# --- the consolidation basis, promoted from title or running header ----------

def basis_structure(headings=(), frames=None):
    return Structure(furniture=set(), page_frames=frames or {}, headings=list(headings))


def test_a_statement_title_names_the_basis_it_reports():
    """A company files the same line items twice under the same subject, the
    same predicate and the same year. The statement's own title is the nearest
    thing that says which set they are."""
    structure = basis_structure(
        [Heading("Consolidated Statement of Cash Flows", 0, 207, 12.0, 0)]
    )
    assert structure.consolidation_at(100) == "consolidated"
    assert structure.inherited_qualifiers(100) == {"consolidation": "consolidated"}


def test_the_running_header_carries_the_basis_where_the_title_does_not():
    """The notes run for a hundred pages after the statement they belong to, so
    the page frame is what reaches them."""
    structure = basis_structure(frames={274: ["Financial Statements: Standalone", "94"]})
    assert structure.consolidation_at(100, page=274) == "standalone"
    assert structure.inherited_qualifiers(100, page=274) == {"consolidation": "standalone"}


def test_the_nearer_title_beats_the_running_header():
    structure = basis_structure(
        [Heading("Standalone Balance Sheet", 0, 313, 12.0, 0)],
        frames={313: ["Financial Statements: Consolidated"]},
    )
    assert structure.consolidation_at(100, page=313) == "standalone"


def test_a_line_naming_both_bases_names_neither():
    """`Consolidated and Standalone Financial Statements` is a contents entry,
    not a scope. Guessing which half applies is worse than abstaining."""
    structure = basis_structure(
        [Heading("Consolidated and Standalone Financial Statements", 0, 3, 12.0, 0)]
    )
    assert structure.consolidation_at(100) is None
    assert structure.inherited_qualifiers(100) == {}


def test_a_document_that_never_says_gets_no_basis():
    structure = basis_structure([Heading("Management Discussion", 0, 12, 12.0, 0)])
    assert structure.consolidation_at(100, page=12) is None


def test_the_basis_and_a_bullet_segment_can_both_be_inherited():
    """They are different dimensions and neither displaces the other."""
    structure = basis_structure(
        [
            Heading("Consolidated Statement of Profit and Loss", 0, 201, 12.0, 0),
            Heading("Express Parcel", 50, 201, 10.0, 1, bulleted=True),
        ]
    )
    assert structure.inherited_qualifiers(100) == {
        "consolidation": "consolidated",
        "segment": "Express Parcel",
    }


def test_the_inherited_basis_is_marked_as_ours():
    bag = build_qualifiers(
        [{"key": "period", "value": "year ended March 31, 2026"}],
        inherited={"consolidation": "consolidated"},
    )
    assert bag["consolidation"].value == "consolidated"
    assert bag["consolidation"].provenance == "inherited"
