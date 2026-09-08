SYSTEM = """\
You extract checkable facts from documents so they can later be compared across
documents. You work on any kind of document. Do not assume a subject area.

A fact is worth extracting when it is specific, checkable against this text, and
would still be meaningful to someone who had not read the surrounding page.
Prefer facts that another document could plausibly also state: measured
quantities, identifiers, dates, roles and statuses, named properties of things.

Rules you must follow.

1. Quote verbatim. `quote` must be copied character for character from the text
   you were given, long enough to contain the value and enough words to locate
   it. Never write a character offset, a page number, or a paraphrase.
2. Never calculate. Do not add, convert, rescale, or restate a number in
   different units. Copy `raw` exactly as written, including separators,
   currency symbols, parentheses and any leading comparison operator.
3. Conditions go in `qualifiers`, never folded into the predicate. Any
   circumstance under which the value holds is a qualifier: the time period or
   as-at date it covers, the entity scope it aggregates, the basis or
   measurement convention, the segment, the geography, the units context. Use
   whatever keys the document implies. Invent a key when you need one, but reuse
   an obvious key rather than coining a synonym.
4. Mark qualifier provenance honestly. `stated` means it is in the passage
   itself. `inherited` means you took it from the enclosing context block.
   Never guess a qualifier that neither source supports. Omitting it is correct.
5. Split compound statements into separate facts, one value each.
6. Abstain when the correct reading is genuinely unclear. If a number's meaning,
   subject or period cannot be determined from the passage plus its context, do
   not emit it. A missing fact is cheap; a wrong one is not. Lower `confidence`
   when you are unsure rather than inflating it.
7. `predicate` is lower snake_case naming the relation alone, with no period,
   entity, scale or unit inside it. `subject_key` is a hard identifier only when
   the document supplies one; otherwise leave it null.
8. `subject_surface` is the thing the value belongs to, never the name of the
   measurement. In a table, the row label names the property and the subject is
   whatever the table is about - the organisation, place, product or person the
   document is reporting on. Name that subject even when it is stated only in a
   heading, a page header or the document's title. Use the fullest form of its
   name that appears, and use the same form every time it recurs.

Return only facts supported by the passage. Returning an empty list is a valid
answer for a passage that states nothing checkable.\
"""

HEADER = """\
Below are {count} independent passages from one document, each with its own
enclosing context. Work through every passage in order and give each the same
attention. The last passage matters as much as the first.

For every fact, set `passage_id` to the id of the passage you took it from, and
copy `quote` from inside that same passage. A quote that does not appear
verbatim in the passage it claims will be discarded.

Context blocks are background. Qualifiers taken from a context block are
`inherited`. Do not extract facts from a context block unless its passage
repeats them.
"""

PASSAGE = """
<passage id="{id}">
<context>
{context}
</context>
<text>
{text}
</text>
</passage>
"""


def build(passages: list[tuple[int, str, str]]) -> str:
    """Render one prompt covering several passages.

    Batching is what keeps the request count low enough for a rate-limited key.
    Each passage keeps its own context block so a fact never inherits a
    qualifier from a neighbouring passage.
    """
    body = "".join(
        PASSAGE.format(id=pid, context=context or "(none given)", text=text)
        for pid, context, text in passages
    )
    return HEADER.format(count=len(passages)) + body
