import re
from dataclasses import dataclass

from concord.parse.pdf import Line, ParsedDoc

DIGITS = re.compile(r"\d+")
WS = re.compile(r"\s+")

TOP_BAND = 0.18
BOTTOM_BAND = 0.88
MIN_REPEAT_PAGES = 3
REPEAT_FRACTION = 0.25
HEADING_SIZE_RATIO = 1.15
HEADING_MAX_CHARS = 80
HEADING_MAX_WORDS = 12
MAX_SIZE_LEVELS = 3
BOLD_RATE_CEILING = 0.40

# Glyphs a document uses to open a scoped list item. Symbol and Wingdings
# bullets arrive as private-use code points rather than as U+2022, so the
# range is accepted alongside the literal characters.
#
# `-` and `*` are deliberately absent even though both are bullets in plain
# text. In a parsed PDF a line that is nothing but a hyphen is almost always a
# table cell standing for nil - 3,648 of them across the shipped corpus, against
# 263 real bullets - and reading those as list openers would attach a heading
# to the wrong facts.
BULLETS = frozenset(
    "\u2022\u2023\u25aa\u25ab\u25b8\u25b9\u25cb\u25cf\u25e6\u2043\u00b7"
)
PRIVATE_USE = (0xE000, 0xF8FF)

# A bullet glyph and the phrase it opens are one line typographically but two
# lines in the parse, because PyMuPDF splits on the horizontal gap between
# them. They belong together only if they sit on the same text line, so the
# vertical distance is bounded rather than ignored.
BULLET_MAX_DY = 6.0

# A bulleted line only names a scope if it reads like a label rather than a
# sentence. Three typographic tests, no vocabulary: short, no figures in it,
# and title-cased. See `scope_label`.
SCOPE_MAX_WORDS = 6
SCOPE_MIN_CASED_WORD = 4


def shape(text: str) -> str:
    """Collapse a line to a repetition-comparable form.

    Page numbers and dates vary from page to page while the surrounding running
    header does not, so digits are masked before counting.
    """
    return WS.sub(" ", DIGITS.sub("#", text.lower())).strip()


@dataclass
class Heading:
    text: str
    start: int
    page: int
    size: float
    level: int
    bulleted: bool = False


@dataclass
class Structure:
    """Where context lives in a document, found structurally rather than semantically.

    This layer finds context by typography and repetition, never by reading it.
    It says "these lines are page furniture" and "this heading encloses that
    offset"; what a heading *means* is the extractor's job, which is what keeps
    the pipeline free of domain rules.

    `inherited_qualifiers` is the one exception, and a narrow one: a bulleted
    heading is promoted to a `segment` qualifier on the facts beneath it. That
    names a structural relation - this bullet scopes those figures - rather
    than interpreting the words, and it is the only reading this layer does.
    """

    furniture: set[str]
    page_frames: dict[int, list[str]]
    headings: list[Heading]

    def enclosing(self, offset: int) -> list[Heading]:
        """The heading stack open at an offset, outermost first."""
        path: list[Heading] = []
        for heading in self.headings:
            if heading.start > offset:
                break
            while path and path[-1].level >= heading.level:
                path.pop()
            path.append(heading)
        return path

    def heading_path(self, offset: int) -> list[str]:
        return [h.text for h in self.enclosing(offset)]

    def inherited_qualifiers(self, offset: int) -> dict[str, str]:
        """Context this offset sits inside, as qualifier key-values.

        A bullet opens a scope. A bullet glyph followed by `PTL Freight` heads
        a run of prose in
        which every unqualified figure is a PTL figure, exactly as a table
        header conditions the cells beneath it - and unlike a prose heading,
        which is as often a rhetorical label ("Strong relationships with a
        diverse customer base") as it is a scope. So the bulleted headings are
        the ones promoted to a qualifier and the rest are not.

        The key is `segment` because that is what the bullet is naming: the
        business line the figures below it belong to. That single naming
        decision is the only interpretation this layer makes, and it is
        recorded as `inherited` so a reader can always see it was ours rather
        than the document's. The value is the heading verbatim; nothing here
        parses or classifies it.

        Only the innermost bulleted heading that reads as a label is used. An
        outer bullet a deeper one has already narrowed is not a second, coarser
        condition on the same figure, and a bulleted sentence is not a
        condition at all - see `scope_label`.
        """
        for heading in reversed(self.enclosing(offset)):
            if not heading.bulleted:
                continue
            label = scope_label(heading.text)
            if label:
                return {"segment": label}
        return {}


def _in_band(line: Line, doc: ParsedDoc) -> bool:
    page = doc.pages[line.page]
    if not page.height:
        return False
    ratio = line.y / page.height
    return ratio <= TOP_BAND or ratio >= BOTTOM_BAND


def find_furniture(doc: ParsedDoc) -> set[str]:
    seen: dict[str, set[int]] = {}
    for line in doc.lines:
        if not _in_band(line, doc):
            continue
        key = shape(line.text)
        if len(key) < 4:
            continue
        seen.setdefault(key, set()).add(line.page)

    threshold = max(MIN_REPEAT_PAGES, int(REPEAT_FRACTION * doc.n_pages))
    return {key for key, pages in seen.items() if len(pages) >= threshold}


def build_page_frames(doc: ParsedDoc, furniture: set[str]) -> dict[int, list[str]]:
    frames: dict[int, list[str]] = {}
    for line in doc.lines:
        if shape(line.text) in furniture and _in_band(line, doc):
            bucket = frames.setdefault(line.page, [])
            if line.text not in bucket:
                bucket.append(line.text)
    return frames


def scope_label(text: str) -> str | None:
    """The heading, if it names a scope; None if it is a bulleted sentence.

    Documents bullet two different things. One is a label - `PTL Freight`,
    `Supply Chain Services` - which scopes every figure under it. The other is
    a body sentence that happens to sit in a list: `70% of load moved through
    46-ft trailers`. Only the first conditions the facts below it, and reading
    the second as a condition would attach a whole paragraph of text to a fact
    as though the document had qualified it.

    Three tests separate them, and all three are typographic:

    - **no digits.** A business line is named, not measured. A bulleted
      sentence almost always carries the figure it is making a point about.
    - **short.** Six words is a label; more is a clause.
    - **title-cased.** Every word long enough to be a content word starts
      upper. Short function words are exempt, so `Supply Chain and Services`
      still reads as a label.

    Measured over the seven documents in the ledger, this accepts six labels -
    the prospectus's five business lines and its platform name - and rejects
    nineteen bulleted sentences, with nothing misclassified either way. A
    label that fails a test costs one inherited qualifier; a sentence that
    passed one would put a false condition on a fact, so the tests are set to
    fail towards silence.
    """
    cleaned = "".join(c for c in text if c.isprintable()).strip()
    cleaned = cleaned.lstrip("".join(BULLETS)).strip()
    if not cleaned or any(c.isdigit() for c in cleaned):
        return None

    words = cleaned.split()
    if not words or len(words) > SCOPE_MAX_WORDS:
        return None

    for word in words:
        letters = [c for c in word if c.isalpha()]
        if len(letters) >= SCOPE_MIN_CASED_WORD and not letters[0].isupper():
            return None
    return cleaned


def _is_bullet(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 2:
        return False
    return all(
        c in BULLETS or PRIVATE_USE[0] <= ord(c) <= PRIVATE_USE[1] for c in stripped
    )


def bullet_starts(doc: ParsedDoc) -> set[int]:
    """Offsets of lines a bullet glyph opens.

    Two shapes, both common: the glyph shares the line with its text, and the
    glyph is a line of its own immediately above it. The second is what
    PyMuPDF produces whenever the gap between glyph and text is wide enough to
    break the line, which is most of the time.
    """
    starts: set[int] = set()
    previous: Line | None = None
    for line in doc.lines:
        head = line.text.strip()[:1]
        if head and _is_bullet(head) and len(line.text.strip()) > 1:
            starts.add(line.start)
        elif (
            previous is not None
            and _is_bullet(previous.text)
            and previous.page == line.page
            and abs(line.y - previous.y) <= BULLET_MAX_DY
        ):
            starts.add(line.start)
        previous = line
    return starts


def _heading_shaped(line: Line, furniture: set[str]) -> bool:
    text = line.text.rstrip()
    return (
        shape(line.text) not in furniture
        and len(text) <= HEADING_MAX_CHARS
        and any(c.isalpha() for c in text)
        and not text.endswith((".", ",", ";", ":"))
        and len(text.split()) <= HEADING_MAX_WORDS
    )


def find_headings(doc: ParsedDoc, furniture: set[str]) -> list[Heading]:
    """Headings from typography alone.

    No single signal works across the corpus: a prospectus marks headings with
    bold at body size and never changes size, while a slide deck sets almost
    every line larger than its tiny body size. So take larger type limited to
    the few largest ranks, union bold-at-body-size, and ignore bold entirely in
    documents that use it for table labels.
    """
    body = doc.body_size()
    if not body:
        return []

    bullets = bullet_starts(doc)
    shaped = [line for line in doc.lines if _heading_shaped(line, furniture)]
    bigger = [line for line in shaped if line.size >= body * HEADING_SIZE_RATIO]

    keep = sorted({round(line.size, 1) for line in bigger}, reverse=True)[:MAX_SIZE_LEVELS]
    ranks = {size: rank for rank, size in enumerate(keep)}
    headings = [
        Heading(
            line.text,
            line.start,
            line.page,
            line.size,
            ranks[round(line.size, 1)],
            line.start in bullets,
        )
        for line in bigger
        if round(line.size, 1) in ranks
    ]

    bold_rate = sum(line.bold for line in doc.lines) / len(doc.lines)
    if bold_rate < BOLD_RATE_CEILING:
        deepest = len(ranks)
        headings += [
            Heading(line.text, line.start, line.page, line.size, deepest, line.start in bullets)
            for line in shaped
            if line.bold and abs(line.size - body) < 1.0
        ]

    headings.sort(key=lambda h: h.start)
    return headings


def analyse(doc: ParsedDoc) -> Structure:
    furniture = find_furniture(doc)
    return Structure(
        furniture=furniture,
        page_frames=build_page_frames(doc, furniture),
        headings=find_headings(doc, furniture),
    )
