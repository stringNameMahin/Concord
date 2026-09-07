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


@dataclass
class Structure:
    """Where context lives in a document, found structurally rather than semantically.

    This layer never interprets what a header means. It only says "these lines
    are page furniture" and "this heading encloses that offset". Turning the
    text into qualifier key-values is the extractor's job, which is what keeps
    the pipeline free of domain rules.
    """

    furniture: set[str]
    page_frames: dict[int, list[str]]
    headings: list[Heading]

    def heading_path(self, offset: int) -> list[str]:
        path: list[Heading] = []
        for heading in self.headings:
            if heading.start > offset:
                break
            while path and path[-1].level >= heading.level:
                path.pop()
            path.append(heading)
        return [h.text for h in path]


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

    shaped = [line for line in doc.lines if _heading_shaped(line, furniture)]
    bigger = [line for line in shaped if line.size >= body * HEADING_SIZE_RATIO]

    keep = sorted({round(line.size, 1) for line in bigger}, reverse=True)[:MAX_SIZE_LEVELS]
    ranks = {size: rank for rank, size in enumerate(keep)}
    headings = [
        Heading(line.text, line.start, line.page, line.size, ranks[round(line.size, 1)])
        for line in bigger
        if round(line.size, 1) in ranks
    ]

    bold_rate = sum(line.bold for line in doc.lines) / len(doc.lines)
    if bold_rate < BOLD_RATE_CEILING:
        deepest = len(ranks)
        headings += [
            Heading(line.text, line.start, line.page, line.size, deepest)
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
