from dataclasses import dataclass

from concord.parse.pdf import ParsedDoc
from concord.parse.structure import Structure

TARGET_CHARS = 4000
OVERLAP_CHARS = 500
MAX_PAGES = 2


@dataclass
class Chunk:
    """A window of document text plus the context that encloses it.

    `context_text` is the raw enclosing material - running header, heading path.
    It is passed to the extractor verbatim rather than parsed here, so that
    qualifiers stay an open vocabulary driven by the document.
    """

    index: int
    start: int
    end: int
    text: str
    pages: list[int]
    page_frame: list[str]
    heading_path: list[str]

    @property
    def context_text(self) -> str:
        parts = []
        if self.page_frame:
            parts.append("[page header]\n" + "\n".join(self.page_frame))
        if self.heading_path:
            parts.append("[section]\n" + " > ".join(self.heading_path))
        return "\n\n".join(parts)


def chunk_document(
    doc: ParsedDoc,
    structure: Structure,
    target: int = TARGET_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> list[Chunk]:
    if not doc.lines:
        return []

    chunks: list[Chunk] = []
    cursor = 0
    index = 0

    while cursor < len(doc.lines):
        start_line = doc.lines[cursor]
        end_line = start_line
        walker = cursor
        while walker < len(doc.lines):
            end_line = doc.lines[walker]
            if end_line.end - start_line.start >= target:
                break
            if end_line.page - start_line.page >= MAX_PAGES:
                break
            walker += 1

        start, end = start_line.start, end_line.end
        pages = sorted({line.page for line in doc.lines[cursor : walker + 1]})
        frame: list[str] = []
        for page in pages:
            for text in structure.page_frames.get(page, []):
                if text not in frame:
                    frame.append(text)

        chunks.append(
            Chunk(
                index=index,
                start=start,
                end=end,
                text=doc.text[start:end],
                pages=pages,
                page_frame=frame,
                heading_path=structure.heading_path(start),
            )
        )
        index += 1

        if walker >= len(doc.lines) - 1:
            break

        # A page-capped chunk can be shorter than the overlap, so stepping back
        # blindly would subdivide forever instead of advancing.
        back = end - overlap
        if back <= start:
            cursor = walker
            continue

        step = walker
        while step > cursor + 1 and doc.lines[step].start > back:
            step -= 1
        cursor = step

    return chunks
