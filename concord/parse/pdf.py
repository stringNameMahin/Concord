import hashlib
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

PARSER = "pymupdf"
PARSER_VERSION = pymupdf.__doc__.strip() if pymupdf.__doc__ else "unknown"
PAGE_SEP = "\n\n"
BOLD_FLAG = 1 << 4


class UnreadablePDF(ValueError):
    """The bytes handed to `parse` are not a PDF this parser can open.

    Deliberately distinct from an extraction failure. Nothing upstream has
    been asked anything yet, so the caller's file is the problem and the
    HTTP surface can say which file instead of blaming a provider.
    """


@dataclass
class Line:
    page: int
    text: str
    start: int
    end: int
    size: float
    bold: bool
    y: float


@dataclass
class Page:
    index: int
    start: int
    end: int
    width: float
    height: float


@dataclass
class ParsedDoc:
    """A document flattened to one canonical string plus a line index into it.

    Every offset in the system points into `text`. Nothing downstream reopens
    the PDF, so grounding is a pure string operation and the round-trip
    guarantee holds by construction.
    """

    sha256: str
    filename: str
    text: str
    lines: list[Line]
    pages: list[Page]
    meta: dict = field(default_factory=dict)
    parser: str = PARSER
    parser_version: str = PARSER_VERSION

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    def page_of(self, offset: int) -> int:
        for page in self.pages:
            if page.start <= offset < page.end:
                return page.index
        return self.pages[-1].index if self.pages else 0

    def body_size(self) -> float:
        weights: dict[float, int] = {}
        for line in self.lines:
            weights[round(line.size, 1)] = weights.get(round(line.size, 1), 0) + len(line.text)
        return max(weights, key=weights.get) if weights else 0.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse(path: str | Path) -> ParsedDoc:
    path = Path(path)
    data = path.read_bytes()

    # Opened from bytes, never from the path. PyMuPDF holds an OS file handle
    # on a Document whose construction then fails, and the traceback of the
    # error it raises keeps that frame - and so the handle - alive for as long
    # as any caller holds the exception. A caller that wraps it inside a
    # TemporaryDirectory (`raise HTTPException(...) from exc`) therefore cannot
    # delete the file on Windows, and the unlink error replaces the real one.
    # No handle, no problem. See docs/decisions.md.
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise UnreadablePDF(f"{path.name} is not a readable PDF: {exc}") from exc

    # An encrypted file opens cleanly and only fails later, inside the page
    # loop, when `get_text` touches content it is not authenticated for. That
    # failure used to leave the endpoint - which cannot tell one exception from
    # another this far down - answering 502, blaming the provider for a problem
    # with the caller's file. Checked here, where the handle is known good, it
    # rides the same path as any other unreadable upload. See bug 24 in
    # docs/status.md.
    if doc.needs_pass:
        doc.close()
        raise UnreadablePDF(
            f"{path.name} is password-protected and no password was supplied"
        )

    buffer: list[str] = []
    cursor = 0
    lines: list[Line] = []
    pages: list[Page] = []

    try:
        for index, page in enumerate(doc):
            page_start = cursor
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for raw in block["lines"]:
                    spans = raw.get("spans", [])
                    text = "".join(s["text"] for s in spans).strip()
                    if not text:
                        continue
                    buffer.append(text)
                    lines.append(
                        Line(
                            page=index,
                            text=text,
                            start=cursor,
                            end=cursor + len(text),
                            size=max((s["size"] for s in spans), default=0.0),
                            bold=any(s.get("flags", 0) & BOLD_FLAG for s in spans),
                            y=raw["bbox"][1],
                        )
                    )
                    cursor += len(text) + 1
                    buffer.append("\n")

            buffer.append(PAGE_SEP)
            cursor += len(PAGE_SEP)
            rect = page.rect
            pages.append(
                Page(
                    index=index,
                    start=page_start,
                    end=cursor,
                    width=rect.width,
                    height=rect.height,
                )
            )

        text = "".join(buffer)
        meta = {k: v for k, v in (doc.metadata or {}).items() if v}
    finally:
        doc.close()

    return ParsedDoc(
        sha256=hashlib.sha256(data).hexdigest(),
        filename=path.name,
        text=text,
        lines=lines,
        pages=pages,
        meta=meta,
    )
