"""One document in, grounded facts out.

The straight line the whole system is: parse to one canonical string, find
where context lives, window it, extract, ground every quote, then materialise
what survived. Phase 6's `POST /ingest` is this function with an HTTP wrapper
around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from concord.config import BATCH_SIZE, WORKERS
from concord.extract.align import Aligner
from concord.extract.runner import ExtractionRun, extract
from concord.facts import Fact, materialize
from concord.normalize.periods import infer_fiscal_year_end
from concord.parse.chunk import Chunk, chunk_document
from concord.parse.pdf import ParsedDoc, parse
from concord.parse.structure import Structure, analyse

# Qualifier keys that, when the extractor emits them, describe the figure's
# units rather than the conditions it holds under. Read by exact name from the
# model's own structured output - not pattern-matched out of the document.
SCALE_KEYS = ("scale", "magnitude")
UNIT_KEYS = ("currency", "unit", "units")


@dataclass
class Ingested:
    doc: ParsedDoc
    structure: Structure
    chunks: list[Chunk]
    extraction: ExtractionRun
    facts: list[Fact] = field(default_factory=list)
    fy_end_month: int | None = None
    duplicates: int = 0

    @property
    def doc_id(self) -> str:
        return Path(self.doc.filename).stem

    def summary(self) -> str:
        return (
            f"{self.doc_id}: {self.doc.n_pages} pages, {len(self.chunks)} chunks, "
            f"{self.extraction.summary()}, {len(self.facts)} facts materialised, "
            f"{self.duplicates} duplicates collapsed"
        )


def _default_units(fact) -> tuple[str | None, str | None]:
    """Let an inherited units qualifier supply what a bare cell omits.

    A table header saying the column is in millions reaches the extractor as a
    qualifier, not as part of the figure. Without this, `81,415.38` under such
    a header would normalise to eighty-one thousand.

    This depends on the model coining one of a handful of key names, and over
    the seven-document ledger it coined `scale` or `magnitude` exactly **zero**
    times out of 773 facts. It is kept because it costs nothing and is right
    when it fires, but it is not what makes a bare cell work. What does is
    `materialize` reading the magnitude back out of the bytes the aligner
    located - see F1 in docs/devRead.md section 8, and see that entry for why
    the page-level `(All amounts in ... in million)` declaration is *not* read
    here: it is row-scoped in the source, and applying it at page granularity
    turned share counts into millions of shares.
    """
    scale = unit = None
    for qualifier in fact.qualifiers:
        key = qualifier.key.strip().lower()
        if key in SCALE_KEYS and scale is None:
            scale = qualifier.value
        elif key in UNIT_KEYS and unit is None:
            unit = qualifier.value
    return scale, unit


def dedupe(facts: list[Fact]) -> tuple[list[Fact], int]:
    """Collapse facts that share an id, keeping the best-qualified one.

    A fact id is the span plus the predicate, so two records sharing one are
    the same claim stated twice - the extractor emitting the same table cell on
    two passes, once with the qualifier it inherited and once without. Left
    alone they would collide in every id-keyed structure downstream and one
    would vanish silently; worse, the impoverished copy would drag pairs into
    `insufficient_context` through the missing-qualifier guard. Keeping the
    richer bag is both the safer and the more truthful choice.
    """
    best: dict[str, Fact] = {}
    dropped = 0
    for fact in facts:
        seen = best.get(fact.fact_id)
        if seen is None:
            best[fact.fact_id] = fact
            continue
        dropped += 1
        if len(fact.qualifier_keys()) > len(seen.qualifier_keys()):
            best[fact.fact_id] = fact
    return list(best.values()), dropped


def ingest(
    path: str | Path,
    client,
    size: int = BATCH_SIZE,
    doc_id: str | None = None,
    workers: int = WORKERS,
) -> Ingested:
    """Parse, extract and ground one document.

    Only grounded facts are materialised. Quarantined ones stay on the
    extraction run, counted and inspectable, and never enter adjudication.
    """
    doc = parse(path)
    structure = analyse(doc)
    chunks = chunk_document(doc, structure)
    aligner = Aligner(doc.text)

    run = extract(chunks, aligner, client, size, workers)
    fy_end_month = infer_fiscal_year_end(doc.text)
    identifier = doc_id or Path(doc.filename).stem

    facts = []
    for record in run.grounded:
        scale, unit = _default_units(record.fact)
        facts.append(
            materialize(
                record.fact,
                identifier,
                record.alignment,
                page=doc.page_of(record.alignment.start),
                fy_end_month=fy_end_month,
                default_scale=scale,
                default_unit=unit,
                # The context stack is walked at the fact's own offset, not the
                # chunk's. A chunk is four thousand characters wide and can
                # cross several bullet sections, so the enclosing scope has to
                # be resolved per fact or every fact in the window inherits
                # whichever scope happened to be open where the window started.
                inherited=structure.inherited_qualifiers(record.alignment.start),
            )
        )

    unique, duplicates = dedupe(facts)
    return Ingested(
        doc=doc,
        structure=structure,
        chunks=chunks,
        extraction=run,
        facts=unique,
        fy_end_month=fy_end_month,
        duplicates=duplicates,
    )
