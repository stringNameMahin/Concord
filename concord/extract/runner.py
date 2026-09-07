from dataclasses import dataclass, field

from concord.extract import prompt
from concord.extract.align import Aligner, Alignment
from concord.extract.models import ExtractionOut, FactOut
from concord.llm.client import LLMClient
from concord.parse.chunk import Chunk

BATCH_SIZE = 8


@dataclass
class Extracted:
    fact: FactOut
    chunk: Chunk
    alignment: Alignment

    @property
    def grounded(self) -> bool:
        return self.alignment.located


@dataclass
class ExtractionRun:
    grounded: list[Extracted] = field(default_factory=list)
    quarantined: list[Extracted] = field(default_factory=list)
    requests: int = 0
    orphaned: int = 0

    @property
    def total(self) -> int:
        return len(self.grounded) + len(self.quarantined)

    @property
    def quarantine_rate(self) -> float:
        return len(self.quarantined) / self.total if self.total else 0.0

    def summary(self) -> str:
        return (
            f"{self.total} facts extracted, {len(self.grounded)} grounded, "
            f"{len(self.quarantined)} quarantined "
            f"({self.quarantine_rate:.1%}), {self.requests} requests"
        )


def batches(chunks: list[Chunk], size: int = BATCH_SIZE):
    for start in range(0, len(chunks), size):
        yield chunks[start : start + size]


def extract(
    chunks: list[Chunk],
    aligner: Aligner,
    client: LLMClient,
    size: int = BATCH_SIZE,
) -> ExtractionRun:
    """Extract facts from chunks and ground every quote before accepting it.

    A fact whose quote cannot be found in the passage it names is quarantined
    rather than repaired. That is what makes batching safe: if the model loses
    track of which passage it is reading, the failure shows up as an unlocated
    quote instead of a plausible-looking fact attached to the wrong evidence.
    """
    run = ExtractionRun()

    for group in batches(chunks, size):
        by_id = {chunk.index: chunk for chunk in group}
        passages = [(c.index, c.context_text, c.text) for c in group]

        out = client.complete(
            prompt.build(passages), ExtractionOut, system=prompt.SYSTEM
        )
        run.requests += 1

        for fact in out.facts:
            chunk = by_id.get(fact.passage_id)
            if chunk is None:
                run.orphaned += 1
                continue

            found = aligner.locate(
                fact.quote, near=(chunk.start, chunk.end), strict=True
            )
            record = Extracted(fact=fact, chunk=chunk, alignment=found)
            (run.grounded if record.grounded else run.quarantined).append(record)

    return run
