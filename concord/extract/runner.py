from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from concord.extract import prompt
from concord.extract.align import Aligner, Alignment
from concord.extract.models import ExtractionOut, FactOut
from concord.llm.cache import MissingFromCache
from concord.llm.client import LLMClient, LLMError
from concord.parse.chunk import Chunk

BATCH_SIZE = 8

# Batches are independent, so they can be in flight together. This was
# deliberately out of scope while the binding limit was five requests per
# minute, where concurrency buys nothing. Against a per-token provider the
# constraint is latency instead, and a corpus pass goes from hours to minutes.
WORKERS = 6


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
    failed_batches: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def total_failure(self) -> bool:
        """Every request failed, so this document produced nothing at all.

        Worth distinguishing from a document that genuinely states no facts:
        one is a working system finding nothing, the other is a broken one, and
        a caller that cannot tell them apart will report the wrong thing.
        """
        return bool(self.requests) and self.failed_batches == self.requests

    @property
    def total(self) -> int:
        return len(self.grounded) + len(self.quarantined)

    @property
    def quarantine_rate(self) -> float:
        return len(self.quarantined) / self.total if self.total else 0.0

    def summary(self) -> str:
        failed = f", {self.failed_batches} batches failed" if self.failed_batches else ""
        return (
            f"{self.total} facts extracted, {len(self.grounded)} grounded, "
            f"{len(self.quarantined)} quarantined "
            f"({self.quarantine_rate:.1%}), {self.requests} requests{failed}"
        )


def batches(chunks: list[Chunk], size: int = BATCH_SIZE):
    for start in range(0, len(chunks), size):
        yield chunks[start : start + size]


def extract(
    chunks: list[Chunk],
    aligner: Aligner,
    client: LLMClient,
    size: int = BATCH_SIZE,
    workers: int = WORKERS,
) -> ExtractionRun:
    """Extract facts from chunks and ground every quote before accepting it.

    A fact whose quote cannot be found in the passage it names is quarantined
    rather than repaired. That is what makes batching safe: if the model loses
    track of which passage it is reading, the failure shows up as an unlocated
    quote instead of a plausible-looking fact attached to the wrong evidence.

    One batch failing does not lose the document. A response that arrives
    truncated - a real occurrence, when a model runs into its token ceiling
    mid-object - costs those passages and is counted, while every other batch
    still lands.
    """
    run = ExtractionRun()
    groups = list(batches(chunks, size))

    def ask(group: list[Chunk]):
        passages = [(c.index, c.context_text, c.text) for c in group]
        return client.complete(prompt.build(passages), ExtractionOut, system=prompt.SYSTEM)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(_safe(ask), groups))

    for group, (out, failure) in zip(groups, results):
        run.requests += 1
        if out is None:
            run.failed_batches += 1
            if failure and failure not in run.failures:
                run.failures.append(failure)
            continue

        by_id = {chunk.index: chunk for chunk in group}
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


def _safe(call):
    """Turn a batch failure into a `None` result rather than a lost document.

    The reason travels with it. A batch that failed because there is no API key
    and a batch that failed because a response arrived truncated need different
    answers from the caller, and only the exception knows which happened.
    """

    def wrapped(group):
        try:
            return call(group), None
        except (LLMError, MissingFromCache) as exc:
            return None, f"{type(exc).__name__}: {exc}"

    return wrapped
