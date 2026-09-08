import pytest

from concord.extract.align import Aligner
from concord.extract.models import ExtractionOut, FactOut, ValueOut
from concord.extract.runner import batches, extract
from concord.parse.chunk import Chunk

TEXT = (
    "Revenue from services 81,415.38 for the year ended March 31, 2024.\n"
    "PADDING " * 400
    + "\nActive customers exceeded 33,200 during the period.\n"
)


def chunk_at(index, needle):
    start = TEXT.index(needle)
    end = start + len(needle)
    return Chunk(index, start, end, TEXT[start:end], [0], [], [])


def fact(passage_id, quote, predicate="revenue_from_services"):
    return FactOut(
        passage_id=passage_id,
        claim_text="c",
        subject_surface="Acme",
        predicate=predicate,
        value=ValueOut(kind="quantity", raw="81,415.38"),
        quote=quote,
        confidence=0.9,
    )


class FakeClient:
    def __init__(self, facts):
        self.facts = facts
        self.calls = 0

    def complete(self, prompt, schema, system=None, temperature=0.0):
        self.calls += 1
        return ExtractionOut(facts=self.facts)


def test_batches_cover_every_chunk_exactly_once():
    items = list(range(17))
    out = list(batches(items, 8))
    assert [len(b) for b in out] == [8, 8, 1]
    assert [x for b in out for x in b] == items


def test_grounded_fact_round_trips_to_source_bytes():
    c = chunk_at(0, "Revenue from services 81,415.38")
    run = extract([c], Aligner(TEXT), FakeClient([fact(0, "Revenue from services 81,415.38")]))

    assert len(run.grounded) == 1 and not run.quarantined
    got = run.grounded[0].alignment
    assert TEXT[got.start : got.end] == got.text


def test_invented_quote_is_quarantined_not_repaired():
    c = chunk_at(0, "Revenue from services 81,415.38")
    run = extract([c], Aligner(TEXT), FakeClient([fact(0, "Revenue was 99,999.00 crore")]))

    assert not run.grounded and len(run.quarantined) == 1
    assert run.quarantined[0].alignment.status == "unlocated"
    assert run.quarantine_rate == 1.0


def test_quote_from_a_different_passage_is_quarantined():
    """The batching safety property: misattributed evidence must not survive."""
    near = chunk_at(0, "Revenue from services 81,415.38")
    far = chunk_at(1, "Active customers exceeded 33,200")
    run = extract(
        [near, far],
        Aligner(TEXT),
        FakeClient([fact(0, "Active customers exceeded 33,200")]),
    )

    assert not run.grounded
    assert run.quarantined[0].alignment.status == "unlocated"


def test_unknown_passage_id_is_counted_as_orphaned():
    c = chunk_at(0, "Revenue from services 81,415.38")
    run = extract([c], Aligner(TEXT), FakeClient([fact(99, "Revenue from services 81,415.38")]))

    assert run.orphaned == 1 and run.total == 0


def test_one_request_per_batch_not_per_chunk():
    chunks = [chunk_at(i, "Revenue from services 81,415.38") for i in range(8)]
    client = FakeClient([])
    run = extract(chunks, Aligner(TEXT), client, size=8)

    assert client.calls == 1 and run.requests == 1


def test_one_failed_batch_does_not_lose_the_document():
    """Observed live: a model ran past its token ceiling and returned JSON that
    ended mid-string. That batch is lost; the rest of the document is not."""
    from concord.llm.client import LLMError

    class FlakyClient:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt, schema, system=None, temperature=0.0):
            self.calls += 1
            if self.calls == 1:
                raise LLMError("model returned JSON that does not fit ExtractionOut")
            return ExtractionOut(facts=[fact(1, "Revenue from services 81,415.38")])

    aligner = Aligner(TEXT)
    chunks = [chunk_at(0, "PADDING"), chunk_at(1, "Revenue from services 81,415.38")]
    run = extract(chunks, aligner, FlakyClient(), size=1, workers=1)

    assert run.failed_batches == 1
    assert run.requests == 2
    assert len(run.grounded) == 1
    assert "1 batches failed" in run.summary()


def test_batches_run_concurrently_without_losing_or_duplicating_any():
    import threading

    seen, lock = [], threading.Lock()

    class CountingClient:
        def complete(self, prompt, schema, system=None, temperature=0.0):
            with lock:
                seen.append(prompt)
            return ExtractionOut(facts=[])

    chunks = [chunk_at(i, "PADDING") for i in range(12)]
    run = extract(chunks, Aligner(TEXT), CountingClient(), size=3, workers=4)

    assert run.requests == 4
    assert len(seen) == 4
    assert len(set(seen)) == 4


def test_a_total_failure_is_distinguishable_from_a_document_with_no_facts():
    """A working system finding nothing and a broken one look identical from
    the outside unless the run says which happened."""
    from concord.llm.client import LLMError

    class Dead:
        def complete(self, *a, **k):
            raise LLMError("no API key; set GEMINI_API_KEY or run with CONCORD_OFFLINE=1")

    class Silent:
        def complete(self, *a, **k):
            return ExtractionOut(facts=[])

    chunks = [chunk_at(i, "PADDING") for i in range(4)]

    dead = extract(chunks, Aligner(TEXT), Dead(), size=2, workers=1)
    assert dead.total_failure is True
    assert dead.failed_batches == 2
    assert "no API key" in dead.failures[0]

    silent = extract(chunks, Aligner(TEXT), Silent(), size=2, workers=1)
    assert silent.total_failure is False
    assert silent.failures == []


def test_identical_failures_are_recorded_once():
    from concord.llm.client import LLMError

    class Dead:
        def complete(self, *a, **k):
            raise LLMError("the same problem every time")

    run = extract([chunk_at(i, "PADDING") for i in range(6)], Aligner(TEXT), Dead(), size=1, workers=1)
    assert run.failed_batches == 6
    assert len(run.failures) == 1
