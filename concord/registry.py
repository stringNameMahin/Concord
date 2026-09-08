"""The emergent schema: predicates the corpus invents, reconciled as it goes.

There is no fixed ontology here and there was never meant to be. `predicate` is
an open string, so a second document says `revenue_from_operations` where the
first said `revenue_from_services`, and a third says `total_revenue`. Nothing
upstream can prevent that - the vocabulary belongs to the documents.

So the registry learns it. Each new predicate is embedded and searched against
what is already known. Below the similarity threshold it is simply a new
predicate. Above it, and only then, the LLM is asked one yes/no question:
are these two names for the same relation? That is the same discipline the
comparison engine uses - filter deterministically, ask a model only about the
residue - applied to the schema instead of to the facts.

Every decision is written to `schema_events`, including the ones where nothing
happened. The timeline of what the layer learned and when is the artifact; a
registry that only recorded its successes would be untestable and unfalsifiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from concord.facts import canonical_predicate

# Cosine above which two predicates are worth one question. Below it they are
# simply different; the threshold exists to bound LLM calls, not to decide
# anything, and a miss here costs recall rather than correctness - the value
# and semantic blocks can still bring the facts together.
ALIAS_THRESHOLD = 0.86


class AliasAnswer(BaseModel):
    same_relation: bool = Field(
        description="True only if the two names denote the same relation, for the same kind of subject"
    )
    reason: str = Field(description="One sentence, naming what makes them the same or different")


SYSTEM = """\
You decide whether two field names denote the same relation. You work on any
kind of document and must not assume a subject area.

Answer `true` only when a value recorded under one name could be recorded under
the other without changing what is being claimed about the subject.

Answer `false` when the names differ in what they measure, over what scope, or
on what basis, however similar the wording. Two names that describe related but
distinct quantities are `false`. If you cannot tell, answer `false`: leaving
two names separate keeps their facts comparable within each name, while merging
them wrongly makes unlike things look like contradictions.\
"""

QUESTION = """\
Do these two names denote the same relation?

A: {left}
B: {right}

They are used in these claims:
A: {left_claim}
B: {right_claim}\
"""


@dataclass
class Entry:
    canonical: str
    aliases: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None
    first_seen_doc: str | None = None
    count: int = 0

    def names(self) -> set[str]:
        return {self.canonical, *self.aliases}


@dataclass
class SchemaEvent:
    """One thing the layer learned, in the order it learned it."""

    event_type: Literal["predicate_registered", "alias_confirmed", "alias_rejected", "predicate_seen"]
    subject: str
    target: str | None = None
    similarity: float | None = None
    decided_by: str = "deterministic"
    doc_id: str | None = None


class PredicateRegistry:
    """Resolves an open predicate vocabulary to canonical names.

    The registry is the only place that decides two predicates are the same,
    and `aliases()` is how that decision reaches the comparison engine - which
    accepts an alias map and otherwise knows nothing about any of this.
    """

    def __init__(
        self,
        entries: list[Entry] | None = None,
        encoder=None,
        client=None,
        threshold: float = ALIAS_THRESHOLD,
    ):
        self.entries: dict[str, Entry] = {e.canonical: e for e in (entries or [])}
        self.encoder = encoder
        self.client = client
        self.threshold = threshold
        self.events: list[SchemaEvent] = []
        self.calls = 0

    # --- reading -----------------------------------------------------------

    def aliases(self) -> dict[str, str]:
        """The alias -> canonical map the comparison engine consults."""
        return {
            alias: entry.canonical
            for entry in self.entries.values()
            for alias in entry.aliases
        }

    def lookup(self, predicate: str) -> Entry | None:
        name = canonical_predicate(predicate)
        for entry in self.entries.values():
            if name in entry.names():
                return entry
        return None

    def __len__(self) -> int:
        return len(self.entries)

    # --- writing -----------------------------------------------------------

    def observe(self, predicate: str, doc_id: str | None = None, claim: str = "") -> SchemaEvent:
        """Place one predicate in the schema, learning from it if it is new."""
        name = canonical_predicate(predicate)
        if not name:
            return self._record(SchemaEvent("predicate_seen", predicate, doc_id=doc_id))

        known = self.lookup(name)
        if known is not None:
            known.count += 1
            return self._record(
                SchemaEvent(
                    "predicate_seen",
                    name,
                    target=known.canonical,
                    decided_by="deterministic",
                    doc_id=doc_id,
                )
            )

        match, similarity = self._nearest(name)
        if match is None or similarity < self.threshold:
            self._register(name, doc_id)
            return self._record(
                SchemaEvent(
                    "predicate_registered",
                    name,
                    similarity=similarity if match else None,
                    decided_by="deterministic",
                    doc_id=doc_id,
                )
            )

        confirmed = self._confirm(name, match, claim)
        if confirmed:
            match.aliases.append(name)
            match.count += 1
            return self._record(
                SchemaEvent(
                    "alias_confirmed", name, match.canonical, similarity, "llm", doc_id
                )
            )

        self._register(name, doc_id)
        return self._record(
            SchemaEvent("alias_rejected", name, match.canonical, similarity, "llm", doc_id)
        )

    def observe_all(self, facts, doc_id: str | None = None) -> list[SchemaEvent]:
        """Walk a document's facts in a stable order.

        Order matters: the first spelling of a relation becomes the canonical
        one, so a run has to be reproducible. Sorting by predicate makes the
        registry independent of the order the extractor happened to emit.
        """
        events = []
        for fact in sorted(facts, key=lambda f: (f.predicate_canonical, f.fact_id)):
            events.append(self.observe(fact.predicate, doc_id, fact.claim_text))
        return events

    # --- internals ---------------------------------------------------------

    def _register(self, name: str, doc_id: str | None) -> Entry:
        entry = Entry(
            canonical=name,
            embedding=self._embed(name),
            first_seen_doc=doc_id,
            count=1,
        )
        self.entries[name] = entry
        return entry

    def _embed(self, name: str) -> np.ndarray | None:
        if self.encoder is None:
            return None
        return self.encoder.encode([name.replace("_", " ")])[0]

    def _nearest(self, name: str) -> tuple[Entry | None, float]:
        """Cosine against every canonical embedding. Brute force is correct and
        fast at this size, and keeps the registry a plain table."""
        vector = self._embed(name)
        if vector is None:
            return None, 0.0

        # Entries embedded by a different model are skipped rather than
        # compared. A stored registry outlives the encoder that built it, and
        # a changed embedding model should cost recall, not raise.
        candidates = [
            e
            for e in self.entries.values()
            if e.embedding is not None and len(e.embedding) == len(vector)
        ]
        if not candidates:
            return None, 0.0

        matrix = np.vstack([e.embedding for e in candidates])
        scores = matrix @ vector
        best = int(np.argmax(scores))
        return candidates[best], float(scores[best])

    def _confirm(self, name: str, match: Entry, claim: str) -> bool:
        """One yes/no question, asked only about a pair already found similar.

        Without a client the answer is no: an unconfirmed alias must not merge
        two predicates, because merging unlike things makes them look like
        contradictions, while leaving them apart only costs recall.
        """
        if self.client is None:
            return False

        prompt = QUESTION.format(
            left=name.replace("_", " "),
            right=match.canonical.replace("_", " "),
            left_claim=claim or "(not given)",
            right_claim="(not given)",
        )
        try:
            answer = self.client.complete(prompt, AliasAnswer, system=SYSTEM)
            self.calls += 1
        except Exception:
            return False
        return answer.same_relation

    def _record(self, event: SchemaEvent) -> SchemaEvent:
        self.events.append(event)
        return event
