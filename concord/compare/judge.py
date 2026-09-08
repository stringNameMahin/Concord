"""The adjudication prompt and the shape of a verdict.

This is the only place in the system where a model is asked to decide
something, and it sees a deliberately narrow slice: pairs the deterministic
layer has already established share a comparison key and hold incompatible
values. LegalWiz measured 37.3% precision for NLI-style contradiction
detection asked cold; asking "are these contradictory?" of an unfiltered pair
stream would drown the ledger in false positives. Filter first, judge second.

The prompt is asymmetric on purpose. It asks the model to *find the qualifier
that reconciles the two facts*, and only to report a conflict when it cannot.
Reconciliation is the direction of the true prior - two documents from the same
publisher usually disagree because they are measuring different things - so the
prompt leans the way the evidence usually leans, and the guards in
`adjudicate.py` catch it leaning too far.

The model is given the qualifier keys each fact actually carries and told to
name one of them. A verdict naming a key that exists in neither record is
rejected in code, not argued with.
"""

from typing import Literal

from pydantic import BaseModel, Field

from concord.facts import Fact

JUDGE_VERDICTS = ("reconciled_by_context", "contradicts", "insufficient_context")


class VerdictOut(BaseModel):
    verdict: Literal["reconciled_by_context", "contradicts", "insufficient_context"]
    qualifier_key: str | None = Field(
        default=None,
        description=(
            "The condition that explains the difference, copied exactly from the "
            "keys listed on the two facts. Null only when no listed key explains it."
        ),
    )
    explanation: str = Field(
        description="Two sentences at most, addressed to someone who has read neither document"
    )
    confidence: float = Field(ge=0.0, le=1.0)


SYSTEM = """\
You reconcile pairs of facts drawn from documents. You work on any kind of
document and must not assume a subject area.

Each pair you receive has already been checked by a deterministic system, which
established two things you may take as given: the two facts describe the same
property of the same subject, and their values are incompatible at the
precision each was written to. Do not re-examine either finding. Do not
recalculate, rescale or convert anything. Arithmetic is not your job here and
your answer will be discarded if it rests on any.

Your job is the remaining question: is there a condition under which both facts
could be true?

Most pairs like this are not disagreements. They are the same property measured
over different periods, at different dates, over different scopes, on different
bases, or for different segments. Look for that condition first and report a
conflict only when you have looked and found none.

How to answer.

1. Compare the qualifier lists given for each fact. If one of those keys holds
   different values on the two sides and that difference would account for the
   difference in the figures, answer `reconciled_by_context` and set
   `qualifier_key` to that key, spelled exactly as it appears in the list.
2. If a key that would settle the question is listed on one fact and missing
   from the other, you cannot tell. Answer `insufficient_context` and set
   `qualifier_key` to the key that is missing.
3. Answer `contradicts` only when the two facts state the same property, under
   the same stated conditions, and cannot both be true. Set `qualifier_key` to
   null.
4. Never invent a qualifier key. You may only name a key that appears in one of
   the two lists you are given. A verdict naming any other key is rejected.
5. `explanation` states what you concluded and why, in plain language, without
   naming this system or its rules. Write it for a reader who has seen neither
   document.\
"""

PAIR = """\
<fact id="A">
  <claim>{a_claim}</claim>
  <subject>{a_subject}</subject>
  <property>{a_predicate}</property>
  <value>{a_value}</value>
  <conditions>
{a_qualifiers}
  </conditions>
  <source>{a_source}</source>
</fact>

<fact id="B">
  <claim>{b_claim}</claim>
  <subject>{b_subject}</subject>
  <property>{b_predicate}</property>
  <value>{b_value}</value>
  <conditions>
{b_qualifiers}
  </conditions>
  <source>{b_source}</source>
</fact>

<deterministic_finding>
{finding}
</deterministic_finding>

Is there a condition under which both of these could be true?\
"""


def render_qualifiers(fact: Fact) -> str:
    """List the keys the model is allowed to name, with their provenance.

    Provenance is shown because `inherited` means the condition came from the
    enclosing page or section rather than the sentence, and a reader deciding
    whether a difference is real should know which.
    """
    lines = []
    for key in sorted(fact.qualifier_keys()):
        qualifier = fact.qualifier(key)
        lines.append(f"    {key}: {qualifier.value}  [{qualifier.provenance}]")
    return "\n".join(lines) or "    (none recorded)"


def _source(fact: Fact) -> str:
    return f"document {fact.doc_id}, page {fact.evidence.page}"


def build(a: Fact, b: Fact, finding: str) -> str:
    """Render the prompt for one ordered pair.

    Order matters to the caller, not to the prompt: the same pair is rendered
    both ways round and the two answers compared, so nothing here should treat
    A as privileged.
    """
    return PAIR.format(
        a_claim=a.claim_text,
        a_subject=a.subject_surface,
        a_predicate=a.predicate,
        a_value=a.value_raw,
        a_qualifiers=render_qualifiers(a),
        a_source=_source(a),
        b_claim=b.claim_text,
        b_subject=b.subject_surface,
        b_predicate=b.predicate,
        b_value=b.value_raw,
        b_qualifiers=render_qualifiers(b),
        b_source=_source(b),
        finding=finding,
    )
