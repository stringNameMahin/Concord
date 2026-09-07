from typing import Literal

from pydantic import BaseModel, Field

VALUE_KINDS = ("quantity", "text", "date", "boolean")
PROVENANCE = ("stated", "inherited")


class ValueOut(BaseModel):
    kind: Literal["quantity", "text", "date", "boolean"]
    raw: str = Field(description="The value exactly as written in the document")
    unit: str | None = Field(
        default=None, description="Currency code or unit of measure, as written"
    )
    scale: str | None = Field(
        default=None, description="Magnitude word such as million or crore, if any"
    )


class QualifierOut(BaseModel):
    key: str = Field(description="Lower snake_case condition name")
    value: str
    provenance: Literal["stated", "inherited"]


class FactOut(BaseModel):
    passage_id: int = Field(
        description="id of the passage this fact and its quote came from"
    )
    claim_text: str = Field(description="One self-contained sentence stating the fact")
    subject_surface: str
    subject_key: str | None = Field(
        default=None, description="Hard identifier for the subject if the document gives one"
    )
    subject_type: str | None = None
    predicate: str = Field(description="Lower snake_case relation name")
    value: ValueOut
    qualifiers: list[QualifierOut] = Field(default_factory=list)
    quote: str = Field(description="Verbatim span copied from the document")
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionOut(BaseModel):
    facts: list[FactOut] = Field(default_factory=list)
