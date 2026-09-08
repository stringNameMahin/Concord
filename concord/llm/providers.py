"""Two ways to ask a model for JSON, behind one interface.

The client's cache, retry schedule, key rotation and pydantic validation are
provider-agnostic and stay where they are. Only three things actually differ
between vendors - where the request goes, what shape it takes, and where the
text sits in the reply - so that is all a provider is.

The point of the seam is not vendor-neutrality for its own sake. It is that the
binding constraint changes: Gemini's free tier caps requests per day, while an
OpenRouter key is priced per token with no daily wall. Being able to move
between them without touching extraction or adjudication is what keeps a quota
problem from becoming a rewrite.
"""

from __future__ import annotations

from typing import Any


class Gemini:
    """Google's generateContent, with a response schema attached."""

    name = "gemini"
    default_endpoint = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, endpoint: str | None = None):
        self.endpoint = endpoint or self.default_endpoint

    def payload(self, model, prompt, schema, system, temperature) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
                "responseJsonSchema": schema.model_json_schema(),
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def url(self, model: str) -> str:
        return f"{self.endpoint}/{model}:generateContent"

    def headers(self, key: str | None) -> dict[str, str]:
        return {"x-goog-api-key": key or ""}

    def body(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in payload.items() if k != "model"}

    def extract(self, data: dict[str, Any]) -> str:
        return data["candidates"][0]["content"]["parts"][0]["text"]


class OpenRouter:
    """OpenAI-shaped chat completions, routed to any hosted model.

    `strict` is off deliberately. Our extraction schema is nested and uses
    `$ref`, and asking a mid-sized open model to satisfy it under
    grammar-constrained decoding measurably wrecked the output: field roles got
    swapped and the response ran past the token limit mid-string. Passing the
    schema as a strong hint and validating with pydantic afterwards produced
    valid, sane results from the same model. Validation still happens either
    way, so nothing unvalidated reaches the ledger - the difference is only
    whether the constraint is applied during decoding or after it.
    """

    name = "openrouter"
    default_endpoint = "https://openrouter.ai/api/v1"

    def __init__(self, endpoint: str | None = None, strict: bool = False, max_tokens: int = 16000):
        self.endpoint = endpoint or self.default_endpoint
        self.strict = strict
        self.max_tokens = max_tokens

    def payload(self, model, prompt, schema, system, temperature) -> dict[str, Any]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "strict": self.strict,
                    "schema": schema.model_json_schema(),
                },
            },
        }

    def url(self, model: str) -> str:
        return f"{self.endpoint}/chat/completions"

    def headers(self, key: str | None) -> dict[str, str]:
        return {"Authorization": f"Bearer {key or ''}", "Content-Type": "application/json"}

    def body(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload  # the model belongs in an OpenAI-shaped body

    def extract(self, data: dict[str, Any]) -> str:
        message = data["choices"][0]["message"]
        content = message.get("content")
        if not content:
            raise KeyError("empty content; the model may have hit its token ceiling")
        return content


def for_model(model: str) -> str:
    """Infer the provider from the model name.

    OpenRouter names are always `vendor/model`; Gemini's never contain a slash.
    Inferring keeps the common case configuration-free, and
    `CONCORD_LLM_PROVIDER` overrides it when that guess is wrong.
    """
    return "openrouter" if "/" in model else "gemini"


def build(name: str, endpoint: str | None = None):
    if name == "openrouter":
        return OpenRouter(endpoint)
    if name == "gemini":
        return Gemini(endpoint)
    raise ValueError(f"unknown provider {name!r}; expected 'gemini' or 'openrouter'")
