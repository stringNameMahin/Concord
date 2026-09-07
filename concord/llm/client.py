import json
import random
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from concord import config
from concord.llm.cache import DiskCache, MissingFromCache, request_key

T = TypeVar("T", bound=BaseModel)

RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


class Retryable(LLMError):
    """Transient: rate limit or server error. A 404 or 400 is not this."""


class LLMClient:
    """Gemini over plain HTTP with a disk cache and strict JSON output.

    Deliberately not a framework. There are four call sites in this system and
    all of them want the same thing: a prompt in, a validated pydantic model
    out, cached by request hash.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        cache: DiskCache | None = None,
        max_attempts: int = 5,
    ):
        self.model = model or config.LLM_MODEL
        self.api_key = api_key if api_key is not None else config.LLM_API_KEY
        self.cache = cache or DiskCache()
        self.max_attempts = max_attempts
        self.calls = 0
        self.cache_hits = 0

    def complete(
        self,
        prompt: str,
        schema: type[T],
        system: str | None = None,
        temperature: float = 0.0,
    ) -> T:
        payload = self._payload(prompt, schema, system, temperature)
        key = request_key(payload)

        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return schema.model_validate(cached)

        if config.OFFLINE:
            raise MissingFromCache(
                f"offline mode and no cached response for {key[:12]}; "
                "run without CONCORD_OFFLINE to populate the cache"
            )
        if not self.api_key:
            raise LLMError("no API key; set GEMINI_API_KEY or run with CONCORD_OFFLINE=1")

        raw = self._post(payload)
        self.calls += 1
        try:
            parsed = schema.model_validate_json(raw)
        except ValidationError as exc:
            raise LLMError(f"model returned JSON that does not fit {schema.__name__}: {exc}") from exc

        self.cache.put(key, payload, json.loads(raw))
        return parsed

    def _payload(
        self, prompt: str, schema: type[T], system: str | None, temperature: float
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
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

    def _post(self, payload: dict[str, Any]) -> str:
        body = {k: v for k, v in payload.items() if k != "model"}
        url = f"{config.LLM_ENDPOINT}/{self.model}:generateContent"
        last: Exception | None = None

        for attempt in range(self.max_attempts):
            try:
                response = httpx.post(
                    url,
                    json=body,
                    headers={"x-goog-api-key": self.api_key},
                    timeout=180.0,
                )
                if response.status_code in RETRY_STATUS:
                    raise Retryable(f"{response.status_code}: {response.text[:200]}")
                if response.status_code >= 400:
                    raise LLMError(f"{response.status_code}: {response.text[:300]}")
                return _first_text(response.json())
            except (httpx.TransportError, Retryable) as exc:
                last = exc
                if attempt == self.max_attempts - 1:
                    break
                time.sleep(min(2**attempt, 30) + random.random())

        raise LLMError(f"giving up after {self.max_attempts} attempts: {last}")


def _first_text(response: dict[str, Any]) -> str:
    try:
        return response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"unexpected response shape: {json.dumps(response)[:400]}") from exc
