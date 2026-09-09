import json
import random
import threading
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from concord import config
from concord.llm import providers
from concord.llm.cache import DiskCache, MissingFromCache, request_key

T = TypeVar("T", bound=BaseModel)

RETRY_STATUS = {429, 500, 502, 503, 504}

# A rate-limited response often names how long to wait. Honouring it beats
# guessing: the exponential schedule gave up after ~30s on a quota that asked
# for 52. Capped so a long daily-quota delay fails fast instead of hanging.
MAX_RETRY_AFTER = 120.0


class LLMError(RuntimeError):
    pass


class Retryable(LLMError):
    """Transient: rate limit or server error. A 404 or 400 is not this."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class BudgetExhausted(LLMError):
    """This client has made the most live requests it was allowed to make.

    An `LLMError` on purpose. Every call site already treats a failed request
    as something to count and carry on from rather than crash on, so a run that
    hits its ceiling degrades exactly like a run that lost its key: the work
    already done is kept, the shortfall is counted in `failed_batches` or
    `unadjudicated`, and the reason travels out with it. Stopping loudly at the
    top would throw away the responses already paid for on the way here.
    """


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
        keys: list[str] | None = None,
        provider: str | object | None = None,
        max_calls: int | None = None,
    ):
        self.model = model or config.LLM_MODEL
        if provider is None:
            provider = (
                config.LLM_PROVIDER if model is None else providers.for_model(self.model)
            )
        self.provider = (
            providers.build(provider, config.LLM_ENDPOINT)
            if isinstance(provider, str)
            else provider
        )
        if keys is not None:
            self.keys = list(keys)
        elif api_key is not None:
            self.keys = [api_key] if api_key else []
        else:
            self.keys = list(config._keys(self.provider.name))
        self.cursor = 0
        self.cache = cache or DiskCache()
        self.max_attempts = max_attempts
        self.max_calls = config.MAX_CALLS if max_calls is None else max_calls
        self.calls = 0
        self.cache_hits = 0
        self.rotations = 0
        self._budget = threading.Lock()

    @property
    def api_key(self) -> str | None:
        return self.keys[self.cursor] if self.keys else None

    def _reserve(self) -> None:
        """Claim one live request from the ceiling before sending it.

        The ceiling used to be a check in `complete()` with the increment after
        `_post` returned, which is check-then-act: batches run concurrently, so
        three workers all read `calls == 0` against a limit of 2 and all three
        posted. A ceiling of 2 bought 4 requests - see bug 21 in docs/status.md.
        Reserving under a lock makes the count a claim on the budget rather
        than a record of what was already spent, so the ceiling holds however
        many workers are in flight.

        A claim is not refunded when the request fails. A request the provider
        received and rejected still consumed quota, and the point of this
        counter is what leaves the machine, not what came back usable.
        """
        with self._budget:
            if self.max_calls and self.calls >= self.max_calls:
                raise BudgetExhausted(
                    f"stopped after {self.calls} live requests, the limit set by "
                    f"CONCORD_MAX_CALLS. {self.cache_hits} came from cache and cost "
                    "nothing. Raise the limit, or set it to 0 for no limit, if this "
                    "run is meant to be this large."
                )
            self.calls += 1

    def _rotate(self) -> bool:
        """Move to the next key. Returns False once every key has been tried.

        Only useful across keys on separate projects: the free-tier quota is
        billed per project, so several keys in one project share one bucket.
        """
        if len(self.keys) < 2:
            return False
        self.cursor = (self.cursor + 1) % len(self.keys)
        self.rotations += 1
        return self.rotations % len(self.keys) != 0

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

        self._reserve()
        raw = self._post(payload)
        try:
            parsed = schema.model_validate_json(raw)
        except ValidationError as exc:
            raise LLMError(f"model returned JSON that does not fit {schema.__name__}: {exc}") from exc

        self.cache.put(key, payload, json.loads(raw))
        return parsed

    def _payload(
        self, prompt: str, schema: type[T], system: str | None, temperature: float
    ) -> dict[str, Any]:
        """The cache key is this payload, and it carries the model, so switching
        provider or model asks again rather than serving another model's answer."""
        return self.provider.payload(self.model, prompt, schema, system, temperature)

    def _post(self, payload: dict[str, Any]) -> str:
        body = self.provider.body(payload)
        url = self.provider.url(self.model)
        last: Exception | None = None

        attempt = 0
        sent = 0
        while attempt < self.max_attempts:
            try:
                # The caller reserved the first send. A retry and a key
                # rotation are each another request over the wire, so each
                # claims its own slot: a rate-limited batch that retried five
                # times used to register as one call and cost five.
                if sent:
                    self._reserve()
                sent += 1
                response = httpx.post(
                    url,
                    json=body,
                    headers=self.provider.headers(self.api_key),
                    timeout=300.0,
                )
                # Trying another key is not a retry: it costs no wait and the
                # backoff budget should survive for genuine transient failures.
                if response.status_code == 429 and self._rotate():
                    continue
                if response.status_code in RETRY_STATUS:
                    raise Retryable(
                        f"{response.status_code}: {response.text[:200]}",
                        retry_after=retry_after(response),
                    )
                if response.status_code >= 400:
                    raise LLMError(f"{response.status_code}: {response.text[:300]}")
                return _first_text(response.json(), self.provider)
            except (httpx.TransportError, Retryable) as exc:
                last = exc
                attempt += 1
                if attempt >= self.max_attempts:
                    break
                wait = getattr(last, "retry_after", None)
                if wait is None:
                    wait = min(2**attempt, 30) + random.random()
                time.sleep(wait)

        raise LLMError(f"giving up after {self.max_attempts} attempts: {last}")


def retry_after(response: httpx.Response) -> float | None:
    """Read the server's own RetryInfo, in preference to guessing.

    Google returns the wait it wants in the error details; a schedule that
    ignores it either hammers the endpoint or gives up early on a quota that
    would have cleared.
    """
    try:
        details = response.json().get("error", {}).get("details", [])
    except (ValueError, AttributeError):
        return None

    for detail in details:
        raw = detail.get("retryDelay") if isinstance(detail, dict) else None
        if not raw:
            continue
        try:
            seconds = float(str(raw).rstrip("s"))
        except ValueError:
            return None
        return min(seconds, MAX_RETRY_AFTER)
    return None


def _first_text(response: dict[str, Any], provider=None) -> str:
    provider = provider or providers.Gemini()
    try:
        return provider.extract(response)
    except (KeyError, IndexError) as exc:
        raise LLMError(f"unexpected response shape: {json.dumps(response)[:400]}") from exc
