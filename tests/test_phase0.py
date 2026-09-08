import pytest
from pydantic import BaseModel

from concord.llm.cache import DiskCache, request_key
from concord.llm.client import LLMClient, LLMError
from concord.store import db


class Answer(BaseModel):
    verdict: str
    score: float


def test_schema_creates_all_tables(tmp_path):
    with db.session(tmp_path / "t.sqlite") as conn:
        names = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"documents", "facts", "relations", "predicate_registry", "schema_events"} <= names


def test_schema_is_idempotent(tmp_path):
    path = tmp_path / "t.sqlite"
    with db.session(path):
        pass
    with db.session(path) as conn:
        conn.execute("SELECT 1 FROM facts")


def test_request_key_is_order_independent():
    assert request_key({"a": 1, "b": 2}) == request_key({"b": 2, "a": 1})


def test_cache_round_trips(tmp_path):
    cache = DiskCache(root=tmp_path)
    payload = {"model": "m", "prompt": "hello"}
    key = request_key(payload)

    assert cache.get(key) is None
    cache.put(key, payload, {"verdict": "corroborates", "score": 0.9})
    assert cache.get(key) == {"verdict": "corroborates", "score": 0.9}
    assert cache.stats()["entries"] == 1


def test_client_serves_from_cache_without_a_key(tmp_path):
    client = LLMClient(model="m", api_key=None, cache=DiskCache(root=tmp_path))
    payload = client._payload("hello", Answer, None, 0.0)
    client.cache.put(request_key(payload), payload, {"verdict": "unrelated", "score": 0.1})

    answer = client.complete("hello", Answer)

    assert answer.verdict == "unrelated"
    assert client.cache_hits == 1
    assert client.calls == 0


def test_client_refuses_to_guess_without_a_key(tmp_path):
    client = LLMClient(model="m", api_key=None, cache=DiskCache(root=tmp_path))
    with pytest.raises(LLMError):
        client.complete("uncached", Answer)


def test_client_does_not_retry_a_permanent_error(tmp_path, monkeypatch):
    import httpx

    from concord.llm import client as mod

    attempts = {"n": 0}

    def fake_post(*args, **kwargs):
        attempts["n"] += 1
        return httpx.Response(404, json={"error": {"message": "model retired"}})

    monkeypatch.setattr(mod.httpx, "post", fake_post)
    c = LLMClient(model="m", api_key="k", cache=DiskCache(root=tmp_path))

    with pytest.raises(LLMError):
        c.complete("hello", Answer)
    assert attempts["n"] == 1, "a 404 must not be retried"


def test_client_retries_a_transient_error(tmp_path, monkeypatch):
    import httpx

    from concord.llm import client as mod

    attempts = {"n": 0}

    def fake_post(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": '{"verdict":"ok","score":1.0}'}]}}
                ]
            },
        )

    monkeypatch.setattr(mod.httpx, "post", fake_post)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    c = LLMClient(model="m", api_key="k", cache=DiskCache(root=tmp_path))

    assert c.complete("hello", Answer).verdict == "ok"
    assert attempts["n"] == 3


def test_key_pool_rotates_past_an_exhausted_key(tmp_path, monkeypatch):
    import httpx

    from concord.llm import client as mod

    used = []

    def fake_post(*args, **kwargs):
        used.append(kwargs["headers"]["x-goog-api-key"])
        if kwargs["headers"]["x-goog-api-key"] == "dead":
            return httpx.Response(429, json={"error": {"message": "credits depleted"}})
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": '{"verdict":"ok","score":1.0}'}]}}]},
        )

    monkeypatch.setattr(mod.httpx, "post", fake_post)
    c = LLMClient(model="m", keys=["dead", "live"], cache=DiskCache(root=tmp_path))

    assert c.complete("hello", Answer).verdict == "ok"
    assert used == ["dead", "live"]
    assert c.rotations == 1


def test_key_pool_gives_up_when_every_key_is_exhausted(tmp_path, monkeypatch):
    import httpx

    from concord.llm import client as mod

    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        return httpx.Response(429, json={"error": {"message": "credits depleted"}})

    monkeypatch.setattr(mod.httpx, "post", fake_post)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    c = LLMClient(model="m", keys=["a", "b", "c"], cache=DiskCache(root=tmp_path), max_attempts=2)

    with pytest.raises(LLMError):
        c.complete("hello", Answer)
    assert calls["n"] < 20, "rotation must terminate, not loop forever"


def test_single_key_does_not_rotate(tmp_path, monkeypatch):
    import httpx

    from concord.llm import client as mod

    monkeypatch.setattr(
        mod.httpx, "post", lambda *a, **k: httpx.Response(429, json={"error": {}})
    )
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    c = LLMClient(model="m", keys=["only"], cache=DiskCache(root=tmp_path), max_attempts=2)

    with pytest.raises(LLMError):
        c.complete("hello", Answer)
    assert c.rotations == 0


def test_the_servers_own_retry_delay_is_preferred_to_guessing():
    """Observed live: the schedule gave up after ~30s on a quota asking 52."""
    import httpx

    from concord.llm.client import MAX_RETRY_AFTER, retry_after

    def response(payload):
        return httpx.Response(429, json=payload)

    quota = {
        "error": {
            "details": [
                {"@type": "type.googleapis.com/google.rpc.Help", "links": []},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "52s"},
            ]
        }
    }
    assert retry_after(response(quota)) == 52.0
    assert retry_after(response({"error": {"details": []}})) is None
    assert retry_after(httpx.Response(429, text="not json")) is None

    huge = {"error": {"details": [{"retryDelay": "86400s"}]}}
    assert retry_after(response(huge)) == MAX_RETRY_AFTER
