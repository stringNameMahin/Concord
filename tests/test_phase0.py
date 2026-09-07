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
