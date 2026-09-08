"""Two vendors, one interface.

The seam exists because the binding constraint differs: Gemini's free tier caps
requests per day, an OpenRouter key is priced per token. These tests pin the
three things that actually differ, and the one thing that must not - that a
cached answer is never served for a different model.
"""

import pytest

from concord import config
from concord.extract.models import ExtractionOut
from concord.llm import providers
from concord.llm.cache import DiskCache, request_key
from concord.llm.client import BudgetExhausted, LLMClient, LLMError, _first_text


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gemini-3.6-flash", "gemini"),
        ("gemini-3.1-flash-lite", "gemini"),
        ("qwen/qwen3-30b-a3b-instruct-2507", "openrouter"),
        ("meta-llama/llama-3.3-70b-instruct", "openrouter"),
    ],
)
def test_the_provider_is_inferred_from_the_model_name(model, expected):
    assert providers.for_model(model) == expected


def test_an_unknown_provider_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown provider"):
        providers.build("anthropic-ish")


def test_gemini_sends_the_schema_and_keeps_the_model_out_of_the_body():
    gemini = providers.Gemini()
    payload = gemini.payload("gemini-3.6-flash", "hello", ExtractionOut, "be careful", 0.0)

    assert payload["model"] == "gemini-3.6-flash"
    assert payload["systemInstruction"]["parts"][0]["text"] == "be careful"
    assert payload["generationConfig"]["responseJsonSchema"]["title"] == "ExtractionOut"
    assert "model" not in gemini.body(payload)
    assert gemini.url("m").endswith("/m:generateContent")
    assert "x-goog-api-key" in gemini.headers("k")


def test_openrouter_sends_openai_shaped_messages_with_the_model_in_the_body():
    router = providers.OpenRouter()
    payload = router.payload("qwen/q", "hello", ExtractionOut, "be careful", 0.0)

    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert payload["messages"][1]["content"] == "hello"
    assert router.body(payload)["model"] == "qwen/q"
    assert router.url("qwen/q").endswith("/chat/completions")
    assert router.headers("k")["Authorization"] == "Bearer k"


def test_openrouter_leaves_strict_decoding_off_by_default():
    """Strict grammar decoding on our nested schema measurably wrecked the
    output; the schema is a hint and pydantic is the enforcement."""
    payload = providers.OpenRouter().payload("m", "p", ExtractionOut, None, 0.0)
    assert payload["response_format"]["json_schema"]["strict"] is False
    assert payload["response_format"]["json_schema"]["schema"]["title"] == "ExtractionOut"


def test_a_system_prompt_is_optional_for_both():
    assert "systemInstruction" not in providers.Gemini().payload("m", "p", ExtractionOut, None, 0.0)
    payload = providers.OpenRouter().payload("m", "p", ExtractionOut, None, 0.0)
    assert [m["role"] for m in payload["messages"]] == ["user"]


def test_each_provider_finds_the_text_where_its_vendor_puts_it():
    gemini = {"candidates": [{"content": {"parts": [{"text": "G"}]}}]}
    router = {"choices": [{"message": {"content": "O"}}]}

    assert providers.Gemini().extract(gemini) == "G"
    assert providers.OpenRouter().extract(router) == "O"
    assert _first_text(gemini) == "G"
    assert _first_text(router, providers.OpenRouter()) == "O"


def test_an_empty_completion_is_an_error_not_an_empty_answer():
    """A model that hits its token ceiling returns no content. Treating that as
    a valid empty response would silently drop a whole batch of facts."""
    with pytest.raises(KeyError, match="token ceiling"):
        providers.OpenRouter().extract({"choices": [{"message": {"content": ""}}]})


def test_a_cached_answer_is_never_served_for_a_different_model(tmp_path):
    cache = DiskCache(root=tmp_path)
    gemini = LLMClient(model="gemini-3.6-flash", api_key="k", cache=cache)
    qwen = LLMClient(model="qwen/qwen3-30b-a3b-instruct-2507", api_key="k", cache=cache)

    assert gemini.provider.name == "gemini"
    assert qwen.provider.name == "openrouter"

    a = request_key(gemini._payload("hello", ExtractionOut, None, 0.0))
    b = request_key(qwen._payload("hello", ExtractionOut, None, 0.0))
    assert a != b


def test_the_client_can_be_pointed_at_a_provider_explicitly():
    client = LLMClient(model="some-local-model", api_key="k", provider="openrouter")
    assert client.provider.name == "openrouter"
    assert client.provider.url("some-local-model").endswith("/chat/completions")


def test_the_committed_cache_defaults_are_in_step_with_each_other():
    """The defaults must describe the cache we shipped, not the newest model.

    `data/cache/` is keyed by the whole request payload, so both the model and
    the batch size are cache keys. A default that disagrees with either makes a
    re-ingest silently re-buy every response it already owns. These two values
    are the ones the committed corpus was extracted with; changing one without
    repopulating the cache is the accident this test exists to catch.
    """
    assert config.LLM_MODEL == "gemini-3.1-flash-lite"
    assert config.BATCH_SIZE == 16


def test_extraction_reads_its_batch_size_from_config():
    """One source of truth. Two copies drifted once and cost a cache."""
    from concord.extract import runner

    assert runner.BATCH_SIZE is config.BATCH_SIZE
    assert runner.WORKERS is config.WORKERS


def test_a_client_stops_once_it_has_spent_its_budget(tmp_path):
    client = LLMClient(api_key="k", cache=DiskCache(root=tmp_path), max_calls=2)
    client.calls = 2

    with pytest.raises(BudgetExhausted, match="stopped after 2 live requests"):
        client.complete("hello", ExtractionOut)


def test_the_budget_counts_live_requests_and_not_cache_hits(tmp_path):
    """A cached answer costs nothing, so it must not consume the allowance."""
    cache = DiskCache(root=tmp_path)
    client = LLMClient(api_key="k", cache=cache, max_calls=1)
    cache.put(
        request_key(client._payload("hello", ExtractionOut, None, 0.0)),
        {},
        {"facts": []},
    )

    for _ in range(5):
        assert client.complete("hello", ExtractionOut).facts == []
    assert (client.calls, client.cache_hits) == (0, 5)


def test_a_budget_of_zero_means_no_ceiling(tmp_path):
    client = LLMClient(api_key="k", cache=DiskCache(root=tmp_path), max_calls=0)
    client.calls = 10_000
    client._post = lambda payload: '{"facts": []}'

    # Well past any plausible ceiling; with the limit off it still goes through.
    assert client.complete("hello", ExtractionOut).facts == []
    assert client.calls == 10_001


def test_an_exhausted_budget_is_survivable_like_any_other_failure():
    """Callers catch `LLMError` and count the failure. The ceiling must land in
    that path, so a run that hits it keeps the responses it already paid for."""
    assert issubclass(BudgetExhausted, LLMError)
