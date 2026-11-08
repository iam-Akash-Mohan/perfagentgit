"""The LLM client. No network: every request is served by a fake `post`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from senapathy.agent import LoopBudget, LoopStatus, run_agent
from senapathy.contracts import ToolCallResult, Usage
from senapathy.errors import SenapathyError
from senapathy.llm import make_client, resolve_key
from senapathy.llm.anthropic import AnthropicClient
from senapathy.llm.base import (CassetteMiss, LLMHTTPError, backoff_seconds,
                                normalise_tools, retry_after_seconds)
from senapathy.llm.openai_compat import OpenAICompatClient
from senapathy.llm.pricing import usd_for
from senapathy.trace import Trace

TOOL = {"name": "read_file", "description": "read it",
        "parameters": {"type": "object", "properties": {}}}

OPENAI_OK = {
    "model": "gpt-oss-120b",
    "choices": [{"finish_reason": "tool_calls",
                 "message": {"role": "assistant", "content": None,
                             "tool_calls": [{"id": "c1", "type": "function",
                                             "function": {"name": "read_file",
                                                          "arguments": '{"path":"a.java"}'}}]}}],
    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
}

ANTHROPIC_OK = {
    "model": "claude-opus-4-5",
    "stop_reason": "tool_use",
    "content": [{"type": "text", "text": "looking"},
                {"type": "tool_use", "id": "t1", "name": "read_file",
                 "input": {"path": "a.java"}}],
    "usage": {"input_tokens": 100, "output_tokens": 20,
              "cache_read_input_tokens": 900, "cache_creation_input_tokens": 50},
}


class Recorder:
    """A stand-in for httpx.post that replays a scripted list of responses."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, *, headers: dict, json: dict,
                 timeout: float) -> httpx.Response:
        self.calls.append({"url": url, "headers": headers, "body": json})
        if not self.responses:
            raise AssertionError("more requests than scripted responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def ok(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload)


def fail(status: int, body: str = "boom",
         headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, text=body, headers=headers or {})


def client(post: Recorder, tmp_path: Path, **kwargs: Any) -> OpenAICompatClient:
    return OpenAICompatClient(
        provider="cerebras", model="gpt-oss-120b", api_key="k",
        cassette_dir=tmp_path / "cassettes", http_post=post,
        sleep=lambda _s: None, jitter=lambda: 0.0, **kwargs)


# ─────────────────────────────  the factory  ─────────────────────────────


def test_the_factory_switches_provider_without_a_code_change(monkeypatch) -> None:
    monkeypatch.setenv("CEREBRAS_API_KEY", "c")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert isinstance(make_client("cerebras", "m"), OpenAICompatClient)
    assert isinstance(make_client("anthropic", "m"), AnthropicClient)
    assert make_client("ollama", "m").api_key is None


def test_a_missing_key_names_the_variable_and_the_provider(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(SenapathyError, match="GROQ_API_KEY"):
        resolve_key("groq")


def test_an_unknown_provider_is_refused(monkeypatch) -> None:
    with pytest.raises(SenapathyError, match="unknown LLM provider"):
        make_client("openai", "gpt-4")


def test_an_explicit_argument_beats_configuration(monkeypatch) -> None:
    monkeypatch.setenv("CEREBRAS_API_KEY", "c")
    monkeypatch.setenv("SENAPATHY_LLM_MODEL", "from-env")
    assert make_client("cerebras").model == "from-env"
    assert make_client("cerebras", "explicit").model == "explicit"


# ──────────────────────────  status handling  ────────────────────────────


def test_a_non_200_is_never_parsed_as_a_completion(tmp_path: Path) -> None:
    """The old client printed the body and then called r.json() anyway, so a
    429 surfaced three frames away as KeyError('choices')."""
    post = Recorder(*[fail(500, "upstream exploded")] * 5)
    with pytest.raises(LLMHTTPError) as caught:
        client(post, tmp_path).complete(messages=[{"role": "user", "content": "x"}])
    assert caught.value.status == 500
    assert "upstream exploded" in str(caught.value)


def test_a_400_is_never_retried_and_says_what_the_api_said(tmp_path: Path) -> None:
    post = Recorder(fail(400, '{"message":"tools[0].name is required"}'))
    with pytest.raises(LLMHTTPError, match="tools\\[0\\].name is required"):
        client(post, tmp_path).complete(messages=[{"role": "user", "content": "x"}])
    assert len(post.calls) == 1


def test_a_401_is_never_retried(tmp_path: Path) -> None:
    post = Recorder(fail(401, "bad key"))
    with pytest.raises(LLMHTTPError, match="401"):
        client(post, tmp_path).complete(messages=[{"role": "user", "content": "x"}])
    assert len(post.calls) == 1


def test_a_429_is_retried_and_then_succeeds(tmp_path: Path) -> None:
    post = Recorder(fail(429, "slow down"), ok(OPENAI_OK))
    response = client(post, tmp_path).complete(
        messages=[{"role": "user", "content": "x"}])
    assert len(post.calls) == 2
    assert response.stop_reason == "tool_use"


def test_a_timeout_is_retried(tmp_path: Path) -> None:
    post = Recorder(httpx.ConnectTimeout("timed out"), ok(OPENAI_OK))
    response = client(post, tmp_path).complete(
        messages=[{"role": "user", "content": "x"}])
    assert len(post.calls) == 2
    assert response.usage.input_tokens == 100


def test_retry_after_is_honoured() -> None:
    assert retry_after_seconds({"Retry-After": "12"}) == 12.0
    assert retry_after_seconds({"retry-after": "not a number"}) is None
    assert retry_after_seconds({}) is None
    # Capped, so a hostile header cannot park the run for an hour.
    assert retry_after_seconds({"Retry-After": "9999"}) == 60.0


def test_backoff_is_exponential_and_capped() -> None:
    values = [backoff_seconds(n, jitter=lambda: 1.0) for n in range(1, 9)]
    assert values[0] == 1.0 and values[1] == 2.0 and values[2] == 4.0
    assert max(values) == 60.0


# ────────────────────────────  request shape  ────────────────────────────


def test_system_is_a_message_for_openai_and_a_field_for_anthropic(
    tmp_path: Path
) -> None:
    post = Recorder(ok(OPENAI_OK))
    client(post, tmp_path).complete(system="be brief",
                                    messages=[{"role": "user", "content": "x"}])
    assert post.calls[0]["body"]["messages"][0] == {"role": "system",
                                                    "content": "be brief"}

    apost = Recorder(ok(ANTHROPIC_OK))
    AnthropicClient(model="claude-opus-4-5", api_key="k",
                    cassette_dir=tmp_path / "c2", http_post=apost).complete(
        system="be brief", messages=[{"role": "user", "content": "x"}])
    body = apost.calls[0]["body"]
    assert "system" in body
    assert not any(m["role"] == "system" for m in body["messages"])


def test_empty_tools_is_omitted_not_sent_as_null(tmp_path: Path) -> None:
    post = Recorder(ok(OPENAI_OK))
    client(post, tmp_path).complete(messages=[{"role": "user", "content": "x"}])
    assert "tools" not in post.calls[0]["body"]
    assert "tool_choice" not in post.calls[0]["body"]


def test_openai_shaped_tool_definitions_are_accepted(tmp_path: Path) -> None:
    """Callers written against the pre-S1 client pass function-wrapped tools."""
    wrapped = {"type": "function", "function": TOOL}
    assert normalise_tools([wrapped]) == normalise_tools([TOOL])

    post = Recorder(ok(OPENAI_OK))
    client(post, tmp_path).complete(messages=[{"role": "user", "content": "x"}],
                                    tools=[wrapped])
    assert post.calls[0]["body"]["tools"][0]["function"]["name"] == "read_file"


def test_anthropic_caches_the_last_system_block_and_the_last_tool(
    tmp_path: Path
) -> None:
    post = Recorder(ok(ANTHROPIC_OK))
    AnthropicClient(model="claude-opus-4-5", api_key="k",
                    cassette_dir=tmp_path / "c", http_post=post).complete(
        system="prompt", messages=[{"role": "user", "content": "x"}],
        tools=[TOOL, {**TOOL, "name": "search_code"}])
    body = post.calls[0]["body"]
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in body["tools"][0]
    assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"}


# ────────────────────────────  the response  ─────────────────────────────


def test_the_response_carries_the_raw_message_and_the_client_stores_nothing(
    tmp_path: Path
) -> None:
    """`complete` is a pure function of its arguments; the provider-shaped
    assistant turn lives on the response, not on the client."""
    post = Recorder(ok(OPENAI_OK))
    instance = client(post, tmp_path)
    response = instance.complete(messages=[{"role": "user", "content": "x"}])
    assert response.raw_message["tool_calls"][0]["id"] == "c1"
    assert not hasattr(instance, "last_raw_message")


def test_tool_arguments_that_are_not_json_do_not_crash(tmp_path: Path) -> None:
    broken = json.loads(json.dumps(OPENAI_OK))
    broken["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{oops"
    post = Recorder(ok(broken))
    response = client(post, tmp_path).complete(
        messages=[{"role": "user", "content": "x"}])
    assert "__malformed_arguments__" in response.tool_uses()[0]["input"]


def test_tool_results_use_each_provider_s_wire_shape(tmp_path: Path) -> None:
    results = [ToolCallResult(id="a", name="read_file", content="hi"),
               ToolCallResult(id="b", name="read_file", content="no",
                              is_error=True)]

    openai = client(Recorder(), tmp_path).tool_result_messages(results)
    assert len(openai) == 2
    assert openai[0]["role"] == "tool" and openai[0]["tool_call_id"] == "a"
    assert openai[1]["content"].startswith("ERROR:")

    anthropic = AnthropicClient(model="m", api_key="k",
                                cassette_dir=tmp_path / "c").tool_result_messages(results)
    assert len(anthropic) == 1, "Anthropic takes all tool results in one message"
    assert anthropic[0]["content"][1]["is_error"] is True


# ──────────────────────────────  cost  ───────────────────────────────────


def test_cache_tokens_are_priced_separately() -> None:
    plain = usd_for("claude-opus-4-5", Usage(input_tokens=1_000_000))
    cached = usd_for("claude-opus-4-5", Usage(cache_read_tokens=1_000_000))
    assert plain == pytest.approx(5.0)
    assert cached == pytest.approx(0.5)


def test_an_unknown_model_prices_at_zero_and_warns(caplog) -> None:
    with caplog.at_level("WARNING"):
        assert usd_for("nobody-has-heard-of-this", Usage(input_tokens=10**6)) == 0.0
    assert "no price is configured" in caplog.text


def test_cost_is_accumulated_across_calls(tmp_path: Path) -> None:
    post = Recorder(ok(OPENAI_OK), ok(OPENAI_OK))
    instance = client(post, tmp_path)
    instance.complete(messages=[{"role": "user", "content": "x"}])
    instance.complete(messages=[{"role": "user", "content": "y"}])
    assert instance.calls == 2
    assert instance.total.input_tokens == 200
    assert instance.total.usd == pytest.approx(2 * instance.price(
        Usage(input_tokens=100, output_tokens=20)))


# ────────────────────────────  cassettes  ────────────────────────────────


def test_a_cassette_hit_makes_no_http_call(tmp_path: Path) -> None:
    post = Recorder(ok(OPENAI_OK))
    first = client(post, tmp_path)
    first.complete(messages=[{"role": "user", "content": "x"}])
    assert len(post.calls) == 1

    replay = Recorder()          # any request now raises
    second = client(replay, tmp_path)
    response = second.complete(messages=[{"role": "user", "content": "x"}])
    assert response.from_cassette is True
    assert response.latency_ms == 0.0, "no time was spent; do not claim any"
    assert replay.calls == []


def test_a_changed_prompt_is_a_cassette_miss(tmp_path: Path) -> None:
    post = Recorder(ok(OPENAI_OK), ok(OPENAI_OK))
    instance = client(post, tmp_path)
    instance.complete(system="v1", messages=[{"role": "user", "content": "x"}])
    instance.complete(system="v2", messages=[{"role": "user", "content": "x"}])
    assert len(post.calls) == 2


def test_offline_names_the_missing_hash(tmp_path: Path) -> None:
    instance = client(Recorder(), tmp_path, offline=True)
    with pytest.raises(CassetteMiss) as caught:
        instance.complete(messages=[{"role": "user", "content": "x"}])
    assert "offline=True" in str(caught.value)
    assert str(tmp_path / "cassettes") in str(caught.value)


def test_offline_replays_a_recorded_call_with_no_network(tmp_path: Path) -> None:
    recorded = client(Recorder(ok(OPENAI_OK)), tmp_path)
    recorded.complete(messages=[{"role": "user", "content": "x"}])

    offline = client(Recorder(), tmp_path, offline=True)
    response = offline.complete(messages=[{"role": "user", "content": "x"}])
    assert response.from_cassette and response.tool_uses()[0]["name"] == "read_file"


# ───────────────────────────────  gemini  ────────────────────────────────
#
# Google's OpenAI compatibility layer. These tests pin the two things that
# could silently break the agent loop against it: the URL the request goes to,
# and the shape `parse` is willing to read back. They are cassette-backed —
# recorded once from a scripted response, then replayed with a client whose
# `post` raises on any call — so "it works end to end" is demonstrated without
# a key and without spending anything.

# A Gemini-shaped tool-calling reply. Deliberately NOT identical to OPENAI_OK:
# `content` is an empty string rather than null, and the tool call carries no
# `id`, both of which this layer has been observed to do.
GEMINI_OK = {
    "model": "gemini-2.5-flash",
    "choices": [{"finish_reason": "tool_calls",
                 "message": {"role": "assistant", "content": "",
                             "tool_calls": [{"type": "function",
                                             "function": {"name": "read_file",
                                                          "arguments": '{"path":"a.java"}'}}]}}],
    "usage": {"prompt_tokens": 400, "completion_tokens": 60,
              "total_tokens": 460},
}

GEMINI_DONE = {
    "model": "gemini-2.5-flash",
    "choices": [{"finish_reason": "stop",
                 "message": {"role": "assistant",
                             "content": "the file is a controller"}}],
    "usage": {"prompt_tokens": 500, "completion_tokens": 10},
}


def gemini(post: Recorder, tmp_path: Path, **kwargs: Any) -> OpenAICompatClient:
    return OpenAICompatClient(
        provider="gemini", model="gemini-2.5-flash", api_key="k",
        cassette_dir=tmp_path / "cassettes", http_post=post,
        sleep=lambda _s: None, jitter=lambda: 0.0, **kwargs)


def test_gemini_is_built_by_the_factory_with_googles_endpoint(monkeypatch) -> None:
    """The endpoint and the auth header, not the model — the configured model
    is whatever senapathy.yml currently names, and this test is not about
    that."""
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    instance = make_client("gemini", "gemini-2.5-flash")
    assert isinstance(instance, OpenAICompatClient)
    assert instance.model == "gemini-2.5-flash"
    assert instance.endpoint() == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
    assert instance.headers()["Authorization"] == "Bearer g"


def test_either_accepted_key_name_works_for_gemini(monkeypatch) -> None:
    """Google's own tooling reads either name, so this must too."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "from-google")
    assert resolve_key("gemini") == "from-google"

    monkeypatch.setenv("GEMINI_API_KEY", "from-gemini")
    assert resolve_key("gemini") == "from-gemini", "the first name wins"


def test_a_missing_gemini_key_names_every_variable_it_tried(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(SenapathyError) as caught:
        resolve_key("gemini")
    # A user holding the other name has to be able to see that it would work.
    assert "GEMINI_API_KEY" in str(caught.value)
    assert "GOOGLE_API_KEY" in str(caught.value)


def test_switching_to_gemini_does_not_send_another_vendors_model(
    monkeypatch
) -> None:
    """senapathy.yml names a Cerebras model; `--provider gemini` must not
    forward it to Google."""
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    # Supplied explicitly because line 3 of this test builds a cerebras
    # client. It used to pass without this only because `make_client` calls
    # `load_dotenv()` and the developer's real `.env` happened to carry a
    # CEREBRAS_API_KEY -- a unit test silently depending on a live
    # credential, which surfaced the moment that dead key was deleted.
    monkeypatch.setenv("CEREBRAS_API_KEY", "c")
    monkeypatch.setenv("SENAPATHY_LLM_PROVIDER", "cerebras")
    monkeypatch.setenv("SENAPATHY_LLM_MODEL", "gpt-oss-120b")
    assert make_client("gemini").model == "gemini-2.5-flash"
    assert make_client("gemini", "gemini-2.5-pro").model == "gemini-2.5-pro"
    assert make_client("cerebras").model == "gpt-oss-120b"


def test_gemini_tool_calling_round_trips(tmp_path: Path) -> None:
    post = Recorder(ok(GEMINI_OK))
    response = gemini(post, tmp_path).complete(
        system="you trace endpoints",
        messages=[{"role": "user", "content": "read OrderController"}],
        tools=[TOOL])

    sent = post.calls[0]
    assert sent["url"].endswith("/v1beta/openai/chat/completions")
    assert sent["body"]["model"] == "gemini-2.5-flash"
    assert sent["body"]["max_tokens"] == 4096
    assert sent["body"]["tools"][0]["function"]["name"] == "read_file"
    # The system prompt is folded into messages; this layer has no system field.
    assert sent["body"]["messages"][0] == {"role": "system",
                                           "content": "you trace endpoints"}

    call = response.tool_uses()[0]
    assert response.stop_reason == "tool_use"
    assert call["name"] == "read_file"
    assert call["input"] == {"path": "a.java"}      # the JSON string is parsed
    assert call["id"], "a tool call with no id must still be addressable"
    assert response.text() == ""                    # content "" is not a text block
    assert response.usage.input_tokens == 400
    assert response.usage.usd == pytest.approx(
        usd_for("gemini-2.5-flash", Usage(input_tokens=400, output_tokens=60)))


def test_a_gemini_tool_call_without_an_id_is_still_correlated(
    tmp_path: Path
) -> None:
    """The id is the correlation key for the result message. Google may not
    send one, and a KeyError here would read as a dead phase."""
    response = gemini(Recorder(ok(GEMINI_OK)), tmp_path).complete(
        messages=[{"role": "user", "content": "x"}], tools=[TOOL])
    call = response.tool_uses()[0]

    results = [ToolCallResult(id=call["id"], name=call["name"], content="ok")]
    wire = gemini(Recorder(), tmp_path).tool_result_messages(results)
    assert wire[0]["tool_call_id"] == call["id"]
    assert wire[0]["role"] == "tool"


def test_gemini_replays_from_a_cassette_with_no_network(tmp_path: Path) -> None:
    recorded = gemini(Recorder(ok(GEMINI_OK)), tmp_path)
    recorded.complete(messages=[{"role": "user", "content": "x"}], tools=[TOOL])

    offline = gemini(Recorder(), tmp_path, offline=True)   # any request raises
    replayed = offline.complete(messages=[{"role": "user", "content": "x"}],
                                tools=[TOOL])
    assert replayed.from_cassette is True
    assert replayed.tool_uses()[0]["name"] == "read_file"


def test_the_agent_loop_drives_gemini_end_to_end(tmp_path: Path) -> None:
    """A real `run_agent` conversation: tool call, tool result, final answer.

    Cassette-backed in both turns, so this exercises the provider's own wire
    shape — the tool_calls block it returns and the `role: tool` message the
    client builds from it — without a key.
    """
    from senapathy.agent import LoopBudget, LoopStatus, run_agent

    # Record both turns.
    recording = gemini(Recorder(ok(GEMINI_OK), ok(GEMINI_DONE)), tmp_path)
    seen: list[str] = []

    def read_file(path: str) -> str:
        seen.append(path)
        return "class OrderController {}"

    first = run_agent(
        client=recording, task="read OrderController",
        tools=[TOOL], handlers={"read_file": read_file},
        budget=LoopBudget(max_steps=4))

    assert first.status is LoopStatus.COMPLETED
    assert seen == ["a.java"]
    assert first.final_text == "the file is a controller"

    # Replay the same conversation offline. Nothing may reach the network.
    seen.clear()
    replay = run_agent(
        client=gemini(Recorder(), tmp_path, offline=True),
        task="read OrderController", tools=[TOOL],
        handlers={"read_file": read_file}, budget=LoopBudget(max_steps=4))

    assert replay.status is LoopStatus.COMPLETED
    assert seen == ["a.java"]
    assert replay.usage.input_tokens == 900        # 400 + 500, both replayed


# ──────────────  output_parse_failed: prose instead of a tool call  ──────


def test_output_parse_failed_is_correctable_not_fatal(tmp_path: Path) -> None:
    """Groq validates `tool_choice` server-side: if the model replies in
    prose when a tool call was required, Groq returns HTTP 400
    `output_parse_failed` instead of handing the prose back as an ordinary
    `end_turn`. That must become a correctable turn, the same as
    `tool_use_failed` already is -- not a crash three frames from anything
    that explains it."""
    body = json.dumps({"error": {
        "code": "output_parse_failed",
        "message": "Failed to parse model output as a tool call.",
        "failed_generation": "Open InvoiceEnricher?",
    }})
    error = LLMHTTPError("groq", 400, body, "https://api.groq.com/x")

    correction = client(Recorder(), tmp_path).correction_for(error)

    assert correction is not None
    assert "a tool call was required" in correction
    assert "you wrote prose" in correction
    assert "Open InvoiceEnricher?" in correction
    # Must not reuse the OTHER code's wording -- that describes a mistake
    # (bad arguments) the model did not make here.
    assert "arguments did not match" not in correction


def test_tool_use_failed_still_describes_a_schema_mismatch(tmp_path: Path) -> None:
    body = json.dumps({"error": {
        "code": "tool_use_failed",
        "message": "parameters did not match schema: missing 'pattern'",
        "failed_generation": '{"name":"search_code","arguments":{"query":"x"}}',
    }})
    error = LLMHTTPError("groq", 400, body, "https://api.groq.com/x")

    correction = client(Recorder(), tmp_path).correction_for(error)

    assert correction is not None
    assert "arguments did not match" in correction
    assert "you wrote prose" not in correction


def test_an_uncorrectable_code_returns_none(tmp_path: Path) -> None:
    body = json.dumps({"error": {"code": "invalid_api_key", "message": "nope"}})
    error = LLMHTTPError("groq", 401, body, "https://api.groq.com/x")

    assert client(Recorder(), tmp_path).correction_for(error) is None


def test_json_validate_failed_is_handled_like_tool_use_failed(tmp_path: Path) -> None:
    """Groq's third documented member of this family: the model's
    structured output broke the schema. Same shape as tool_use_failed, so
    it gets the same correction."""
    body = json.dumps({"error": {
        "code": "json_validate_failed",
        "message": "structured output failed schema validation",
        "failed_generation": '{"name":"search_code","arguments":{"query":"x"}}',
    }})
    error = LLMHTTPError("groq", 400, body, "https://api.groq.com/x")

    correction = client(Recorder(), tmp_path).correction_for(error)

    assert correction is not None
    assert "arguments did not match" in correction


def test_tool_use_failed_names_a_hallucinated_tool(tmp_path: Path) -> None:
    """COL-03: the model reached for print_tree, which is not in the patch
    registry. 'Read the schema again' is the wrong fix for a tool that does
    not exist — the correction has to say so and name what IS available."""
    body = json.dumps({"error": {
        "code": "tool_use_failed",
        "message": "model attempted to call tool print_tree",
        "failed_generation": '{"name":"print_tree","arguments":{}}',
    }})
    error = LLMHTTPError("groq", 400, body, "https://api.groq.com/x")

    correction = client(Recorder(), tmp_path).correction_for(
        error, tool_names=["apply_patch", "build", "run_tests", "emit_patch"])

    assert correction is not None
    assert "'print_tree' is not a tool in this phase's registry" in correction
    assert "apply_patch" in correction and "emit_patch" in correction
    assert "arguments did not match" not in correction


def test_tool_use_failed_with_a_real_tool_name_still_describes_a_schema_mismatch(
    tmp_path: Path,
) -> None:
    """The registry check must not false-positive on a VALID tool whose
    arguments were simply wrong — the exact case the original message was
    written for."""
    body = json.dumps({"error": {
        "code": "tool_use_failed",
        "message": "parameters did not match schema: missing 'pattern'",
        "failed_generation": '{"name":"search_code","arguments":{"query":"x"}}',
    }})
    error = LLMHTTPError("groq", 400, body, "https://api.groq.com/x")

    correction = client(Recorder(), tmp_path).correction_for(
        error, tool_names=["search_code", "read_file"])

    assert correction is not None
    assert "arguments did not match" in correction
    assert "not a tool in this phase's registry" not in correction


# ─────────  the loop, end to end, against the real provider client  ──────
#
# Not a fake `correction_for` -- `client()` builds a real OpenAICompatClient,
# so these prove the actual integration (run_agent -> _correction_for ->
# OpenAICompatClient.correction_for) rather than a test double's opinion of
# what it should do. `this must work for every agentic phase` is a property
# of this wiring, not of patch specifically, so phase="patch" below is
# incidental -- any phase string would behave identically.

_FINISH_CALL = {
    "model": "gpt-oss-120b",
    "choices": [{"finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": None,
                            "tool_calls": [{"id": "c1", "type": "function",
                                            "function": {"name": "finish",
                                                         "arguments": "{}"}}]}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}


def test_output_parse_failed_drives_the_loop_to_a_correction(tmp_path: Path) -> None:
    reject = json.dumps({"error": {
        "code": "output_parse_failed",
        "message": "Failed to parse model output as a tool call.",
        "failed_generation": "Open InvoiceEnricher?",
    }})
    post = Recorder(fail(400, reject), ok(_FINISH_CALL))
    llm = client(post, tmp_path)
    tools = [{"name": "finish", "description": "d",
             "parameters": {"type": "object", "properties": {}}}]

    outcome = run_agent(
        client=llm, task="go", tools=tools, handlers={"finish": lambda **k: "done"},
        terminal_tool="finish", budget=LoopBudget(max_steps=5))

    assert outcome.status is LoopStatus.COMPLETED
    assert len(post.calls) == 2, "it asked again instead of dying"
    second = post.calls[1]["body"]["messages"]
    assert any("you wrote prose" in str(m.get("content")) for m in second)


def test_a_hallucinated_tool_is_named_and_the_registry_is_offered(
    tmp_path: Path,
) -> None:
    reject = json.dumps({"error": {
        "code": "tool_use_failed",
        "message": "model attempted an unknown tool",
        "failed_generation": '{"name":"print_tree","arguments":{}}',
    }})
    post = Recorder(fail(400, reject), ok(_FINISH_CALL))
    llm = client(post, tmp_path)
    tools = [{"name": "apply_patch", "description": "d",
             "parameters": {"type": "object", "properties": {}}},
             {"name": "finish", "description": "d",
             "parameters": {"type": "object", "properties": {}}}]

    outcome = run_agent(
        client=llm, task="go", tools=tools, handlers={"finish": lambda **k: "done"},
        terminal_tool="finish", budget=LoopBudget(max_steps=5))

    assert outcome.status is LoopStatus.COMPLETED
    second = post.calls[1]["body"]["messages"]
    joined = " ".join(str(m.get("content")) for m in second)
    assert "print_tree" in joined
    assert "not a tool in this phase's registry" in joined
    assert "apply_patch" in joined and "finish" in joined


def test_three_rejections_in_a_row_terminate_with_a_named_reason(
    tmp_path: Path,
) -> None:
    reject = json.dumps({"error": {
        "code": "tool_use_failed", "message": "bad arguments",
        "failed_generation": '{"name":"finish","arguments":{}}',
    }})
    post = Recorder(*(fail(400, reject) for _ in range(4)))
    llm = client(post, tmp_path)
    tools = [{"name": "finish", "description": "d",
             "parameters": {"type": "object", "properties": {}}}]

    outcome = run_agent(
        client=llm, task="go", tools=tools, handlers={"finish": lambda **k: "done"},
        terminal_tool="finish", budget=LoopBudget(max_steps=10))

    assert outcome.status is LoopStatus.ERROR
    assert len(post.calls) == 4, "three corrections, then the fourth is fatal"
    assert "400" in outcome.stopped_because


def test_each_correction_emits_a_trace_note(tmp_path: Path) -> None:
    reject = json.dumps({"error": {
        "code": "tool_use_failed", "message": "bad arguments",
        "failed_generation": '{"name":"finish","arguments":{}}',
    }})
    post = Recorder(fail(400, reject), ok(_FINISH_CALL))
    llm = client(post, tmp_path)
    tools = [{"name": "finish", "description": "d",
             "parameters": {"type": "object", "properties": {}}}]
    trace = Trace(tmp_path / "run", "r1")

    run_agent(client=llm, task="go", tools=tools,
             handlers={"finish": lambda **k: "done"}, terminal_tool="finish",
             budget=LoopBudget(max_steps=5), trace=trace, phase="patch")

    events = [json.loads(line) for line in
             (tmp_path / "run" / "trace.jsonl").read_text().splitlines()]
    notes = [e for e in events if e["type"] == "note" and e["phase"] == "patch"]
    assert any("the provider rejected a malformed tool call (1/3)" in e["note"]
              for e in notes)
