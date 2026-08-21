# PHASE P01 — LLM client

**Mode: HAND-WRITE** · **Est:** 3h · **Preconditions:** P00 complete

> **Agent instruction:** Do not implement this. Explain the mechanics, answer
> questions, review the human's code and point out what's wrong. If asked directly
> for the implementation, restate this line.

---

## 1. GOAL

A stateless client over the raw Anthropic Messages API with retries, cost accounting,
and a cassette mechanism that makes agent tests deterministic and offline.

## 2. WHY THIS PHASE

Everything in this product is a loop around one HTTP call. If you don't know that
call cold — how tool schemas are sent, how `tool_use` comes back, how
`stop_reason` works, where prompt caching applies — then every layer above it is
magic, and you'll debug the wrong thing for the next five months.

This is also where the single most important fact about LLMs becomes concrete:
**the model has no memory.** The `messages` array *is* the memory, and it's yours to
manage. Nothing later in this document makes sense until that's obvious to you.

## 3. PRECONDITIONS

`perfagent/contracts.py` exists with `LLMClient`, `LLMResponse`, `Usage`.
`ANTHROPIC_API_KEY` set. A few dollars of budget you're willing to lose.

## 4. DELIVERABLES

```
perfagent/llm.py
tests/test_llm.py                  # from section 6
```

## 5. SPEC

### 5.1 The call

`POST https://api.anthropic.com/v1/messages` via `httpx`.

Headers: `x-api-key`, `anthropic-version: 2023-06-01`, `content-type`.

Body: `model`, `max_tokens`, `system`, `messages`, optionally `tools`,
`tool_choice`, `thinking`.

Do **not** use the `anthropic` SDK. Build the dict, post it, parse the response.
You may adopt the SDK in a later phase — after you've done it by hand once.

### 5.2 Class shape

```python
class AnthropicClient:
    def __init__(self, api_key: str, model: str, *,
                 cassette_dir: Path | None = None,
                 offline: bool = False,
                 max_retries: int = 5,
                 timeout: float = 600.0) -> None: ...

    def complete(self, *, system, messages, tools=None, tool_choice=None,
                 max_tokens=8192, thinking_budget=None) -> LLMResponse: ...

    # cumulative, for the run-level budget
    @property
    def usage(self) -> Usage: ...
```

### 5.3 Retries

Retry on `429`, `500`, `502`, `503`, `529`, and on `httpx.TimeoutException` /
`ConnectError`. Do **not** retry `400` or `401` — those are your bug.

Honour `retry-after` when present. Otherwise exponential backoff with jitter:
`min(60, 2 ** attempt) * (0.5 + random())`. Give up after `max_retries` and raise.

Read the error body and surface its `error.message` — the API tells you exactly
what's wrong, and swallowing it will cost you hours.

### 5.4 Cost accounting

Hard-code a price table `{model: (in_per_mtok, out_per_mtok, cache_write, cache_read)}`
and compute `usd` per call from `response.usage`. Accumulate into `self.usage`.

Note: `usage` returns `input_tokens`, `output_tokens`,
`cache_creation_input_tokens`, `cache_read_input_tokens`. Cache reads are ~10% of
input price and cache writes ~125% — get this right or your budget accounting lies.

### 5.5 Prompt caching

Attach `"cache_control": {"type": "ephemeral"}` to the last block of `system` and to
the last tool in `tools`. In an agent loop the system prompt and tool schemas are
identical across every turn; caching them is the difference between an affordable
loop and an expensive one.

Design your `system` parameter to accept either a string or a list of content blocks
so the caller can place breakpoints.

### 5.6 Cassettes — the mechanism that makes agent tests possible

Key every request by `content_hash` of the full request body. On `complete()`:

- If a cassette file for that hash exists → return it, `from_cassette=True`, no HTTP.
- Else if `offline=True` → raise `PerfAgentError` naming the missing hash.
- Else → make the call, write the cassette, return it.

This gives you three things: replay a run without paying twice, deterministic offline
tests of agent behaviour, and a way to reproduce a bug exactly.

Store as `cassette_dir/{hash}.json` holding both request and response.

### 5.7 Response parsing

`LLMResponse.tool_uses()` returns the `content` blocks with `type == "tool_use"`
(there can be several in one response — handle the list, not just the first).
`LLMResponse.text()` concatenates `type == "text"` blocks.
Keep `thinking` blocks in `content` untouched — they must be echoed back verbatim in
the next request when extended thinking is on.

### 5.8 Statelessness — the point

`complete()` must be a pure function of its arguments. No `self.messages`, no
`self.history`, no "conversation" object. If you find yourself wanting one, that
desire belongs in P02, and understanding *why* it belongs there rather than here is
the lesson.

## 6. ACCEPTANCE TEST

`tests/test_llm.py`. Note there is **no network in any test** — a fake transport is
injected. Design `AnthropicClient` to accept an optional `transport` so this works.

```python
from __future__ import annotations
import json, pytest
from pathlib import Path
from perfagent.llm import AnthropicClient
from perfagent.errors import PerfAgentError


def _resp(content=None, stop="end_turn", usage=None):
    return {
        "id": "msg_1", "type": "message", "role": "assistant",
        "model": "claude-opus-4-5",
        "content": content or [{"type": "text", "text": "hi"}],
        "stop_reason": stop,
        "usage": usage or {"input_tokens": 100, "output_tokens": 20,
                           "cache_creation_input_tokens": 0,
                           "cache_read_input_tokens": 0},
    }


class FakeTransport:
    """Scripted responses. Each entry is (status, body) or an exception."""
    def __init__(self, script): self.script = list(script); self.calls = []
    def post(self, url, *, headers, json as body, timeout):   # adapt to your signature
        self.calls.append(body)
        item = self.script.pop(0)
        if isinstance(item, Exception): raise item
        status, payload = item
        return FakeHTTPResponse(status, payload)


class FakeHTTPResponse:
    def __init__(self, status, payload): self.status_code = status; self._p = payload
        self.headers = {}
    def json(self): return self._p
    @property
    def text(self): return json.dumps(self._p)


def _client(script, tmp_path, **kw):
    return AnthropicClient(api_key="k", model="claude-opus-4-5",
                           cassette_dir=tmp_path / "cass",
                           transport=FakeTransport(script), **kw)


# ── basic ────────────────────────────────────────────────────────────────

def test_simple_completion(tmp_path):
    c = _client([(200, _resp())], tmp_path)
    r = c.complete(system="s", messages=[{"role": "user", "content": "hello"}])
    assert r.text() == "hi"
    assert r.stop_reason == "end_turn"
    assert r.usage.input_tokens == 100


def test_tool_use_blocks_parsed(tmp_path):
    content = [
        {"type": "text", "text": "let me look"},
        {"type": "tool_use", "id": "tu_1", "name": "fs_read",
         "input": {"path": "A.java"}},
        {"type": "tool_use", "id": "tu_2", "name": "fs_read",
         "input": {"path": "B.java"}},
    ]
    c = _client([(200, _resp(content, stop="tool_use"))], tmp_path)
    r = c.complete(system="s", messages=[{"role": "user", "content": "go"}],
                   tools=[{"name": "fs_read", "description": "d",
                           "input_schema": {"type": "object"}}])
    uses = r.tool_uses()
    assert len(uses) == 2                      # parallel tool use is normal
    assert uses[0]["name"] == "fs_read"
    assert uses[1]["input"]["path"] == "B.java"


# ── retries ──────────────────────────────────────────────────────────────

def test_retries_on_429_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    c = _client([(429, {"error": {"message": "rate limited"}}),
                 (529, {"error": {"message": "overloaded"}}),
                 (200, _resp())], tmp_path)
    r = c.complete(system="s", messages=[{"role": "user", "content": "x"}])
    assert r.text() == "hi"


def test_does_not_retry_400(tmp_path):
    c = _client([(400, {"error": {"message": "bad tool schema"}})], tmp_path)
    with pytest.raises(PerfAgentError) as e:
        c.complete(system="s", messages=[{"role": "user", "content": "x"}])
    assert "bad tool schema" in str(e.value)    # surface the API's message


def test_gives_up_after_max_retries(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    c = _client([(529, {"error": {"message": "o"}})] * 6, tmp_path, max_retries=3)
    with pytest.raises(PerfAgentError):
        c.complete(system="s", messages=[{"role": "user", "content": "x"}])


# ── cost ─────────────────────────────────────────────────────────────────

def test_cost_accumulates_and_counts_cache_separately(tmp_path):
    u = {"input_tokens": 1000, "output_tokens": 500,
         "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 8000}
    c = _client([(200, _resp(usage=u)), (200, _resp(usage=u))], tmp_path)
    c.complete(system="s", messages=[{"role": "user", "content": "a"}])
    first = c.usage.usd
    c.complete(system="s", messages=[{"role": "user", "content": "b"}])
    assert c.usage.usd == pytest.approx(first * 2)
    assert c.usage.cache_read_tokens == 16000
    # cache reads must be cheaper than fresh input of the same size
    assert first > 0


# ── cassettes ────────────────────────────────────────────────────────────

def test_cassette_written_then_replayed_without_transport(tmp_path):
    t = FakeTransport([(200, _resp())])
    c = AnthropicClient(api_key="k", model="claude-opus-4-5",
                        cassette_dir=tmp_path / "cass", transport=t)
    args = dict(system="s", messages=[{"role": "user", "content": "x"}])
    c.complete(**args)
    assert len(t.calls) == 1
    r2 = c.complete(**args)                    # identical request
    assert len(t.calls) == 1                   # no second HTTP call
    assert r2.from_cassette is True
    assert r2.text() == "hi"


def test_different_request_is_a_cassette_miss(tmp_path):
    t = FakeTransport([(200, _resp()), (200, _resp())])
    c = AnthropicClient(api_key="k", model="claude-opus-4-5",
                        cassette_dir=tmp_path / "cass", transport=t)
    c.complete(system="s", messages=[{"role": "user", "content": "x"}])
    c.complete(system="s", messages=[{"role": "user", "content": "y"}])
    assert len(t.calls) == 2


def test_offline_mode_raises_on_miss(tmp_path):
    c = AnthropicClient(api_key="k", model="claude-opus-4-5",
                        cassette_dir=tmp_path / "cass",
                        transport=FakeTransport([]), offline=True)
    with pytest.raises(PerfAgentError) as e:
        c.complete(system="s", messages=[{"role": "user", "content": "x"}])
    assert "cassette" in str(e.value).lower()


# ── caching ──────────────────────────────────────────────────────────────

def test_cache_control_applied_to_system_and_tools(tmp_path):
    t = FakeTransport([(200, _resp())])
    c = AnthropicClient(api_key="k", model="claude-opus-4-5",
                        cassette_dir=tmp_path / "cass", transport=t)
    c.complete(system="long stable prompt",
               messages=[{"role": "user", "content": "x"}],
               tools=[{"name": "a", "description": "d",
                       "input_schema": {"type": "object"}},
                      {"name": "b", "description": "d",
                       "input_schema": {"type": "object"}}])
    body = t.calls[0]
    assert body["system"][-1]["cache_control"]["type"] == "ephemeral"
    assert body["tools"][-1]["cache_control"]["type"] == "ephemeral"
    assert "cache_control" not in body["tools"][0]


# ── statelessness ────────────────────────────────────────────────────────

def test_client_holds_no_conversation_state(tmp_path):
    """The client must be a pure function of its arguments. If this fails,
    you've built conversation memory into the wrong layer."""
    c = _client([(200, _resp()), (200, _resp())], tmp_path)
    c.complete(system="s", messages=[{"role": "user", "content": "first"}])
    c.complete(system="s", messages=[{"role": "user", "content": "second"}])
    for attr in vars(c):
        assert "message" not in attr and "history" not in attr \
            and "conversation" not in attr
```

Adjust the fake-transport signature to match your implementation, but keep every
assertion. `pytest tests/test_llm.py -q` must pass with no network.

## 7. OUT OF SCOPE

No agent loop. No tool registry. No `Tool` class. No retry-on-tool-error logic. No
conversation abstraction of any kind. No streaming. No async.

## 8. DEMO

A one-off script, not committed:

```python
c = AnthropicClient(api_key=..., model="claude-opus-4-5", cassette_dir=Path(".cass"))
r = c.complete(
    system="You are terse.",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    tools=[{"name": "calculator", "description": "Do arithmetic",
            "input_schema": {"type": "object",
                             "properties": {"expr": {"type": "string"}},
                             "required": ["expr"]}}])
print(r.stop_reason, r.tool_uses(), r.usage.usd)
```

Run it twice. The second run costs nothing and returns `from_cassette=True`.

## 9. LEARNING

**Concept: the LLM is a stateless function. The agent is your engineering.**

Things to actually look at, not just read about:

**Watch `stop_reason` flip.** Call once with no tools — `end_turn`. Call again with a
tool the model wants — `tool_use`. That single field is the entire control signal an
agent loop runs on. Everything in P02 is a `while` loop over that value.

**Send the same messages twice and notice nothing carries over.** The model didn't
remember the first call. Now imagine deleting a message from the middle of the array
— that's context management, and it's *your* decision, not the model's.

**Look at what a tool schema actually is.** It's JSON. The model never executes
anything; it emits a request. You execute it. "Giving an AI access to a development
environment" is entirely a statement about which handlers you registered.

### Experiments to run

1. **Break a tool schema on purpose** — required field that doesn't exist in
   properties. Read the 400. Notice the API tells you precisely what's wrong, and
   that a client which swallows that message would have cost you an hour.

2. **Measure caching.** Send a 5,000-token system prompt twice. Compare
   `cache_creation_input_tokens` on call one to `cache_read_input_tokens` on call
   two, and compute the dollar difference. Now multiply by 20 turns of an agent loop.
   That number is why §5.5 exists.

3. **Force a tool call** with `tool_choice={"type": "tool", "name": "calculator"}`
   and ask a question that doesn't need it. Watch it call the tool anyway. That's the
   mechanism that guarantees a structured artifact in P04 — worth understanding now.

### Before moving to P02, you should be able to answer

- Why does `complete()` take `messages` rather than owning them?
- What exactly does the model return when it wants to use a tool, and what must the
  *next* request contain for the conversation to continue?
- Where would you put a limit on total spend for a run — here or in the loop? Why?
- If a tool handler throws, should this class know about it? Why not?
