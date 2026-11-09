"""Everything the providers share: retries, cassettes, cost, containment.

The one rule this file exists to enforce is that a non-200 is never parsed as
a completion. The client it replaces printed the error body and then called
`r.json()` anyway, so a 429 arrived downstream as `KeyError: 'choices'` — an
exception that says nothing about rate limiting, three frames away from the
thing that would have.
"""

from __future__ import annotations

import json
import logging
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx

from ..contracts import LLMResponse, ToolCallResult, Usage
from ..errors import SenapathyError
from ..ids import content_hash
from . import pricing

log = logging.getLogger(__name__)

# Transient: the request was fine, the server was not. Worth repeating.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

# Our bug, not theirs. Repeating it wastes money and hides the message that
# would have explained it, so the provider's own words are surfaced verbatim.
NEVER_RETRY_STATUS = frozenset({400, 401, 403, 404, 405, 413, 422})

MAX_ATTEMPTS = 5
MAX_BACKOFF_S = 60.0
BASE_BACKOFF_S = 1.0


# Some providers validate tool calls SERVER-SIDE and reject the request rather
# than returning the malformed call. Groq does: a call whose arguments do not
# match the schema comes back as HTTP 400 `tool_use_failed`, so the loop's
# "a bad tool call is a correctable mistake, not a crash" guarantee never gets
# to run — the phase dies three frames away from anything that explains it.
# That is a MODEL mistake wearing a transport error's clothes.
#
# `output_parse_failed` is the same failure one level up: the model emitted
# PROSE where `tool_choice` demanded a call at all (observed in the plan
# phase — `failed_generation: "Open InvoiceEnricher?"`), and Groq rejects the
# request server-side instead of handing the prose back as a normal
# `end_turn`. Same cure: feed it back as a correctable mistake, not a crash.
#
# `json_validate_failed` is the third member of this family — Groq's own
# documented code for "the model's structured output broke the schema"
# (confirmed against reports from other Groq SDK users; there is no public
# errors page enumerating it, Groq's error-codes doc only lists HTTP
# statuses). Not observed in a run here yet, but it is the same shape as
# `tool_use_failed` -- a generation rejected before it reached the
# conversation -- so it gets the same treatment rather than waiting for a
# third fatal run to prove it.
TOOL_ERROR_CODES = frozenset({
    "tool_use_failed", "invalid_tool_call", "json_validate_failed",
    "output_parse_failed",
})


def _attempted_tool_name(failed_generation: str) -> str | None:
    """The tool name out of Groq's `failed_generation`, when it looks like
    the JSON tool-call shape it usually is for `tool_use_failed` /
    `invalid_tool_call` / `json_validate_failed`.

    Best-effort: a string that is not JSON, or that has no `name`, means
    "could not tell" rather than an error — the caller falls back to the
    generic schema-mismatch message in that case, which was the only
    message that existed before this.
    """
    try:
        parsed = json.loads(failed_generation)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed, dict):
        name = parsed.get("name")
        return str(name) if name else None
    return None


class LLMHTTPError(SenapathyError):
    """A non-200 that survived the retry policy."""

    def __init__(self, provider: str, status: int, body: str, url: str) -> None:
        self.provider = provider
        self.status = status
        self.body = body
        self.url = url
        try:
            self.payload: dict[str, Any] = dict(json.loads(body)).get("error", {})
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            self.payload = {}
        super().__init__(
            f"{provider} returned HTTP {status} from {url}. "
            f"The provider said:\n{body.strip()[:2000]}"
        )


class CassetteMiss(SenapathyError):
    """offline=True and this exact request has never been recorded.

    `recorded_correction`, when set, is not a guess: `Cassette.
    recover_self_correction()` found exactly one recorded conversation whose
    messages are this exact request plus one more `user`-role message, read
    verbatim from that cassette file. See that method's docstring for why
    this situation exists at all and why reading it back, rather than
    inventing it, is the only honest recovery.
    """

    def __init__(self, message: str, *,
                recorded_correction: str | None = None) -> None:
        super().__init__(message)
        self.recorded_correction = recorded_correction


def normalise_tools(tools: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Accept both the neutral shape and OpenAI's function-wrapped one.

    Neutral is `{"name", "description", "parameters"}`. Callers written against
    the old client pass `{"type": "function", "function": {...}}`; unwrapping
    here is what lets those callers keep working while the loop and the
    Anthropic client speak one vocabulary.
    """
    out: list[dict[str, Any]] = []
    for tool in tools or []:
        nested = tool.get("function")
        inner: dict[str, Any] = nested if isinstance(nested, dict) else tool
        out.append({
            "name": inner["name"],
            "description": inner.get("description", ""),
            "parameters": inner.get("parameters")
                          or inner.get("input_schema")
                          or {"type": "object", "properties": {}},
        })
    return out


def tool_names(tools: Iterable[dict[str, Any]] | None) -> list[str]:
    return [t["name"] for t in normalise_tools(tools)]


def retry_after_seconds(headers: Any) -> float | None:
    """Honour Retry-After when it is a plain number of seconds.

    HTTP-date form is deliberately not parsed: a malformed date would silently
    become "retry immediately", and every provider we speak to sends seconds.
    """
    try:
        raw: Any = headers.get("Retry-After") or headers.get("retry-after")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        value = float(str(raw).strip())
    except ValueError:
        return None
    return min(max(value, 0.0), MAX_BACKOFF_S)


def backoff_seconds(attempt: int, jitter: Callable[[], float] = random.random) -> float:
    """Exponential with jitter, capped. `attempt` is 1-based."""
    ceiling: float = min(BASE_BACKOFF_S * float(2 ** (attempt - 1)),
                         MAX_BACKOFF_S)
    return float(ceiling * (0.5 + 0.5 * jitter()))


#  Fields from `build_body()`'s output that do not belong in the cassette
# key: they bound or shape the WIRE CALL, not what is being asked. Diverging
# on one of these must never read as "this is a different request" -- the
# semantic request (model, messages, tools, tool_choice) is unchanged, and a
# cached response is still the right answer. `max_tokens` is the clean
# example: it is a response-length ceiling, resolved from `senapathy.yml`/
# `SENAPATHY_LLM_MAX_TOKENS`/a CLI flag at CLIENT-CONSTRUCTION time, wholly
# unrelated to the conversation's content -- yet because key() used to hash
# the full body wholesale, any future change to that config (a different
# default, a different working directory failing to find senapathy.yml, a
# flag passed in one invocation but not another) silently invalidated every
# existing cassette, with a miss that could land on the first, a middle, or
# the last turn of a conversation depending on nothing about the
# conversation itself. Add a field here only when it is TRULY incidental to
# wire transport, never semantic to the request -- `model` and `tool_choice`
# stay IN the key on purpose: switching models or forcing a different tool
# genuinely changes what is being asked, and reusing a cached answer across
# that difference would be a bug, not a convenience.
_KEY_EXCLUDED_BODY_FIELDS = frozenset({"max_tokens"})


class Cassette:
    """Recorded request/response pairs, keyed by the hash of the request body.

    This is what makes agent tests deterministic and free. It is keyed on the
    SEMANTIC body — system prompt, every message, every tool schema, the
    model, tool_choice — so a prompt edit is a cassette miss rather than a
    stale replay, but a change to incidental wire configuration
    (`_KEY_EXCLUDED_BODY_FIELDS`) is not.
    """

    def __init__(self, directory: Path | str, *, offline: bool = False) -> None:
        self.dir = Path(directory)
        self.offline = offline

    def key(self, provider: str, url: str, body: dict[str, Any]) -> str:
        semantic = {k: v for k, v in body.items()
                   if k not in _KEY_EXCLUDED_BODY_FIELDS}
        return content_hash({"provider": provider, "url": url, "body": semantic})

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def read(self, key: str) -> dict[str, Any] | None:
        path = self.path(key)
        if not path.exists():
            return None
        try:
            document: dict[str, Any] = json.loads(
                path.read_text(encoding="utf-8"))
            return dict(document["response"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            log.warning("cassette %s is unreadable; treating it as a miss", path)
            return None

    def write(self, key: str, provider: str, model: str,
              body: dict[str, Any], response: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        document = {
            "hash": key, "provider": provider, "model": model,
            "request": body, "response": response,
        }
        self.path(key).write_text(
            json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def recover_self_correction(self, messages: list[dict[str, Any]]) -> str | None:
        """Why a cassette miss can be a replay bug, not a stale recording.

        When the LIVE recording hit a malformed tool call, `run_agent`
        appended a `{"role": "user", "content": <the provider's own schema
        complaint>}` message and retried -- see `agent.py`'s
        `_correction_for`. That first, REJECTED attempt produced no 200
        response, so nothing was ever cassette-written for it; only the
        SECOND, corrected attempt succeeded and got recorded, with the
        correction message already sitting inside its own `request.messages`.
        A fresh replay reconstructs the conversation by appending exactly
        what `run_agent` itself appends -- the assistant's reply and the
        tool's result -- which is correct but is missing that one
        correction message the live run only added because of a rejection
        replay never triggers. The request replay computes is one message
        short of the one that was actually recorded next, and it stays
        short forever after, because nothing told it to catch up.

        This does not invent that message: it is read back VERBATIM from
        whichever later cassette file recorded it, by finding the one whose
        own request is this exact conversation plus exactly one more
        `user`-role message. If more than one file matches (ambiguous) or
        none does (this really is just a stale cassette), returns None --
        a wrong guess here would corrupt the replayed conversation instead
        of fixing it.
        """
        candidates: list[str] = []
        for path in self.dir.glob("*.json"):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                recorded = doc["request"]["messages"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue
            if (len(recorded) == len(messages) + 1
                    and recorded[:len(messages)] == messages
                    and recorded[len(messages)].get("role") == "user"):
                candidates.append(str(recorded[len(messages)].get("content") or ""))
        if len(candidates) != 1:
            return None
        return candidates[0]


class HTTPLLMClient(ABC):
    """Shared skeleton. Subclasses own the wire format and nothing else."""

    provider: str = "unknown"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        timeout_s: float = 120.0,
        cassette_dir: Path | str | None = None,
        offline: bool = False,
        max_attempts: int = MAX_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
        http_post: Callable[..., httpx.Response] | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or self.default_base_url()
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.offline = offline
        self.max_attempts = max_attempts
        self._sleep = sleep
        self._jitter = jitter
        self._post = http_post or httpx.post
        self.cassette = Cassette(
            cassette_dir or Path(".senapathy/cassettes"), offline=offline
        )
        # Bookkeeping only. `complete()` remains a pure function of its
        # arguments — nothing read here ever reaches a request.
        self.total = Usage()
        self.calls = 0

    # ── the provider-specific parts ──────────────────────────────────────

    @abstractmethod
    def default_base_url(self) -> str: ...

    @abstractmethod
    def endpoint(self) -> str: ...

    @abstractmethod
    def headers(self) -> dict[str, str]: ...

    @abstractmethod
    def build_body(
        self, *, system: str | list[dict[str, Any]] | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: dict[str, Any] | None,
        max_tokens: int,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def parse(self, raw: dict[str, Any], latency_ms: float,
              from_cassette: bool) -> LLMResponse: ...

    @abstractmethod
    def tool_result_messages(
        self, results: list[ToolCallResult]
    ) -> list[dict[str, Any]]: ...

    # ── the shared part ──────────────────────────────────────────────────

    def complete(
        self,
        *,
        system: str | list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        body = self.build_body(
            system=system,
            messages=messages,
            tools=normalise_tools(tools) if tools else None,
            tool_choice=tool_choice,
            max_tokens=max_tokens or self.max_tokens,
        )
        url = self.endpoint()
        key = self.cassette.key(self.provider, url, body)

        recorded = self.cassette.read(key)
        if recorded is not None:
            # Zero latency is the honest number: no time was spent.
            response = self.parse(recorded, 0.0, True)
            self._account(response)
            return response

        if self.offline:
            correction = self.cassette.recover_self_correction(
                body.get("messages") or [])
            raise CassetteMiss(
                f"offline=True and no cassette matches request {key} "
                f"({self.provider}/{self.model}). Expected it at "
                f"{self.cassette.path(key)}. Re-run once without --offline to "
                f"record it, or check whether a prompt change invalidated it.",
                recorded_correction=correction,
            )

        started = time.perf_counter()
        raw = self._request(url, body)
        latency_ms = (time.perf_counter() - started) * 1000.0

        self.cassette.write(key, self.provider, self.model, body, raw)
        response = self.parse(raw, latency_ms, False)
        self._account(response)
        return response

    def _account(self, response: LLMResponse) -> None:
        self.calls += 1
        self.total = Usage(
            input_tokens=self.total.input_tokens + response.usage.input_tokens,
            output_tokens=self.total.output_tokens + response.usage.output_tokens,
            cache_read_tokens=self.total.cache_read_tokens
                              + response.usage.cache_read_tokens,
            cache_write_tokens=self.total.cache_write_tokens
                               + response.usage.cache_write_tokens,
            usd=round(self.total.usd + response.usage.usd, 8),
        )

    def correction_for(
        self, error: Exception, *, tool_names: list[str] | None = None,
    ) -> str | None:
        """A message that turns a provider-side rejection back into a turn.

        Returns None for anything that is not the model's fault — a 500, a
        rate limit, a bad key. Those are not correctable by saying something
        different, and pretending otherwise would spend the budget repeating
        a request that cannot work.

        `tool_names` is this phase's registry for the call that just failed
        — optional, so a caller that does not have it (or a hand-rolled test
        double written against the old one-argument signature) still gets a
        correction, just without the registry check below. When it IS
        supplied and the attempted call named a tool that is not in it, that
        is a different mistake from bad arguments against a real tool — a
        hallucinated `print_tree` is not fixed by "re-read the schema," it
        is fixed by "that tool does not exist, here is what does."
        """
        if not isinstance(error, LLMHTTPError):
            return None
        code = str(error.payload.get("code"))
        if code not in TOOL_ERROR_CODES:
            return None

        message = str(error.payload.get("message", "")).strip()
        attempted = str(error.payload.get("failed_generation", "")).strip()

        if code == "output_parse_failed":
            # Not a malformed call -- no call at all. Saying "the arguments
            # did not match the schema" here would describe a mistake the
            # model did not make and leave the real one (prose instead of a
            # tool call) for it to guess at.
            what_went_wrong = (
                f"{self.provider} rejected your last reply before it reached "
                f"the conversation, because a tool call was required here and "
                f"none arrived — you wrote prose instead."
            )
            fix = ("Nothing ran and nothing changed. Call one of the tools "
                   "defined above instead of replying in prose.")
        else:
            attempted_name = _attempted_tool_name(attempted)
            if (attempted_name and tool_names is not None
                    and attempted_name not in tool_names):
                what_went_wrong = (
                    f"{self.provider} rejected your last tool call before it "
                    f"reached anything, because '{attempted_name}' is not a "
                    f"tool in this phase's registry."
                )
                fix = (
                    "Nothing ran and nothing changed. The tools available to "
                    f"you here are: {', '.join(tool_names)}. Call one of "
                    f"those instead."
                )
            else:
                what_went_wrong = (
                    f"{self.provider} rejected your last tool call before it "
                    f"reached the tool, because the arguments did not match "
                    f"the tool's schema."
                )
                fix = ("Nothing ran and nothing changed. Read the tool's "
                       "parameter names again in the tool definitions above "
                       "— they are exact — and call it again with every "
                       "required property present.")

        return (
            f"{what_went_wrong}\n\nWhat the provider said:\n  {message[:600]}\n\n"
            + (f"What you sent:\n  {attempted[:600]}\n\n" if attempted else "")
            + fix
        )

    def price(self, usage: Usage) -> float:
        return round(pricing.usd_for(self.model, usage), 8)

    def _request(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST with the retry policy. Returns the parsed body of a 200 only."""
        last: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._post(
                    url, headers=self.headers(), json=body, timeout=self.timeout_s
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
                if attempt == self.max_attempts:
                    break
                self._sleep(backoff_seconds(attempt, self._jitter))
                continue

            status = response.status_code
            if status == 200:
                try:
                    return dict(response.json())
                except (json.JSONDecodeError, ValueError) as exc:
                    raise LLMHTTPError(
                        self.provider, status,
                        f"a 200 whose body is not JSON: {response.text[:500]}", url,
                    ) from exc

            if status in NEVER_RETRY_STATUS:
                raise LLMHTTPError(self.provider, status, response.text, url)

            if status not in RETRYABLE_STATUS or attempt == self.max_attempts:
                raise LLMHTTPError(self.provider, status, response.text, url)

            wait = retry_after_seconds(response.headers)
            if wait is None:
                wait = backoff_seconds(attempt, self._jitter)
            log.info("%s HTTP %s; retrying in %.1fs (attempt %d/%d)",
                     self.provider, status, wait, attempt, self.max_attempts)
            self._sleep(wait)

        raise LLMHTTPError(
            self.provider, 0,
            f"no response after {self.max_attempts} attempts: "
            f"{type(last).__name__}: {last}",
            url,
        )
