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
from ..errors import PerfAgentError
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
TOOL_ERROR_CODES = frozenset({"tool_use_failed", "invalid_tool_call"})


class LLMHTTPError(PerfAgentError):
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


class CassetteMiss(PerfAgentError):
    """offline=True and this exact request has never been recorded."""


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


class Cassette:
    """Recorded request/response pairs, keyed by the hash of the request body.

    This is what makes agent tests deterministic and free. It is keyed on the
    FULL body — system prompt, every message, every tool schema — so a prompt
    edit is a cassette miss rather than a stale replay.
    """

    def __init__(self, directory: Path | str, *, offline: bool = False) -> None:
        self.dir = Path(directory)
        self.offline = offline

    def key(self, provider: str, url: str, body: dict[str, Any]) -> str:
        return content_hash({"provider": provider, "url": url, "body": body})

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
            cassette_dir or Path(".perfagent/cassettes"), offline=offline
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
            raise CassetteMiss(
                f"offline=True and no cassette matches request {key} "
                f"({self.provider}/{self.model}). Expected it at "
                f"{self.cassette.path(key)}. Re-run once without --offline to "
                f"record it, or check whether a prompt change invalidated it."
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

    def correction_for(self, error: Exception) -> str | None:
        """A message that turns a provider-side rejection back into a turn.

        Returns None for anything that is not the model's fault — a 500, a
        rate limit, a bad key. Those are not correctable by saying something
        different, and pretending otherwise would spend the budget repeating
        a request that cannot work.
        """
        if not isinstance(error, LLMHTTPError):
            return None
        if str(error.payload.get("code")) not in TOOL_ERROR_CODES:
            return None

        message = str(error.payload.get("message", "")).strip()
        attempted = str(error.payload.get("failed_generation", "")).strip()
        return (
            f"{self.provider} rejected your last tool call before it reached "
            f"the tool, because the arguments did not match the tool's "
            f"schema.\n\nWhat the provider said:\n  {message[:600]}\n\n"
            + (f"What you sent:\n  {attempted[:600]}\n\n" if attempted else "")
            + "Nothing ran and nothing changed. Read the tool's parameter "
              "names again in the tool definitions above — they are exact — "
              "and call it again with every required property present."
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
