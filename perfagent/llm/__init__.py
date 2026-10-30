"""One factory, one switch.

Nothing outside this package knows a base URL or an environment variable name.
That is the whole point: switching provider is a config edit, and acceptance
criterion 7 of session S1 is exactly the claim that no code has to change.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ..config import (PROVIDER_DEFAULT_MODEL, PROVIDER_KEY_ENV,
                      Settings)
from ..errors import PerfAgentError
from .anthropic import AnthropicClient
from .base import (Cassette, CassetteMiss, HTTPLLMClient, LLMHTTPError,
                   normalise_tools, tool_names)
from .openai_compat import OpenAICompatClient
from .pricing import usd_for

__all__ = [
    "AnthropicClient", "Cassette", "CassetteMiss", "HTTPLLMClient",
    "LLMHTTPError", "OpenAICompatClient", "GroqClient", "make_client",
    "normalise_tools", "tool_names", "usd_for", "resolve_key",
]

OPENAI_COMPAT = ("cerebras", "groq", "gemini", "ollama")

# What the pre-S1 GroqClient defaulted to. Only the deprecated alias uses it.
LEGACY_MODEL = "gpt-oss-120b"


def resolve_key(provider: str) -> str | None:
    """The API key for a provider, or a clear error naming what is missing."""
    if provider not in PROVIDER_KEY_ENV:
        raise PerfAgentError(
            f"unknown LLM provider {provider!r}. "
            f"Known providers: {', '.join(sorted(PROVIDER_KEY_ENV))}."
        )
    env_names = PROVIDER_KEY_ENV[provider]
    if env_names is None:         # ollama runs locally and needs no key
        return None
    for env_name in env_names:
        key = os.environ.get(env_name)
        if key:
            return key

    # Name EVERY variable that would have worked. A user with GOOGLE_API_KEY
    # set and an error that says only GEMINI_API_KEY has been told the truth
    # and still cannot act on it.
    accepted = " or ".join(env_names)
    raise PerfAgentError(
        f"the {provider} provider needs {accepted}, and none of them is set. "
        f"Export one, put it in .env, or choose another provider with "
        f"--provider (one of {', '.join(sorted(PROVIDER_KEY_ENV))})."
    )


def _default_model(provider: str, config: Any) -> str:
    """The model to use when the caller named a provider but not a model.

    `config.model` is only the right answer for the provider it was written
    for. Asking for a different provider on the command line and silently
    keeping the configured model sends one vendor's model id to another
    vendor's API.
    """
    if provider != config.provider and provider in PROVIDER_DEFAULT_MODEL:
        return PROVIDER_DEFAULT_MODEL[provider]
    return str(config.model)


def make_client(
    provider: str | None = None,
    model: str | None = None,
    *,
    settings: Settings | None = None,
    max_tokens: int | None = None,
    timeout_s: float | None = None,
    cassette_dir: Path | str | None = None,
    offline: bool | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> HTTPLLMClient:
    """Build a client.

    Resolution order for every value: explicit argument → perfagent.yml →
    environment → built-in default. `Settings.load()` already folds the last
    three together, so this function only has to prefer its arguments.
    """
    load_dotenv()
    config = (settings or Settings.load()).llm

    chosen = (provider or config.provider).lower()
    common: dict[str, Any] = {
        "model": model or _default_model(chosen, config),
        "max_tokens": max_tokens if max_tokens is not None else config.max_tokens,
        "timeout_s": timeout_s if timeout_s is not None else config.timeout_s,
        "cassette_dir": cassette_dir if cassette_dir is not None
                        else config.cassette_dir,
        "offline": offline if offline is not None else config.offline,
        "base_url": base_url or config.base_url,
        "api_key": resolve_key(chosen),
        **kwargs,
    }

    if chosen == "anthropic":
        return AnthropicClient(**common)
    if chosen in OPENAI_COMPAT:
        return OpenAICompatClient(provider=chosen, **common)
    raise PerfAgentError(
        f"unknown LLM provider {chosen!r}. "
        f"Known providers: {', '.join(sorted(PROVIDER_KEY_ENV))}."
    )


def GroqClient(api_key: str | None = None, model: str | None = None,
               **kwargs: Any) -> HTTPLLMClient:
    """DEPRECATED alias for the Cerebras client.

    Kept for one release so that P09/P12, which construct it directly, keep
    working while their sessions land. The name was always wrong — it posted
    to Cerebras from the first commit.

    Its default model stays `gpt-oss-120b` — what the old class used — so that
    "deprecated alias" means the same behaviour, not merely a working import.
    """
    import warnings

    warnings.warn(
        "GroqClient is deprecated; use perfagent.llm.make_client(...). "
        "Despite the name it has always posted to Cerebras.",
        DeprecationWarning, stacklevel=2,
    )
    load_dotenv()
    config = Settings.load().llm
    return OpenAICompatClient(
        provider="cerebras",
        model=model or LEGACY_MODEL,
        api_key=api_key or os.environ.get("CEREBRAS_API_KEY"),
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
        cassette_dir=config.cassette_dir,
        offline=config.offline,
        **kwargs,
    )
