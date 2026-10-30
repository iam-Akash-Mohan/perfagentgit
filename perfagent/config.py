from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

CONFIG_FILE = "perfagent.yml"

_ENV_PREFIX = "PERFAGENT_"

# Where each provider's key lives, and what it is called in an error message.
# This table is the ONLY place a provider name maps to an environment
# variable; nothing outside llm/ and here needs to know.
PROVIDER_KEY_ENV: dict[str, str | None] = {
    "cerebras": "CEREBRAS_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,          # a local server; no key
}

DEFAULT_PROVIDER = "cerebras"
DEFAULT_MODEL = "gpt-oss-120b"

# A model id belongs to ONE provider. `DEFAULT_MODEL` is a Cerebras model, so
# `--provider gemini` with no `--model` would otherwise send "gpt-oss-120b" to
# Google and get a 404 that reads like a configuration bug. Only providers
# whose default differs from the global one appear here; everything else keeps
# the resolution order exactly as it was.
PROVIDER_DEFAULT_MODEL: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
}


class LLMSettings(BaseModel):
    """The `llm:` block of perfagent.yml.

    Every phase may override `model` — diagnosis and planning want a stronger
    model than mechanical summarisation, and that is a configuration decision
    rather than a code one.
    """

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    max_tokens: int = 4096
    timeout_s: float = 120.0
    cassette_dir: Path = Field(default=Path(".perfagent/cassettes"))
    offline: bool = False
    base_url: str | None = Field(
        default=None,
        description="Override the provider's endpoint. Ollama users on a "
                    "non-default port need this; nobody else should.")


class Settings(BaseModel):
    """Runtime configuration. Env wins over perfagent.yml, which wins over defaults."""

    anthropic_api_key: str | None = None
    model: str = "claude-opus-4-5"
    cheap_model: str = "claude-haiku-4-5"
    runs_dir: Path = Field(default=Path(".perfagent/runs"))
    cassette_dir: Path = Field(default=Path(".perfagent/cassettes"))
    max_usd_per_run: float = 5.0
    offline: bool = False
    llm: LLMSettings = Field(default_factory=LLMSettings)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Settings":
        values: dict[str, Any] = {}

        path = Path(config_path) if config_path else Path(CONFIG_FILE)
        if path.exists():
            loaded = yaml.safe_load(path.read_text()) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            values.update(loaded)

        env_map = {
            "anthropic_api_key": "ANTHROPIC_API_KEY",
            "model": _ENV_PREFIX + "MODEL",
            "cheap_model": _ENV_PREFIX + "CHEAP_MODEL",
            "runs_dir": _ENV_PREFIX + "RUNS_DIR",
            "cassette_dir": _ENV_PREFIX + "CASSETTE_DIR",
            "max_usd_per_run": _ENV_PREFIX + "MAX_USD_PER_RUN",
            "offline": _ENV_PREFIX + "OFFLINE",
        }
        for field, env_name in env_map.items():
            raw = os.environ.get(env_name)
            if raw is not None:
                values[field] = raw

        # The llm block is nested, so its env overrides merge INTO whatever
        # the file supplied rather than replacing the whole block.
        llm: dict[str, Any] = dict(values.get("llm") or {})
        llm_env = {
            "provider": _ENV_PREFIX + "LLM_PROVIDER",
            "model": _ENV_PREFIX + "LLM_MODEL",
            "max_tokens": _ENV_PREFIX + "LLM_MAX_TOKENS",
            "timeout_s": _ENV_PREFIX + "LLM_TIMEOUT_S",
            "cassette_dir": _ENV_PREFIX + "LLM_CASSETTE_DIR",
            "offline": _ENV_PREFIX + "LLM_OFFLINE",
            "base_url": _ENV_PREFIX + "LLM_BASE_URL",
        }
        for field, env_name in llm_env.items():
            raw = os.environ.get(env_name)
            if raw is not None:
                llm[field] = raw
        values["llm"] = llm

        return cls.model_validate(values)
