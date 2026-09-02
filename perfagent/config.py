from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

CONFIG_FILE = "perfagent.yml"

_ENV_PREFIX = "PERFAGENT_"


class Settings(BaseModel):
    """Runtime configuration. Env wins over perfagent.yml, which wins over defaults."""

    anthropic_api_key: str | None = None
    model: str = "claude-opus-4-5"
    cheap_model: str = "claude-haiku-4-5"
    runs_dir: Path = Field(default=Path(".perfagent/runs"))
    cassette_dir: Path = Field(default=Path(".perfagent/cassettes"))
    max_usd_per_run: float = 5.0
    offline: bool = False

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

        return cls.model_validate(values)
