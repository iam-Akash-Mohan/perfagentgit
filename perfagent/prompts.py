"""Prompts as data, not string literals.

Prompts get tuned dozens of times. Keeping them in Python means every tuning
pass is a code diff, they cannot be read by anyone who does not read Python,
and — worst — there is no way to tell afterwards WHICH version of a prompt
produced a given artifact.

So: one file per prompt, substituted here, and the file's content hash recorded
in the trace and on the artifact. A run is then reproducible in the only sense
that matters — you can tell what it was actually asked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ContractViolation
from .ids import content_hash
from .trace import EventType, Trace

# Deliberately not str.format: a prompt containing a JSON example is full of
# braces that are not placeholders, and str.format would either explode on
# them or silently eat them. A placeholder is {lower_snake_case} and nothing
# else.
PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@dataclass(frozen=True)
class Prompt:
    """A rendered prompt and the identity of the file it came from."""

    name: str
    path: Path
    text: str
    hash: str          # of the FILE, before substitution — the version id

    def __str__(self) -> str:
        return self.text


def prompts_dir(override: Path | str | None = None) -> Path:
    return Path(override) if override is not None else PROMPTS_DIR


def load_prompt(
    name: str,
    *,
    directory: Path | str | None = None,
    trace: Trace | None = None,
    phase: str = "understand",
    **substitutions: Any,
) -> Prompt:
    """Read `prompts/<name>.md`, substitute placeholders, record the version.

    A placeholder in the file with no value supplied is an error, and so is a
    value supplied for a placeholder the file does not contain. Both are the
    same bug — the prompt and its caller have drifted apart — and both produce
    a prompt that silently says the wrong thing.
    """
    path = prompts_dir(directory) / f"{name}.md"
    if not path.exists():
        raise ContractViolation(
            f"no prompt file at {path}. Prompts live in {prompts_dir(directory)}; "
            f"create {name}.md there."
        )

    raw = path.read_text(encoding="utf-8")
    digest = content_hash(raw)

    wanted = set(PLACEHOLDER.findall(raw))
    given = set(substitutions)

    missing = sorted(wanted - given)
    if missing:
        raise ContractViolation(
            f"{path} needs {', '.join('{' + m + '}' for m in missing)} but no "
            f"value was supplied. A blank placeholder would change what the "
            f"model is asked without anyone noticing."
        )
    unused = sorted(given - wanted)
    if unused:
        raise ContractViolation(
            f"{', '.join(unused)} was passed to prompt '{name}' but {path} "
            f"contains no such placeholder. The prompt and its caller have "
            f"drifted apart."
        )

    text = PLACEHOLDER.sub(lambda m: str(substitutions[m.group(1)]), raw)

    if trace is not None:
        trace.emit(EventType.PROMPT_LOADED, phase=phase, prompt=name,
                   prompt_hash=digest, path=str(path), chars=len(text))

    return Prompt(name=name, path=path, text=text, hash=digest)
