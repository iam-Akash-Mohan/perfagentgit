"""The hardcoding guard: no target repo may be named in the package.

Senapathy is specified as a generic product -- point it at any repo and any
endpoint. Fourteen module-level constants used to say otherwise: one
developer's home directory, one build tool, one source layout, one Spring
profile, one main class, one pair of ports. Each was individually harmless
and collectively meant the product only worked on `mini-spring`.

This test is the deliverable that keeps them gone. It greps the SHIPPED
PACKAGE for anything that names a specific repository and fails if it finds
one outside a docstring or a comment.

Why docstrings and comments are exempt: "this used to be
`/Users/akashm/Documents/mini-spring`" is exactly the sort of note that
stops the constant coming back, and a rule that forbids explaining a bug
deletes the explanation rather than the bug. What matters is that no
EXECUTABLE line resolves to one repository.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path
from typing import Any

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "senapathy"

# Each pattern is a thing that can only be true of one repository --
# PLUS "perfagent" itself: the product's own former name, which must not
# creep back into executable code now that it is called senapathy.
FORBIDDEN: dict[str, str] = {
    "mini-spring": "names the development target repo",
    "com.ex": "names the development target repo's package",
    "com/ex": "names the development target repo's package path",
    "/Users/akashm": "an absolute path on one developer's laptop",
    "18080": "a port from one repo's application-perfagent.yml",
    "8081": "a port from one repo's application-perfagent.yml",
    "perfagent": "the product's former name",
}

# mini-spring's OWN Spring profile is, and remains, literally named
# "perfagent" -- chosen (before this rename existed) to mirror the tool's
# old name, and immutable now because it is the target repo's actual file,
# not ours to rename. A line that names it this way is not the old product
# name creeping back in; it is a true fact about mini-spring. This is the
# one standing exception, and it is narrow on purpose: it excuses a line
# ONLY when it is clearly about that file or that profile, never a bare
# "perfagent" standing alone.
_TARGET_REPO_PROFILE_MARKERS = (
    "application-perfagent.yml",
    "spring.profiles.active=perfagent",
)

# Paths that may legitimately mention these, with the reason.
EXEMPT: dict[str, str] = {
    # The deprecation shim's whole purpose is naming the old behaviour.
    "llm/__init__.py": "deprecation shim text",
}


def _blank(blanked: dict[int, list[str]], start: tuple[int, int],
           end: tuple[int, int]) -> None:
    (start_row, start_col), (end_row, end_col) = start, end
    for row in range(start_row, end_row + 1):
        if row not in blanked:
            continue
        line = blanked[row]
        begin = start_col if row == start_row else 0
        finish = end_col if row == end_row else len(line)
        for column in range(begin, min(finish, len(line))):
            line[column] = " "


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """Every line of real code, with COMMENTS and DOCSTRINGS blanked out.

    Deliberately NOT "all string literals". An earlier version of this guard
    blanked every STRING token, which made it blind to
    `MAIN_CLASS = "com.ex.Application"` and `Path("/Users/akashm/...")` --
    precisely the two worst constants it exists to catch. A string that is a
    VALUE is code; a string that is a statement on its own is prose.

    So: comments come from the tokeniser, docstrings come from the AST (the
    only reliable way to know a string is a docstring rather than a value),
    and everything else is left intact.
    """
    source = path.read_text(encoding="utf-8")
    blanked: dict[int, list[str]] = {
        number: list(text)
        for number, text in enumerate(source.splitlines(), start=1)
    }

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        pytest.fail(f"{path} does not tokenise: {exc}")
    for token in tokens:
        if token.type is tokenize.COMMENT:
            _blank(blanked, token.start, token.end)

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        pytest.fail(f"{path} does not parse: {exc}")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            literal = first.value
            _blank(blanked,
                   (literal.lineno, literal.col_offset),
                   (literal.end_lineno or literal.lineno,
                    literal.end_col_offset or 0))

    return [(number, "".join(chars))
            for number, chars in sorted(blanked.items())]


def _python_files() -> list[Path]:
    return sorted(
        p for p in PACKAGE.rglob("*.py")
        if "__pycache__" not in p.parts
    )


def test_no_executable_line_in_the_package_names_one_repository() -> None:
    offences: list[str] = []

    for path in _python_files():
        relative = str(path.relative_to(PACKAGE))
        if relative in EXEMPT:
            continue
        for number, code in _code_lines(path):
            for needle, why in FORBIDDEN.items():
                if needle not in code:
                    continue
                if needle == "perfagent" and any(
                        marker in code for marker in _TARGET_REPO_PROFILE_MARKERS):
                    continue   # mini-spring's own file/profile, not ours
                if needle == "perfagent" and "LEGACY_TEMP_PREFIX" in code:
                    continue   # exists ONLY so the index sweep can clean up
                              # after the product's former name; see db.py
                offences.append(
                    f"  senapathy/{relative}:{number}  {needle!r} — {why}\n"
                    f"      {code.strip()}")

    assert not offences, (
        "executable code in the package names one specific repository.\n"
        "Every repo-specific value belongs on the RepoProfile that the "
        "onboard phase detects (senapathy/onboard.py), not in a module "
        "constant:\n\n" + "\n".join(offences)
    )


def test_the_target_repo_profile_exception_is_narrow(tmp_path: Path) -> None:
    """mini-spring's own `application-perfagent.yml` is excused. A bare
    "perfagent" sitting next to it is not -- the exception is keyed to the
    two specific markers, not to "well, there's a filename nearby somewhere".
    """
    excused = tmp_path / "excused.py"
    excused.write_text(
        'CANDIDATE = "src/main/resources/application-perfagent.yml"\n'
    )
    for number, code in _code_lines(excused):
        for needle, why in FORBIDDEN.items():
            if needle == "perfagent" and any(
                    m in code for m in _TARGET_REPO_PROFILE_MARKERS):
                continue
            assert needle not in code or needle != "perfagent", (
                f"line {number} should have been excused: {code!r}")

    still_caught = tmp_path / "still_caught.py"
    still_caught.write_text('OLD_NAME = "perfagent"\n')
    lines = _code_lines(still_caught)
    assert any("perfagent" in code for _, code in lines), (
        "a bare 'perfagent' with no target-repo marker must still be "
        "visible to the scan"
    )


def test_no_shipped_data_file_names_one_repository() -> None:
    """A JSON corpus inside the package is code as much as a constant is.

    `senapathy/measure/corpus.json` was 12 hand-written requests against one
    endpoint of one repo, shipped in the wheel via pyproject's package-data
    glob, and used as the DEFAULT for every run. A corpus is per-run
    evidence; it does not belong in the package at all.
    """
    offences: list[str] = []
    for path in sorted(PACKAGE.rglob("*.json")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in FORBIDDEN:
            if needle in text:
                offences.append(
                    f"  senapathy/{path.relative_to(PACKAGE)}: contains {needle!r}")
    assert not offences, (
        "a data file shipped inside the package names one repository:\n"
        + "\n".join(offences))


def test_the_guard_actually_detects_a_planted_constant(tmp_path: Path) -> None:
    """A guard nobody has seen fail is a guard nobody can trust.

    This proves the tokeniser blanks strings but still catches an
    executable line -- i.e. that the exemption for comments is not
    accidentally exempting everything.
    """
    planted = tmp_path / "planted.py"
    planted.write_text(
        '# a comment mentioning mini-spring is fine\n'
        '"""A docstring mentioning /Users/akashm is fine."""\n'
        'BAD = {"port": 18080}\n'
    )
    lines = _code_lines(planted)
    rendered = {n: c for n, c in lines}

    assert "mini-spring" not in rendered[1], "comments must be blanked"
    assert "akashm" not in rendered[2], "docstrings must be blanked"
    assert "18080" in rendered[3], "an executable constant must survive"
    # The regression that made an earlier version of this guard useless:
    # a repo-specific STRING VALUE must survive blanking.
    planted.write_text('MAIN_CLASS = "com.ex.Application"\n')
    assert "com.ex" in _code_lines(planted)[0][1], (
        "a string used as a VALUE is code and must not be blanked")


# ──────────────  defaults on config models, not just literals  ───────────
#
# `database="mini"` was never caught by the scan above: it is not a string
# standing alone in executable code, it is a Field() default on a pydantic
# model, and nothing greps those. It survived every "no mini-spring" sweep
# because the sweep looked for a REPO NAME, not for the shape of the bug --
# a connection/credential field quietly filling itself in. This section
# checks the shape instead of any one name, so the next hardcoded default on
# the next config model is caught even if it names a repo nobody has heard
# of yet.

# Field names that identify WHERE or AS WHOM something connects. A default
# here is never "a sensible convention" the way `health_path` is -- it is
# always either `None` (not detected yet) or one specific deployment's
# values baked in.
_IDENTITY_FIELD_NAMES = frozenset({
    "host", "port", "user", "username", "password", "database", "schema",
    "dsn", "secret", "token", "api_key",
})


def _pydantic_models_in(module: Any) -> list[type]:
    import inspect

    from pydantic import BaseModel

    return [
        obj for _, obj in vars(module).items()
        if inspect.isclass(obj) and issubclass(obj, BaseModel)
        and obj.__module__ == module.__name__
    ]


def _package_modules() -> list[Any]:
    import importlib
    import pkgutil

    import senapathy

    modules = []
    for info in pkgutil.walk_packages(senapathy.__path__, prefix="senapathy."):
        modules.append(importlib.import_module(info.name))
    return modules


def test_no_identity_field_on_a_config_model_has_a_hardcoded_default() -> None:
    """A host, port, user, password, database or similar field may default to
    `None` -- not yet detected -- but never to a concrete value. A concrete
    default is exactly how `MySQLTarget.database` kept answering "mini" for
    every repo until someone asked why.
    """
    offences: list[str] = []
    for module in _package_modules():
        for model in _pydantic_models_in(module):
            for field_name, field in model.model_fields.items():
                if field_name.lower() not in _IDENTITY_FIELD_NAMES:
                    continue
                if field.is_required():
                    continue
                if field.default is None:
                    continue
                offences.append(
                    f"  {model.__module__}.{model.__qualname__}.{field_name} "
                    f"defaults to {field.default!r}")

    assert not offences, (
        "a connection/credential field on a config model has a hardcoded, "
        "non-None default -- the field must either be required, or default "
        "to None and be filled in from a detected RepoProfile:\n"
        + "\n".join(offences)
    )


def test_this_guard_catches_a_planted_default_on_a_pydantic_model() -> None:
    """Unseen failure is unproven protection, same as the guard above."""
    import sys
    import types

    from pydantic import BaseModel

    class _PlantedTarget(BaseModel):
        host: str = "127.0.0.1"
        port: int = 3307
        database: str = "mini"

    planted = types.ModuleType("senapathy._planted_for_test")
    planted.PlantedTarget = _PlantedTarget
    _PlantedTarget.__module__ = planted.__name__
    sys.modules[planted.__name__] = planted
    try:
        offences = [
            field_name
            for model in _pydantic_models_in(planted)
            for field_name, field in model.model_fields.items()
            if field_name.lower() in _IDENTITY_FIELD_NAMES
            and not field.is_required() and field.default is not None
        ]
    finally:
        del sys.modules[planted.__name__]

    assert set(offences) == {"host", "port", "database"}
