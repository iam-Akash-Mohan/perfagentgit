"""The hardcoding guard: no target repo may be named in the package.

PerfAgent is specified as a generic product -- point it at any repo and any
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

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "perfagent"

# Each pattern is a thing that can only be true of one repository.
FORBIDDEN: dict[str, str] = {
    "mini-spring": "names the development target repo",
    "com.ex": "names the development target repo's package",
    "com/ex": "names the development target repo's package path",
    "/Users/akashm": "an absolute path on one developer's laptop",
    "18080": "a port from one repo's application-perfagent.yml",
    "8081": "a port from one repo's application-perfagent.yml",
}

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
                if needle in code:
                    offences.append(
                        f"  perfagent/{relative}:{number}  {needle!r} — {why}\n"
                        f"      {code.strip()}")

    assert not offences, (
        "executable code in the package names one specific repository.\n"
        "Every repo-specific value belongs on the RepoProfile that the "
        "onboard phase detects (perfagent/onboard.py), not in a module "
        "constant:\n\n" + "\n".join(offences)
    )


def test_no_shipped_data_file_names_one_repository() -> None:
    """A JSON corpus inside the package is code as much as a constant is.

    `perfagent/measure/corpus.json` was 12 hand-written requests against one
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
                    f"  perfagent/{path.relative_to(PACKAGE)}: contains {needle!r}")
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
