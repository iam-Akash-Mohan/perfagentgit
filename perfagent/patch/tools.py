"""The three tools the patch agent gets, and the guardrails around them.

Each is a named tool with a JSON schema. `build` and `run_tests` take NO
arguments from the model and run one fixed argv list each, so there is no
string the model can supply that becomes part of a command. `apply_patch`
takes a path, and every refusal it can make is enforced here in code — never
asked for in the prompt.

Tools return error STRINGS rather than raising. A refusal is the most useful
context the agent will get, and it is only useful if it reaches the model: an
exception would end the run instead of teaching it something.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .worktree import GreenWorktree, build_env

# Writes are confined to this prefix. CLAUDE.md invariant 1 asks for an
# allowlist, not a denylist: a denylist protects the paths you thought of.
WRITE_ALLOWLIST = ("src/main/java",)
TEST_TREE = "src/test"

MAX_BUILD_LINES = 40
MAX_ITERATIONS = 8
BUILD_TIMEOUT_S = 600
TEST_TIMEOUT_S = 900

_SUREFIRE = re.compile(
    r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)"
)


def _first_lines(text: str, limit: int = MAX_BUILD_LINES) -> str:
    """First `limit` lines, VERBATIM. Compiler output is never summarised —
    it is the highest-value context the agent will ever receive."""
    lines = text.splitlines()
    if len(lines) <= limit:
        return text.rstrip()
    hidden = len(lines) - limit
    return "\n".join(lines[:limit]) + f"\n[... {hidden} further lines truncated ...]"


@dataclass
class TestCounts:
    """What `mvn test` actually executed, plus what is on disk.

    Two numbers because they fail differently. `executed` catches a patch that
    breaks or disables a test. `files` catches a patch that deletes one — which
    in this repo is the only one that can bite, since surefire's default
    includes (`*Test`, `Test*`, `*Tests`, `*TestCase`) do not match
    `CustomerSearchIT`, so `executed` is 0 before and after any patch.
    """

    __test__ = False   # a dataclass named Test*, not a pytest test class

    executed: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    files: int = 0

    @property
    def passed(self) -> int:
        return max(0, self.executed - self.failures - self.errors)

    def render(self) -> str:
        return (
            f"tests executed {self.executed}, passed {self.passed}, "
            f"failures {self.failures}, errors {self.errors}, "
            f"skipped {self.skipped}; test files on disk {self.files}"
        )


def count_test_files(green: Path) -> int:
    tree = green / TEST_TREE
    if not tree.exists():
        return 0
    return sum(1 for _ in tree.rglob("*.java"))


def parse_test_counts(output: str, green: Path) -> TestCounts:
    """Sum surefire's per-class lines. Arithmetic in Python, per invariant 5."""
    counts = TestCounts(files=count_test_files(green))
    for run, failures, errors, skipped in _SUREFIRE.findall(output):
        counts.executed += int(run)
        counts.failures += int(failures)
        counts.errors += int(errors)
        counts.skipped += int(skipped)
    return counts


@dataclass
class PatchTools:
    """Stateful tool surface for one patch run.

    Holds the iteration budget, the pre-patch test baseline, and a log of every
    call including refusals — so the CLI can show what the agent tried, not
    just what worked.
    """

    green: GreenWorktree
    baseline: TestCounts
    java_home: str | None = None
    max_iterations: int = MAX_ITERATIONS

    iterations: int = 0
    log: list[dict[str, Any]] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    guardrail_violations: list[str] = field(default_factory=list)

    edits_since_verify: int = 0
    last_build_ok: bool = False
    last_tests_ok: bool = False
    last_error: str = ""

    # ─────────────────────────────  apply_patch  ─────────────────────────

    def apply_patch(self, file: str, old_text: str, new_text: str) -> str:
        """Exact single-occurrence string replacement in one file."""
        refusal = self._check_writable(file)
        if refusal is not None:
            self.refusals.append(refusal)
            self._record("apply_patch", {"file": file}, refusal, ok=False)
            return refusal

        path = (self.green.path / file).resolve()
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            message = (
                f"REFUSED: cannot read {file}: {exc}. Check the path — it must be "
                f"relative to the repository root, e.g. "
                f"src/main/java/com/ex/service/CustomerService.java"
            )
            self.refusals.append(message)
            self._record("apply_patch", {"file": file}, message, ok=False)
            return message

        if not old_text:
            message = (
                "REFUSED: old_text is empty. apply_patch replaces an exact "
                "existing snippet; it cannot insert into an empty anchor. Pass "
                "the exact lines you want to replace."
            )
            self.refusals.append(message)
            self._record("apply_patch", {"file": file}, message, ok=False)
            return message

        occurrences = source.count(old_text)
        if occurrences == 0:
            message = (
                f"REFUSED: old_text was not found in {file}. Nothing was written.\n"
                f"WHAT TO DO NEXT: your old_text must match the file byte for byte, "
                f"including indentation and line breaks. The exact current contents "
                f"of {file} were included in your instructions — copy the snippet "
                f"from there rather than retyping it."
            )
            self.refusals.append(message)
            self._record("apply_patch", {"file": file}, message, ok=False)
            return message
        if occurrences > 1:
            message = (
                f"REFUSED: old_text appears {occurrences} times in {file}, so the "
                f"edit is ambiguous. Nothing was written.\n"
                f"WHAT TO DO NEXT: extend old_text with the surrounding lines until "
                f"it appears exactly once — include the enclosing method signature "
                f"or the preceding line."
            )
            self.refusals.append(message)
            self._record("apply_patch", {"file": file}, message, ok=False)
            return message

        path.write_text(source.replace(old_text, new_text, 1), encoding="utf-8")
        self.edits_since_verify += 1
        self.last_build_ok = False
        self.last_tests_ok = False

        line_number = source[: source.index(old_text)].count("\n") + 1
        result = (
            f"OK: {file} updated at line {line_number}. "
            f"{len(old_text.splitlines())} line(s) replaced with "
            f"{len(new_text.splitlines())}.\n"
            f"You must now call build, and then run_tests, before emit_patch."
        )
        self._record("apply_patch", {"file": file, "line": line_number}, result, ok=True)
        return result

    def _check_writable(self, file: str) -> str | None:
        """Every path refusal, in code. Returns None when the path is allowed."""
        candidate = Path(file)
        if candidate.is_absolute():
            return (
                f"REFUSED: {file} is an absolute path. Pass a path relative to the "
                f"repository root, e.g. src/main/java/com/ex/service/CustomerService.java"
            )

        resolved = (self.green.path / candidate).resolve()
        try:
            relative = resolved.relative_to(self.green.path)
        except ValueError:
            return (
                f"REFUSED: {file} resolves outside the worktree you may write to. "
                f"Nothing was written. Only files inside the repository are "
                f"writable, and only under {', '.join(WRITE_ALLOWLIST)}/."
            )

        posix = relative.as_posix()
        if posix.startswith(TEST_TREE):
            # Invariant 3, in code. A prompt asking nicely is not a guardrail.
            return (
                f"REFUSED: {posix} is a test file. Test files are immutable — they "
                f"are the evidence that behaviour did not change, so a patch may "
                f"never edit them.\n"
                f"WHAT TO DO NEXT: make the production code satisfy the existing "
                f"test. Edit only files under {', '.join(WRITE_ALLOWLIST)}/."
            )
        if not posix.startswith(WRITE_ALLOWLIST):
            return (
                f"REFUSED: {posix} is outside the write allowlist. Nothing was "
                f"written.\n"
                f"WHAT TO DO NEXT: this task may only change application source. "
                f"Edit files under {', '.join(WRITE_ALLOWLIST)}/ only — not the "
                f"build file, not configuration, not resources."
            )
        return None

    # ───────────────────────────────  build  ─────────────────────────────

    def build(self) -> str:
        """`mvn -q -DskipTests package` in the green worktree."""
        budget = self._spend_iteration("build")
        if budget is not None:
            return budget

        completed = self._maven(["-q", "-DskipTests", "package"], BUILD_TIMEOUT_S)
        output = (completed.stdout or "") + (completed.stderr or "")

        if completed.returncode == 0:
            self.last_build_ok = True
            self.last_error = ""
            result = "BUILD OK — compiled cleanly. Now call run_tests."
        else:
            self.last_build_ok = False
            # Verbatim, truncated only by line count. Never summarised.
            self.last_error = _first_lines(output)
            result = (
                f"BUILD FAILED (exit {completed.returncode}). Compiler output, "
                f"first {MAX_BUILD_LINES} lines, verbatim:\n\n{self.last_error}"
            )

        self._record("build", {}, result, ok=completed.returncode == 0)
        return result

    # ─────────────────────────────  run_tests  ───────────────────────────

    def run_tests(self) -> str:
        """`mvn -q test` in the green worktree, with the count guardrail."""
        if not self.last_build_ok:
            message = (
                "REFUSED: the last build did not succeed, so there is nothing to "
                "test. Call build first and fix any compiler errors."
            )
            self._record("run_tests", {}, message, ok=False)
            return message

        budget = self._spend_iteration("run_tests")
        if budget is not None:
            return budget

        completed = self._maven(["-q", "test"], TEST_TIMEOUT_S)
        output = (completed.stdout or "") + (completed.stderr or "")
        counts = parse_test_counts(output, self.green.path)

        # The guardrail: computed here, and it fails the phase. Not a warning,
        # and not something the model is asked to check.
        violation = self._check_test_counts(counts)
        if violation:
            self.guardrail_violations.append(violation)
            self.last_tests_ok = False
            self._record("run_tests", {}, violation, ok=False)
            return violation

        failed = counts.failures + counts.errors
        if completed.returncode == 0 and failed == 0:
            self.last_tests_ok = True
            self.edits_since_verify = 0
            self.last_error = ""
            result = (
                f"TESTS OK — {counts.render()}. "
                f"Baseline before your patch: {self.baseline.render()}. "
                f"You may now call emit_patch."
            )
        else:
            self.last_tests_ok = False
            self.last_error = _first_lines(output)
            result = (
                f"TESTS FAILED (exit {completed.returncode}) — {counts.render()}.\n"
                f"Output, first {MAX_BUILD_LINES} lines, verbatim:\n\n{self.last_error}"
            )

        self._record("run_tests", {"counts": counts.render()}, result, ok=self.last_tests_ok)
        return result

    def _check_test_counts(self, counts: TestCounts) -> str | None:
        if counts.files < self.baseline.files:
            return (
                f"GUARDRAIL VIOLATION: test files on disk dropped from "
                f"{self.baseline.files} to {counts.files}. A patch may not delete a "
                f"test. This run is over — restore the file and nothing else will "
                f"be accepted."
            )
        if counts.executed < self.baseline.executed:
            return (
                f"GUARDRAIL VIOLATION: tests executed dropped from "
                f"{self.baseline.executed} to {counts.executed}. A patch may not "
                f"reduce the number of tests that run. This run is over."
            )
        return None

    # ──────────────────────────────  internals  ──────────────────────────

    def _spend_iteration(self, tool: str) -> str | None:
        """One shared budget across build and run_tests."""
        if self.iterations >= self.max_iterations:
            message = (
                f"REFUSED: the build/test budget of {self.max_iterations} iterations "
                f"is exhausted. No further {tool} is possible. If you have a working "
                f"patch, call emit_patch now; otherwise this run will be abandoned."
            )
            self._record(tool, {}, message, ok=False)
            return message
        self.iterations += 1
        return None

    def _maven(self, args: list[str], timeout_s: int) -> subprocess.CompletedProcess:
        """Fixed argv, no shell, no model-supplied token anywhere in it."""
        try:
            return subprocess.run(
                ["mvn", *args],
                cwd=str(self.green.path),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=build_env(self.java_home),
            )
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(
                exc.cmd, returncode=124,
                stdout=(exc.stdout or b"").decode("utf-8", "replace")
                if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=f"maven timed out after {timeout_s}s",
            )
        except FileNotFoundError:
            return subprocess.CompletedProcess(
                ["mvn"], returncode=127,
                stdout="", stderr="mvn is not on PATH",
            )

    def _record(self, tool: str, args: dict[str, Any], result: str, ok: bool) -> None:
        self.log.append(
            {
                "iteration": self.iterations,
                "tool": tool,
                "args": args,
                "ok": ok,
                "result": result,
            }
        )


# ──────────────────────────────  tool schemas  ───────────────────────────

APPLY_PATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "apply_patch",
        "description": (
            "Replace an exact snippet in one source file. old_text must match the "
            "file byte for byte and must appear exactly once. Only files under "
            "src/main/java may be written; test files are immutable and will be "
            "refused. Refusals explain what to do next — read them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": (
                        "Path relative to the repository root, e.g. "
                        "src/main/java/com/ex/service/CustomerService.java"
                    ),
                },
                "old_text": {
                    "type": "string",
                    "description": (
                        "The exact existing text to replace, copied from the source "
                        "shown in your instructions. Include enough surrounding "
                        "lines that it is unique in the file."
                    ),
                },
                "new_text": {
                    "type": "string",
                    "description": "The replacement text.",
                },
            },
            "required": ["file", "old_text", "new_text"],
        },
    },
}

BUILD_TOOL = {
    "type": "function",
    "function": {
        "name": "build",
        "description": (
            "Compile the project (mvn -q -DskipTests package). Takes no arguments. "
            "On failure you get the compiler output verbatim — read it literally, "
            "it names the file and line."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

RUN_TESTS_TOOL = {
    "type": "function",
    "function": {
        "name": "run_tests",
        "description": (
            "Run the existing test suite (mvn -q test). Takes no arguments. "
            "Requires a successful build first. The test count may not decrease; "
            "if it does, the run is over."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}
