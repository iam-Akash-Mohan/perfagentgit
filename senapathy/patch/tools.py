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
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..trace import EventType, Trace
from .worktree import GreenWorktree, build_env

# These three used to be module constants naming one repo's Maven layout.
# They are now FALLBACKS, used only when no RepoProfile has been supplied --
# which `PatchTools` records as a guardrail note rather than passing over in
# silence, because an allowlist that does not match the repository under test
# is an allowlist protecting the wrong tree.
#
# The real values come from `profile.write_allowlist`, `profile.layout
# .test_root` and `profile.migrations.directory`, all DETECTED by the onboard
# phase. CLAUDE.md invariant 1 asks for an allowlist, not a denylist, because
# a denylist protects the paths you thought of.
WRITE_ALLOWLIST = ("src/main/java",)
TEST_TREE = "src/test"
MIGRATION_DIR = "src/main/resources/db/migration"

MAX_BUILD_LINES = 40
# Larger than MAX_BUILD_LINES: a refused apply_patch is a source FILE, not a
# compiler log, and most files in this repo fit comfortably. A dedicated
# read_file tool exists for the rest (400 lines, continuable) -- this is
# just enough to unblock the common single-file retry without a second call.
MAX_REFUSAL_FILE_LINES = 200
MAX_ITERATIONS = 8
BUILD_TIMEOUT_S = 600
TEST_TIMEOUT_S = 900

_SUREFIRE = re.compile(
    r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)"
)


def _numbered(text: str, *, limit: int, label: str) -> str:
    """`label` (lines 1-N of TOTAL)` plus the content, numbered -- the same
    shape read_file returns, so a refusal and an explicit read give the
    model identical, citable output."""
    lines = text.splitlines()
    shown = lines[:limit]
    numbered = "\n".join(f"{i + 1:5d}| {line}" for i, line in enumerate(shown))
    header = f"{label} (lines 1-{len(shown)} of {len(lines)})"
    if len(shown) < len(lines):
        return (f"{header}\n{numbered}\n\n... truncated at line {len(shown)} "
                f"of {len(lines)}. Call read_file(path={label!r}, "
                f"start_line={len(shown) + 1}) to continue.")
    return f"{header}\n{numbered}"


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
    """Surefire's counts. Arithmetic in Python, per invariant 5.

    Surefire prints the same numbers twice: once per test class, with a
    `Time elapsed` on the line, and once as a summary without it. Summing
    every match double-counts — which was invisible while `mvn test` ran zero
    tests in this repo, and became a false "tests executed dropped from 2 to
    1" guardrail violation the moment it ran one. The summary wins when there
    is one; otherwise the per-class lines are summed.
    """
    counts = TestCounts(files=count_test_files(green))
    per_class: list[tuple[str, ...]] = []
    summary: list[tuple[str, ...]] = []
    for line in output.splitlines():
        match = _SUREFIRE.search(line)
        if match is None:
            continue
        (per_class if "Time elapsed" in line or " -- in " in line
         else summary).append(match.groups())

    for run, failures, errors, skipped in (summary or per_class):
        counts.executed += int(run)
        counts.failures += int(failures)
        counts.errors += int(errors)
        counts.skipped += int(skipped)
    return counts


_MAX_FRAMES_PER_FAILURE = 5
_MAX_MESSAGE_CHARS = 300


def _surefire_failures(green: Path, *, limit: int = MAX_BUILD_LINES) -> str | None:
    """Every failing/erroring test's class, method, and assertion message,
    read from surefire's XML reports -- not scraped from stdout.

    Test failures print at the END of `mvn test`'s output; a head-of-output
    truncation (right, for a compile error) shows the agent 40 lines of
    "Scanning for projects..." and nothing about what actually failed. The
    XML under target/surefire-reports/ already has this structured, one
    <failure>/<error> element per test case, so there is nothing to scrape.

    Returns None when there is nothing to read (no reports directory, or
    every report parsed but named no failure) so the caller falls back to
    the raw head-of-output behaviour -- the one case that still needs it is
    a JVM-level crash before any test produced a report at all.
    """
    reports_dir = green / "target" / "surefire-reports"
    if not reports_dir.exists():
        return None

    src = green / "src"
    project_files = {p.name for p in src.rglob("*.java")} if src.exists() else set()

    failures: list[tuple[str, str, str, str, list[str]]] = []
    for report in sorted(reports_dir.glob("TEST-*.xml")):
        try:
            root = ET.parse(report).getroot()
        except ET.ParseError:
            continue
        for case in root.findall("testcase"):
            classname = case.get("classname", "?")
            name = case.get("name", "?")
            node = case.find("failure")
            kind = "failure"
            if node is None:
                node = case.find("error")
                kind = "error"
            if node is None:
                continue

            message = (node.get("message") or "").strip().replace("\n", " ")
            if len(message) > _MAX_MESSAGE_CHARS:
                message = message[:_MAX_MESSAGE_CHARS] + " […]"

            frames: list[str] = []
            for line in (node.text or "").splitlines():
                stripped = line.strip()
                if not stripped.startswith("at "):
                    continue
                paren = stripped.rfind("(")
                source_file = (stripped[paren + 1:-1].split(":")[0]
                              if paren != -1 and stripped.endswith(")") else "")
                if source_file in project_files:
                    frames.append(stripped)

            failures.append((kind, classname, name, message, frames))

    if not failures:
        return None

    out = [f"{len(failures)} failing test(s), read from "
          f"target/surefire-reports/:", ""]
    shown = 0
    for kind, classname, name, message, frames in failures:
        block = [f"{classname}.{name} — {kind}"]
        if message:
            block.append(f"  {message}")
        block.extend(f"    {frame}" for frame in frames[:_MAX_FRAMES_PER_FAILURE])
        if len(out) + len(block) + 1 > limit and shown:
            out.append(f"[... {len(failures) - shown} more failing test(s) not "
                       f"shown; full reports under target/surefire-reports/ ...]")
            break
        out.extend(block)
        out.append("")
        shown += 1
    return "\n".join(out).rstrip()


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

    # The plan's `files_to_touch`. Empty means "no plan named files", and the
    # prefix rule above applies instead — which is what the pre-F3 tests use.
    allowed_files: tuple[str, ...] = ()
    # Tables the endpoint actually reads; an index migration may touch no other.
    allowed_tables: tuple[str, ...] = ()
    # Every file on THIS endpoint's own traced call path (comprehension's
    # call_chain) — the universe request_file_access may grant from. A plan
    # that scoped correctly never needs this; one that didn't gets a bounded
    # escape hatch instead of an abandonment (COL-06).
    call_path_files: tuple[str, ...] = ()
    # Files request_file_access has granted this run, on top of allowed_files.
    # Never includes a file outside call_path_files.
    _granted_files: set[str] = field(default_factory=set)
    # The detected repository profile. When present its layout wins over the
    # module fallbacks above, so the guardrail defends the tree actually
    # under test.
    profile: Any = None
    # Returns a rendered delta, or a note saying why there is none.
    benchmark: Any = None
    # Phase="patch" on every event -- without this handle, a patch run's
    # build/test iterations are invisible in trace.jsonl, which is exactly
    # the gap that left two ABANDONED runs with no record of what happened.
    trace: Trace | None = None

    iterations: int = 0
    log: list[dict[str, Any]] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    guardrail_violations: list[str] = field(default_factory=list)

    edits_since_verify: int = 0
    last_benchmark: str = ""
    benchmark_regressed: bool = False
    migrations_written: list[str] = field(default_factory=list)
    standards_violations: list[str] = field(default_factory=list)
    last_build_ok: bool = False
    last_tests_ok: bool = False
    last_error: str = ""

    # A monotonic count of accepted writes (apply_patch, add_migration),
    # NEVER reset -- unlike edits_since_verify, which a successful run_tests
    # zeroes out. build()/run_tests() compare this against its value at
    # their own last call to tell "nothing changed" from "something did",
    # which is the only question that decides whether a repeat is free.
    _edits_total: int = 0
    _last_build_edits: int | None = None
    _last_build_result: str = ""
    _last_test_edits: int | None = None
    _last_test_result: str = ""

    # ───────────────────────  layout, from the profile  ──────────────────

    def write_allowlist(self) -> tuple[str, ...]:
        """Prefixes a write may land under, from the profile when there is one."""
        if self.profile is not None:
            return tuple(self.profile.write_allowlist)
        return WRITE_ALLOWLIST

    def test_tree(self) -> str:
        """The write-DENIED test root (invariant 3)."""
        if self.profile is not None:
            return self.profile.layout.test_root
        return TEST_TREE

    def migration_dir(self) -> str | None:
        """Where `add_migration` may create a file, or None if the repo has
        no migration tool -- in which case a schema change cannot ship and
        saying so is better than writing a file nothing will apply."""
        if self.profile is not None:
            return self.profile.migrations.directory
        return MIGRATION_DIR

    def _allowlist_hint(self) -> str:
        return " or ".join(self.write_allowlist())

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
                f"REFUSED: cannot read {file}: {exc}. Check the path — it "
                f"must be relative to the repository root and under "
                f"{self._allowlist_hint()}."
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
                f"WHAT TO DO NEXT: your old_text must match the file byte for "
                f"byte, including indentation and line breaks. Copy it from "
                f"the CURRENT content below — not from what you were shown "
                f"at the start of this task, which this file may no longer "
                f"match after an earlier edit.\n\n"
                f"{_numbered(source, limit=MAX_REFUSAL_FILE_LINES, label=file)}"
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
        self._edits_total += 1
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
                f"REFUSED: {file} is an absolute path. Pass a path relative "
                f"to the repository root, under {self._allowlist_hint()}."
            )

        resolved = (self.green.path / candidate).resolve()
        try:
            relative = resolved.relative_to(self.green.path)
        except ValueError:
            return (
                f"REFUSED: {file} resolves outside the worktree you may write to. "
                f"Nothing was written. Only files inside the repository are "
                f"writable, and only under {self._allowlist_hint()}/."
            )

        posix = relative.as_posix()
        migration_dir = self.migration_dir()
        if migration_dir and posix.startswith(migration_dir):
            return (
                f"REFUSED: {posix} is a database migration. apply_patch cannot "
                f"write one — call add_migration instead, which parses the SQL "
                f"and refuses anything that is not a single CREATE INDEX, then "
                f"numbers the file for you."
            )
        if posix.startswith(self.test_tree()):
            # Invariant 3, in code. A prompt asking nicely is not a guardrail.
            return (
                f"REFUSED: {posix} is a test file. Test files are immutable — they "
                f"are the evidence that behaviour did not change, so a patch may "
                f"never edit them.\n"
                f"WHAT TO DO NEXT: make the production code satisfy the existing "
                f"test. Edit only files under {self._allowlist_hint()}/."
            )
        if self.allowed_files:
            # The PLAN decided which files this fix touches, before any code
            # was written. A patch that wanders outside it is not the patch
            # that was reviewed, and DIFF_MINIMALITY would fail it later
            # anyway — failing here costs one tool call instead of a run.
            # request_file_access can widen this at runtime, but only to a
            # file already on this endpoint's own call path — never to one
            # the model simply wanted.
            if posix not in self.allowed_files and posix not in self._granted_files:
                return (
                    f"REFUSED: {posix} is not in this plan's files_to_touch. "
                    f"Nothing was written.\n"
                    f"You may edit exactly these files: "
                    f"{', '.join(self.allowed_files)}.\n"
                    f"WHAT TO DO NEXT: if this fix genuinely cannot be "
                    f"written without {posix}, call "
                    f"request_file_access('{posix}') — it is granted "
                    f"automatically when the file is on this endpoint's own "
                    f"traced call path. If it is refused, this is the wrong "
                    f"candidate; say so rather than editing around the plan."
                )
            return None

        if not posix.startswith(self.write_allowlist()):
            return (
                f"REFUSED: {posix} is outside the write allowlist. Nothing was "
                f"written.\n"
                f"WHAT TO DO NEXT: this task may only change application source. "
                f"Edit files under {self._allowlist_hint()}/ only — not the "
                f"build file, not configuration, not resources."
            )
        return None

    # ────────────────────────  request_file_access  ───────────────────────

    def request_file_access(self, file: str) -> str:
        """Widen the write allowlist by exactly one file, bounded.

        A plan that scoped itself correctly never calls this. One that
        didn't (COL-06: a query_rewrite that changed a repository method's
        signature, with the calling service left out of files_to_touch) has
        one way out: name a file already on THIS endpoint's own traced call
        path. Granted automatically when it is; refused by name, with the
        same hard limits apply_patch enforces, when it is not. Every
        request is recorded here either way, so the report shows the scope
        actually used against the scope the plan declared.
        """
        candidate = Path(file)
        if candidate.is_absolute():
            message = (
                f"REFUSED: {file} is an absolute path. Pass a path relative "
                f"to the repository root. Nothing was granted."
            )
            self.refusals.append(message)
            self._record("request_file_access", {"file": file}, message, ok=False)
            return message

        resolved = (self.green.path / candidate).resolve()
        try:
            relative = resolved.relative_to(self.green.path)
        except ValueError:
            message = (
                f"REFUSED: {file} resolves outside the repository. Nothing "
                f"was granted."
            )
            self.refusals.append(message)
            self._record("request_file_access", {"file": file}, message, ok=False)
            return message

        posix = relative.as_posix()

        # These never move, regardless of the call path: never a test file,
        # never a schema file outside the migration tool's own directory.
        if posix.startswith(self.test_tree()):
            message = (
                f"REFUSED: {posix} is a test file. Test files are immutable "
                f"— this can never be granted, on the call path or off it."
            )
            self.refusals.append(message)
            self._record("request_file_access", {"file": file}, message, ok=False)
            return message

        migration_dir = self.migration_dir()
        if posix.endswith(".sql") and not (migration_dir and posix.startswith(migration_dir)):
            where = migration_dir or "the migration directory"
            message = (
                f"REFUSED: {posix} is a schema file outside {where}. "
                f"A schema change may only land there, via add_migration."
            )
            self.refusals.append(message)
            self._record("request_file_access", {"file": file}, message, ok=False)
            return message

        if not posix.startswith(self.write_allowlist()):
            message = (
                f"REFUSED: {posix} is outside the detected source root "
                f"({self._allowlist_hint()}). Nothing was granted, "
                f"regardless of the call path."
            )
            self.refusals.append(message)
            self._record("request_file_access", {"file": file}, message, ok=False)
            return message

        if posix not in self.call_path_files:
            message = (
                f"REFUSED: {posix} is not on this endpoint's own traced call "
                f"path. Nothing was granted.\n"
                f"WHAT TO DO NEXT: this is only ever granted for a file this "
                f"endpoint's own request actually passes through. If the fix "
                f"needs a file genuinely unrelated to this endpoint, this is "
                f"the wrong candidate — call abandon_patch and say so."
            )
            self.refusals.append(message)
            self._record("request_file_access", {"file": file}, message, ok=False)
            return message

        self._granted_files.add(posix)
        message = (
            f"GRANTED: {posix} is on this endpoint's own traced call path. "
            f"You may now edit it with apply_patch, in addition to the "
            f"plan's own files_to_touch."
        )
        self._record("request_file_access", {"file": file}, message, ok=True)
        return message

    # ───────────────────────────────  build  ─────────────────────────────

    def build(self) -> str:
        """`mvn -q -DskipTests package` in the green worktree.

        A repeat call with no `apply_patch`/`add_migration` since the last
        one is answered from cache, free: the source has not changed, so
        the compiler cannot have a different opinion, and the one real
        resource here (the iteration budget) should not be spent finding
        that out twice.
        """
        if (self._last_build_edits is not None
                and self._last_build_edits == self._edits_total):
            result = (
                "SAME AS LAST TIME — no file has changed since the previous "
                "build call, so this did not spend an iteration. Edit "
                "something first if you want a different answer.\n\n"
                + self._last_build_result
            )
            self._record("build", {"repeated": True}, result, ok=self.last_build_ok)
            return result

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
        self._last_build_edits = self._edits_total
        self._last_build_result = result
        return result

    # ─────────────────────────────  run_tests  ───────────────────────────

    def run_tests(self) -> str:
        """`mvn test` in the green worktree, with the count guardrail.

        Deliberately NOT `-q`: quiet mode suppresses surefire's own
        "Tests run: N, Failures: ..." summary lines, which is exactly what
        `parse_test_counts` greps for. With `-q` this guardrail was inert
        for every repo, always reporting zero tests executed regardless of
        how many actually ran -- not because surefire skipped them, but
        because nothing printed the number that would have said otherwise.
        """
        if not self.last_build_ok:
            message = (
                "REFUSED: the last build did not succeed, so there is nothing to "
                "test. Call build first and fix any compiler errors."
            )
            self._record("run_tests", {}, message, ok=False)
            return message

        # Same rationale as build(): no edit since the last run_tests call
        # means the suite cannot have a different verdict, so a repeat is
        # answered from cache instead of spending an iteration to learn
        # nothing -- this is the exact pattern COL-03 burned two calls on.
        if (self._last_test_edits is not None
                and self._last_test_edits == self._edits_total):
            result = (
                "SAME AS LAST TIME — no file has changed since the previous "
                "run_tests call, so this did not spend an iteration. Edit "
                "something first if you want a different answer.\n\n"
                + self._last_test_result
            )
            self._record("run_tests", {"repeated": True}, result,
                        ok=self.last_tests_ok)
            return result

        budget = self._spend_iteration("run_tests")
        if budget is not None:
            return budget

        completed = self._maven(["test"], TEST_TIMEOUT_S)
        output = (completed.stdout or "") + (completed.stderr or "")
        counts = parse_test_counts(output, self.green.path)

        # The guardrail: computed here, and it fails the phase. Not a warning,
        # and not something the model is asked to check.
        violation = self._check_test_counts(counts)
        if violation:
            self.guardrail_violations.append(violation)
            self.last_tests_ok = False
            self._record("run_tests", {}, violation, ok=False)
            self._last_test_edits = self._edits_total
            self._last_test_result = violation
            return violation

        failed = counts.failures + counts.errors
        if completed.returncode == 0 and failed == 0:
            self.last_tests_ok = True
            self.edits_since_verify = 0
            self.last_error = ""
            result = (
                f"TESTS OK — {counts.render()}. "
                f"Baseline before your patch: {self.baseline.render()}."
                + self._measure_delta()
                + "\nYou may now call emit_patch."
            )
        else:
            self.last_tests_ok = False
            # Read from surefire's own XML first -- the failure detail that
            # matters is at the END of stdout, past where a head-of-output
            # truncation would ever reach. Falls back to the raw head only
            # when there is nothing structured to read (a JVM-level crash
            # before any test produced a report).
            detail = _surefire_failures(self.green.path, limit=MAX_BUILD_LINES)
            if detail is None:
                detail = _first_lines(output)
                label = f"Output, first {MAX_BUILD_LINES} lines, verbatim"
            else:
                label = "Failing tests"
            self.last_error = detail
            result = (
                f"TESTS FAILED (exit {completed.returncode}) — {counts.render()}.\n"
                f"{label}:\n\n{detail}"
            )

        self._record("run_tests", {"counts": counts.render()}, result, ok=self.last_tests_ok)
        self._last_test_edits = self._edits_total
        self._last_test_result = result
        return result

    def _measure_delta(self) -> str:
        """The agent's own before/after, measured here and handed over.

        Runs only after build AND tests have passed, which is the first moment
        the tree is worth booting. A patch that is three times slower should be
        reconsidered by the agent that wrote it, not discovered two phases
        later by a gate.
        """
        if self.benchmark is None:
            return ""
        try:
            outcome = self.benchmark()
        except Exception as exc:  # noqa: BLE001 — a heuristic never fails a phase
            return f"\n(the in-loop benchmark could not run: {exc})"
        rendered = outcome if isinstance(outcome, str) else outcome.render()
        self.last_benchmark = rendered
        if not isinstance(outcome, str) and outcome.regressed:
            self.benchmark_regressed = True
        self._record("benchmark", {}, rendered, ok=True)
        return "\n" + rendered

    # ────────────────────────────  add_migration  ────────────────────────

    def add_migration(self, sql: str, rationale: str,
                      evidence: str = "") -> str:
        """Write an additive index migration — after parsing the SQL.

        The model supplies SQL and this refuses anything that is not exactly
        one CREATE INDEX. Not because a model would knowingly send `DROP
        TABLE`, but because "additive index only" has to be a property of the
        code rather than of the instructions: the parse decides, and the file
        that lands is then GENERATED from the parsed table and columns rather
        than from the string that was sent.
        """
        from ..plan.migration import write_index_migration
        from ..plan.risk import is_additive_index, parse_index

        if not is_additive_index(sql):
            message = (
                f"REFUSED: that is not a single additive CREATE INDEX "
                f"statement, and an index is the only schema change this "
                f"phase may make. Nothing was written.\n"
                f"Received: {str(sql).strip()[:200]!r}\n"
                f"WHAT TO DO NEXT: send exactly one statement of the form "
                f"CREATE INDEX idx_name ON table (col_a, col_b);"
            )
            self.refusals.append(message)
            self._record("add_migration", {"sql": sql[:120]}, message, ok=False)
            return message

        parsed = parse_index(sql)
        if parsed is None:                        # unreachable after the check
            message = ("REFUSED: the table and columns could not be read out "
                       "of that statement.")
            self._record("add_migration", {"sql": sql[:120]}, message, ok=False)
            return message
        _name, table, columns = parsed

        if self.allowed_tables and table not in self.allowed_tables:
            message = (
                f"REFUSED: {table} is not a table this endpoint reads. "
                f"Nothing was written. Tables on the path: "
                f"{', '.join(self.allowed_tables)}."
            )
            self.refusals.append(message)
            self._record("add_migration", {"table": table}, message, ok=False)
            return message

        try:
            path = write_index_migration(
                self.green.path, table, columns, rationale or "index added by "
                "Senapathy", evidence=evidence or None)
        except Exception as exc:  # noqa: BLE001 — a refusal the agent can read
            message = f"REFUSED: the migration could not be written: {exc}"
            self.refusals.append(message)
            self._record("add_migration", {"table": table}, message, ok=False)
            return message

        relative = path.relative_to(self.green.path).as_posix()
        self.migrations_written.append(relative)
        self.edits_since_verify += 1
        self._edits_total += 1
        self.last_build_ok = False
        self.last_tests_ok = False

        result = (
            f"OK: wrote {relative}\n{path.read_text(encoding='utf-8').strip()}\n"
            f"The file was GENERATED from the table and columns parsed out of "
            f"your statement, so what landed is what was checked. You must now "
            f"call build, then run_tests."
        )
        self._record("add_migration", {"file": relative}, result, ok=True)
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
        # A note per build/test iteration, distinct from the generic
        # tool_call/tool_result pair run_agent already emits: those carry a
        # clipped preview of the raw output, this carries the one-line
        # verdict a reader actually scans for.
        if self.trace is not None and tool in ("build", "run_tests"):
            self.trace.emit(
                EventType.NOTE, phase="patch",
                note=f"{tool} iteration {self.iterations}/{self.max_iterations}: "
                     f"{'OK' if ok else 'FAILED'} — {result.splitlines()[0][:200]}",
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
                        "Path relative to the repository root, inside the "
                        "write allowlist you were given."
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
            "Run the existing test suite (mvn test). Takes no arguments. "
            "Requires a successful build first. The test count may not decrease; "
            "if it does, the run is over. On a failure you get every failing "
            "test's class, method and assertion message, read from the "
            "XML reports. A repeat call with no edit since the last one "
            "costs nothing — the same answer is returned without spending "
            "an iteration."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


REQUEST_FILE_ACCESS_TOOL = {
    "type": "function",
    "function": {
        "name": "request_file_access",
        "description": (
            "Ask to write ONE file beyond the plan's files_to_touch. Granted "
            "automatically if, and only if, the file is already on this "
            "endpoint's own traced call path; refused by name otherwise. "
            "Never a test file, never a schema file outside the migration "
            "directory. Use this instead of abandoning when the fix needs a "
            "caller or callee the plan left out — not to reach for a file "
            "unrelated to this endpoint."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Path relative to the repository root.",
                },
            },
            "required": ["file"],
        },
    },
}

ADD_MIGRATION_TOOL = {
    "type": "function",
    "function": {
        "name": "add_migration",
        "description": (
            "Add a database index as a versioned migration. Send ONE "
            "`CREATE INDEX` statement; it is parsed before anything is "
            "written, and anything else — ALTER, DROP, a second statement — is "
            "refused. The file is numbered and named for you. This is the only "
            "schema change this phase can make, and the only way a new file "
            "can be created."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "Exactly one statement: "
                                   "CREATE INDEX idx_name ON table (col_a, col_b);",
                },
                "rationale": {
                    "type": "string",
                    "description": "Why this index, in one sentence. It becomes "
                                   "a comment in the migration, which is what a "
                                   "reviewer reads first.",
                },
                "evidence": {
                    "type": "string",
                    "description": "The measurement behind it, e.g. the "
                                   "index_benefit output: 'rows examined "
                                   "2,645 -> 571'. Quoted verbatim into the "
                                   "file.",
                },
            },
            "required": ["sql", "rationale"],
        },
    },
}
