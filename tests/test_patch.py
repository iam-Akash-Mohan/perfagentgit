"""P12 tests: the worktree, the write guardrails, and the count guardrail.

No network and no LLM. Maven is never invoked — `_maven` is replaced with a
stub, because the point of these tests is the guardrail logic, and a test that
needs a JDK and a 90-second build is a test nobody runs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from perfagent.errors import ContractViolation
from perfagent.patch import tools as patch_tools
from perfagent.patch.tools import (MAX_BUILD_LINES, PatchTools, TestCounts,
                                   count_test_files, parse_test_counts)
from perfagent.patch.worktree import GreenWorktree

CONTROLLER = "src/main/java/com/ex/web/CustomerController.java"
SERVICE = "src/main/java/com/ex/service/CustomerService.java"
TEST_FILE = "src/test/java/com/ex/CustomerSearchIT.java"


# ──────────────────────────────  fixtures  ───────────────────────────────


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return result.stdout


@pytest.fixture
def blue(tmp_path: Path) -> Path:
    """A miniature target repo: committed source, plus uncommitted files that
    the build depends on — the shape the real repo is actually in."""
    repo = tmp_path / "blue"
    (repo / "src/main/java/com/ex/service").mkdir(parents=True)
    (repo / "src/main/java/com/ex/web").mkdir(parents=True)
    (repo / "src/test/java/com/ex").mkdir(parents=True)

    (repo / SERVICE).write_text(
        "class CustomerService {\n"
        "    void enrich() { outstanding(); }\n"
        "    void other() { outstanding(); }\n"
        "}\n"
    )
    (repo / CONTROLLER).write_text("class CustomerController {}\n")
    (repo / TEST_FILE).write_text("class CustomerSearchIT {}\n")
    (repo / "pom.xml").write_text("<project><artifactId>x</artifactId></project>\n")

    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")

    # Uncommitted, and the app needs them — exactly mini-spring's situation.
    (repo / "pom.xml").write_text(
        "<project><artifactId>x</artifactId><!-- actuator --></project>\n"
    )
    (repo / "src/main/resources").mkdir(parents=True)
    (repo / "src/main/resources/application-perfagent.yml").write_text("port: 18080\n")

    # Untracked and NOT a build input: mini-spring's seed/ tree is 936 files.
    (repo / "seed").mkdir()
    for i in range(5):
        (repo / f"seed/rows_{i}.sql").write_text(f"insert {i};\n")

    # A symlink pointing outside the repository — mini-spring really has one of
    # these under src/main/resources.
    outside = tmp_path / "OUTSIDE.txt"
    outside.write_text("secret from outside the repo\n")
    (repo / "src/main/resources/escape-link.txt").symlink_to(outside)
    return repo


@pytest.fixture
def green(blue: Path, tmp_path: Path) -> GreenWorktree:
    worktree = GreenWorktree.create(
        "test-run", repo=blue, green=tmp_path / "green"
    )
    yield worktree
    worktree.remove()


def make_tools(green: GreenWorktree, **kwargs) -> PatchTools:
    baseline = TestCounts(executed=0, files=count_test_files(green.path))
    return PatchTools(green=green, baseline=baseline, **kwargs)


def stub_maven(tools: PatchTools, returncode: int, output: str) -> None:
    """Replace the one place a subprocess is spawned."""
    tools._maven = lambda args, timeout_s: subprocess.CompletedProcess(
        ["mvn", *args], returncode=returncode, stdout=output, stderr=""
    )


# ────────────────────────────────  worktree  ──────────────────────────────


def test_green_is_a_worktree_on_a_new_branch(green: GreenWorktree, blue: Path) -> None:
    assert green.path.exists()
    assert green.branch == "perfagent/test-run"
    assert (green.path / SERVICE).exists()
    branches = _git(blue, "branch", "--list", "perfagent/test-run")
    assert "perfagent/test-run" in branches


def test_blue_working_tree_is_never_written(green: GreenWorktree, blue: Path) -> None:
    before = green.blue_status()
    (green.path / SERVICE).write_text("class CustomerService { /* patched */ }\n")
    green.checkpoint("edited")
    # Blue's own status is byte-identical: the edit landed only in green.
    assert green.blue_status() == before
    assert "patched" not in (blue / SERVICE).read_text()


def test_uncommitted_files_are_carried_into_green(green: GreenWorktree) -> None:
    """A worktree is checked out from a COMMIT, so without this green is
    missing the runtime profile and the pom additions the app needs."""
    assert "src/main/resources/application-perfagent.yml" in green.carried_over
    assert "pom.xml" in green.carried_over
    assert (green.path / "src/main/resources/application-perfagent.yml").exists()
    assert "actuator" in (green.path / "pom.xml").read_text()


def test_untracked_non_build_files_are_skipped_not_copied(green: GreenWorktree) -> None:
    """seed/ is 936 files in the real repo and maven compiles none of them."""
    assert not any(rel.startswith("seed/") for rel in green.carried_over)
    assert not (green.path / "seed").exists()
    # Skipped visibly, not silently.
    assert any("seed/rows_0.sql" in entry for entry in green.skipped)


def test_a_symlink_out_of_the_repo_is_never_carried(green: GreenWorktree) -> None:
    """shutil.copy2 follows symlinks, so carrying one would import content from
    outside the repository into the tree the agent is allowed to edit."""
    assert not any("escape-link" in rel for rel in green.carried_over)
    assert not (green.path / "src/main/resources/escape-link.txt").exists()
    assert any("symlink" in entry for entry in green.skipped)


def test_editing_a_path_that_escapes_via_symlink_is_refused(
    green: GreenWorktree, blue: Path, tmp_path: Path
) -> None:
    # The link exists in blue; even if it reached green it is outside the
    # allowlist AND resolves outside the worktree.
    tools = make_tools(green)
    result = tools.apply_patch(
        "src/main/resources/escape-link.txt", "secret", "overwritten"
    )
    assert result.startswith("REFUSED")
    assert (tmp_path / "OUTSIDE.txt").read_text() == "secret from outside the repo\n"


def test_create_refuses_an_existing_green_directory(blue: Path, tmp_path: Path) -> None:
    existing = tmp_path / "already"
    existing.mkdir()
    with pytest.raises(ContractViolation, match="already exists"):
        GreenWorktree.create("r2", repo=blue, green=existing)


def test_create_refuses_a_reused_run_id(green: GreenWorktree, blue: Path,
                                        tmp_path: Path) -> None:
    with pytest.raises(ContractViolation, match="already exists"):
        GreenWorktree.create("test-run", repo=blue, green=tmp_path / "green2")


def test_reset_to_checkpoint_discards_a_bad_iteration(green: GreenWorktree) -> None:
    original = (green.path / SERVICE).read_text()
    (green.path / SERVICE).write_text("this does not compile\n")
    (green.path / "src/main/java/com/ex/Junk.java").write_text("junk\n")

    green.reset_to_checkpoint()

    # A bad state is a reset, not a debugging session: tracked edits reverted
    # AND untracked leftovers removed.
    assert (green.path / SERVICE).read_text() == original
    assert not (green.path / "src/main/java/com/ex/Junk.java").exists()


def test_checkpoints_stack_and_reset_targets_the_first(green: GreenWorktree) -> None:
    (green.path / SERVICE).write_text("one\n")
    first = green.checkpoint("one")
    (green.path / SERVICE).write_text("two\n")
    green.checkpoint("two")

    green.reset_to_checkpoint(first)
    assert (green.path / SERVICE).read_text() == "one\n"


def test_source_diff_ignores_build_artifacts_dropped_into_the_tree(
    green: GreenWorktree,
) -> None:
    """A real bug: arm builds wrote a classpath file into the worktree, `diff()`
    staged it, and the patch identity changed — so `measure` refused a patch it
    had just verified."""
    (green.path / SERVICE).write_text("class CustomerService { /* patched */ }\n")
    before = green.source_diff()

    (green.path / "perfagent-classpath.txt").write_text("/some/jar:/another/jar\n")
    (green.path / "generated-report.txt").write_text("noise\n")

    assert green.source_diff() == before, "a build artifact changed the patch identity"
    # ...while the unrestricted diff does see it, which is why identity uses
    # source_diff and not diff.
    assert "perfagent-classpath.txt" in green.diff()


def test_diff_stat_is_computed_not_reported(green: GreenWorktree) -> None:
    (green.path / SERVICE).write_text("class CustomerService {\n}\n")
    files, added, removed = green.diff_stat()
    assert files == [SERVICE]
    # Two method lines removed, nothing added: exact numbers, not "more than 0".
    assert (added, removed) == (0, 2)
    assert SERVICE in green.diff()

    (green.path / CONTROLLER).write_text("class CustomerController {\n  int x;\n}\n")
    files, added, removed = green.diff_stat()
    assert sorted(files) == sorted([CONTROLLER, SERVICE])
    assert (added, removed) == (3, 3)


# ───────────────────────────  apply_patch refusals  ───────────────────────


def test_a_write_to_a_test_file_is_refused_and_logged(green: GreenWorktree) -> None:
    tools = make_tools(green)
    before = (green.path / TEST_FILE).read_text()

    result = tools.apply_patch(TEST_FILE, "class CustomerSearchIT {}", "class X {}")

    assert result.startswith("REFUSED")
    assert "immutable" in result
    assert "WHAT TO DO NEXT" in result
    assert (green.path / TEST_FILE).read_text() == before
    # Logged, not silently swallowed.
    assert tools.refusals and tools.refusals[-1] == result
    assert tools.log[-1]["tool"] == "apply_patch" and tools.log[-1]["ok"] is False


def test_a_write_outside_the_worktree_is_refused(green: GreenWorktree) -> None:
    tools = make_tools(green)
    result = tools.apply_patch("../../../etc/hosts", "a", "b")
    assert result.startswith("REFUSED")
    assert "outside the worktree" in result


def test_an_absolute_path_is_refused(green: GreenWorktree) -> None:
    tools = make_tools(green)
    result = tools.apply_patch("/etc/hosts", "a", "b")
    assert result.startswith("REFUSED")
    assert "absolute path" in result


def test_a_write_outside_src_main_java_is_refused(green: GreenWorktree) -> None:
    tools = make_tools(green)
    result = tools.apply_patch("pom.xml", "<project>", "<project><!-- x -->")
    assert result.startswith("REFUSED")
    assert "write allowlist" in result
    assert "actuator" in (green.path / "pom.xml").read_text()  # untouched


def test_old_text_not_found_is_refused_with_advice(green: GreenWorktree) -> None:
    tools = make_tools(green)
    result = tools.apply_patch(SERVICE, "does not exist anywhere", "x")
    assert result.startswith("REFUSED")
    assert "not found" in result
    assert "byte for byte" in result


def test_ambiguous_old_text_is_refused_with_the_count(green: GreenWorktree) -> None:
    tools = make_tools(green)
    result = tools.apply_patch(SERVICE, "outstanding();", "cheaper();")
    assert result.startswith("REFUSED")
    assert "appears 2 times" in result
    assert "exactly once" in result
    # Nothing written on an ambiguous match.
    assert "cheaper" not in (green.path / SERVICE).read_text()


def test_empty_old_text_is_refused(green: GreenWorktree) -> None:
    tools = make_tools(green)
    assert tools.apply_patch(SERVICE, "", "x").startswith("REFUSED")


def test_a_valid_edit_is_applied_and_reports_the_line(green: GreenWorktree) -> None:
    tools = make_tools(green)
    result = tools.apply_patch(
        SERVICE, "    void enrich() { outstanding(); }", "    void enrich() { fast(); }"
    )
    assert result.startswith("OK")
    assert "line 2" in result
    assert "fast();" in (green.path / SERVICE).read_text()
    # An edit invalidates any previous verification.
    assert tools.edits_since_verify == 1
    assert tools.last_build_ok is False and tools.last_tests_ok is False


def test_a_refusal_never_raises(green: GreenWorktree) -> None:
    """A refusal is the most useful context the agent gets, and it is only
    useful if it reaches the model rather than ending the run."""
    tools = make_tools(green)
    for args in (
        (TEST_FILE, "class CustomerSearchIT {}", "x"),
        ("/etc/hosts", "a", "b"),
        ("nope/missing.java", "a", "b"),
        (SERVICE, "not here", "x"),
    ):
        assert isinstance(tools.apply_patch(*args), str)


# ──────────────────────────────  build output  ────────────────────────────


def test_compiler_output_is_verbatim_and_truncated_by_line_count(
    green: GreenWorktree,
) -> None:
    tools = make_tools(green)
    error = "[ERROR] CustomerService.java:[12,9] cannot find symbol\n  symbol: fast()"
    stub_maven(tools, 1, error)

    result = tools.build()

    assert "BUILD FAILED" in result
    # Verbatim: the exact compiler text, not a paraphrase.
    assert error in result
    assert tools.last_build_ok is False


def test_long_compiler_output_is_cut_at_forty_lines(green: GreenWorktree) -> None:
    tools = make_tools(green)
    stub_maven(tools, 1, "\n".join(f"[ERROR] line {i}" for i in range(200)))

    result = tools.build()

    assert "[ERROR] line 39" in result
    assert "[ERROR] line 40" not in result
    assert f"{200 - MAX_BUILD_LINES} further lines truncated" in result


def test_a_successful_build_unlocks_tests(green: GreenWorktree) -> None:
    tools = make_tools(green)
    stub_maven(tools, 0, "")
    assert "BUILD OK" in tools.build()
    assert tools.last_build_ok is True


def test_run_tests_refuses_without_a_successful_build(green: GreenWorktree) -> None:
    tools = make_tools(green)
    result = tools.run_tests()
    assert result.startswith("REFUSED")
    assert "did not succeed" in result
    # A refused call costs no iteration.
    assert tools.iterations == 0


# ─────────────────────────────  test counting  ────────────────────────────


def test_surefire_counts_are_summed_in_python() -> None:
    output = (
        "Tests run: 3, Failures: 0, Errors: 0, Skipped: 0\n"
        "Tests run: 5, Failures: 1, Errors: 2, Skipped: 1\n"
    )
    counts = parse_test_counts(output, Path("/nonexistent"))
    assert counts.executed == 8
    assert counts.failures == 1 and counts.errors == 2 and counts.skipped == 1
    assert counts.passed == 5


def test_zero_executed_tests_parses_as_zero_not_as_a_failure() -> None:
    """mvn test on this repo runs NO tests: surefire's default includes are
    *Test / Test* / *Tests / *TestCase, and CustomerSearchIT matches none."""
    counts = parse_test_counts("[INFO] --- surefire:3.1.2:test ---\n", Path("/nope"))
    assert counts.executed == 0 and counts.failures == 0


def test_a_dropped_test_file_is_a_guardrail_violation(green: GreenWorktree) -> None:
    tools = make_tools(green)
    stub_maven(tools, 0, "")
    tools.build()

    (green.path / TEST_FILE).unlink()
    result = tools.run_tests()

    assert "GUARDRAIL VIOLATION" in result
    assert "dropped from 1 to 0" in result
    assert tools.guardrail_violations
    assert tools.last_tests_ok is False


def test_a_dropped_executed_count_is_a_guardrail_violation(
    green: GreenWorktree,
) -> None:
    tools = PatchTools(
        green=green,
        baseline=TestCounts(executed=4, files=count_test_files(green.path)),
    )
    stub_maven(tools, 0, "Tests run: 2, Failures: 0, Errors: 0, Skipped: 0")
    tools.build()

    result = tools.run_tests()

    assert "GUARDRAIL VIOLATION" in result
    assert "dropped from 4 to 2" in result


def test_passing_tests_clear_the_edit_flag(green: GreenWorktree) -> None:
    tools = make_tools(green)
    tools.apply_patch(SERVICE, "    void other() { outstanding(); }", "    void other() {}")
    stub_maven(tools, 0, "Tests run: 0, Failures: 0, Errors: 0, Skipped: 0")
    tools.build()
    assert "TESTS OK" in tools.run_tests()
    assert tools.edits_since_verify == 0
    assert tools.last_tests_ok is True


def test_failing_tests_return_the_output_verbatim(green: GreenWorktree) -> None:
    tools = make_tools(green)
    stub_maven(tools, 0, "")
    tools.build()
    # Build succeeded; only the test run fails.
    stub_maven(tools, 1, "Tests run: 1, Failures: 1, Errors: 0, Skipped: 0\n"
                         "expected:<200> but was:<500>")
    result = tools.run_tests()
    assert "TESTS FAILED" in result
    assert "expected:<200> but was:<500>" in result
    assert tools.last_tests_ok is False


# ──────────────────────────────  loop control  ────────────────────────────


def test_the_iteration_budget_is_shared_and_enforced(green: GreenWorktree) -> None:
    tools = make_tools(green, max_iterations=3)
    stub_maven(tools, 0, "")
    for _ in range(3):
        tools.build()
    assert tools.iterations == 3

    result = tools.build()
    assert result.startswith("REFUSED")
    assert "budget of 3 iterations is exhausted" in result
    # Budget is not silently exceeded.
    assert tools.iterations == 3


def test_maven_absence_is_reported_not_raised(green: GreenWorktree) -> None:
    tools = make_tools(green)
    tools._maven = PatchTools._maven.__get__(tools)  # real implementation
    original_run = subprocess.run
    try:
        subprocess.run = lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError())
        result = tools.build()
    finally:
        subprocess.run = original_run
    assert "BUILD FAILED" in result and "not on PATH" in result


# ──────────────────────────────  invariants  ──────────────────────────────


def test_build_and_run_tests_take_no_model_supplied_arguments() -> None:
    """Invariant 1: no string the model provides becomes part of a command."""
    for tool in (patch_tools.BUILD_TOOL, patch_tools.RUN_TESTS_TOOL):
        assert tool["function"]["parameters"]["properties"] == {}


def test_the_patch_agent_cannot_reach_the_oracle() -> None:
    """Invariant 2: the grader is not in the agent's tool surface."""
    root = Path(__file__).resolve().parents[1] / "perfagent" / "patch"
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "perfagent.oracle" not in source and "from ..oracle" not in source
        assert "oracle.json" not in source
