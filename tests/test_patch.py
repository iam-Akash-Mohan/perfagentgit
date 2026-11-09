"""P12 tests: the worktree, the write guardrails, and the count guardrail.

No network and no LLM. Maven is never invoked — `_maven` is replaced with a
stub, because the point of these tests is the guardrail logic, and a test that
needs a JDK and a 90-second build is a test nobody runs.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from senapathy.errors import ContractViolation
from senapathy.patch import tools as patch_tools
from senapathy.patch.tools import (MAX_BUILD_LINES, PatchTools, TestCounts,
                                   count_test_files, parse_test_counts)
from senapathy.patch.worktree import GreenWorktree

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
    assert green.branch == "senapathy/test-run"
    assert (green.path / SERVICE).exists()
    branches = _git(blue, "branch", "--list", "senapathy/test-run")
    assert "senapathy/test-run" in branches


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


def test_create_heals_a_leftover_directory_at_the_path(blue: Path, tmp_path: Path) -> None:
    """Self-healing, not fatal: a plain directory left at the target path
    (not even a git worktree — debris from something else entirely) must
    not be the reason `create()` refuses. `add` is never the first thing
    tried against a path that may be dirty."""
    existing = tmp_path / "already"
    existing.mkdir()
    (existing / "junk.txt").write_text("leftover\n")

    worktree = GreenWorktree.create("r2", repo=blue, green=existing)

    assert worktree.path == existing.resolve()
    assert (existing / "pom.xml").exists()
    assert not (existing / "junk.txt").exists()


def test_create_refuses_a_run_id_that_already_produced_arms(
    green: GreenWorktree, blue: Path, tmp_path: Path
) -> None:
    with pytest.raises(ContractViolation, match="already produced arms"):
        GreenWorktree.create("test-run", repo=blue, green=tmp_path / "green2")


def test_a_stale_registration_with_a_missing_directory_is_healed(
    blue: Path, tmp_path: Path
) -> None:
    """git's own bookkeeping still believes a worktree is at this path --
    its directory was removed directly, not via `git worktree remove` --
    and that must be healed, not the reason the next run fails."""
    path = tmp_path / "green"
    first = GreenWorktree.create("first", repo=blue, green=path)
    shutil.rmtree(first.path)  # bypasses `git worktree remove` on purpose

    second = GreenWorktree.create("second", repo=blue, green=path)

    assert second.path == path.resolve()
    assert (path / "pom.xml").exists()


def test_a_failed_worktree_add_leaves_no_branch_behind(
    blue: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`git worktree add -b` is two things, not one: the branch, then the
    checkout. If the real git call fails after creating the branch, the
    branch must not survive to poison the next attempt at this run id."""
    import senapathy.patch.worktree as worktree_mod

    real_git = worktree_mod._git

    def flaky_git(repo, *args, check=True):
        if args[:2] == ("worktree", "add"):
            branch = args[3]
            base = args[5]
            # What a real partial failure leaves behind: the branch ref
            # exists, the worktree does not.
            real_git(repo, "branch", branch, base)
            raise ContractViolation("simulated: worktree add failed after "
                                    "the branch was created")
        return real_git(repo, *args, check=check)

    monkeypatch.setattr(worktree_mod, "_git", flaky_git)

    with pytest.raises(ContractViolation, match="simulated"):
        GreenWorktree.create("flaky", repo=blue, green=tmp_path / "green")

    remaining = real_git(blue, "branch", "--list", "senapathy/flaky").strip()
    assert remaining == ""


def test_blue_on_a_stale_commit_is_recreated(blue: Path, tmp_path: Path) -> None:
    from senapathy.measure.arms import prepare_blue_worktree

    green_path = tmp_path / "blue-arm"
    first = prepare_blue_worktree("r1", repo=blue, blue=green_path)
    old_base = first.base_commit

    # The target repo moves on -- a new commit lands after blue was built.
    (blue / "src/main/java/com/ex/service/NewFeature.java").parent.mkdir(
        parents=True, exist_ok=True)
    (blue / "src/main/java/com/ex/service/NewFeature.java").write_text(
        "package com.ex.service;\nclass NewFeature {}\n")
    subprocess.run(["git", "-C", str(blue), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(blue), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "a new commit blue has not seen"],
        check=True,
    )

    second = prepare_blue_worktree("r2", repo=blue, blue=green_path)

    assert second.base_commit != old_base
    assert (second.path / "src/main/java/com/ex/service/NewFeature.java").exists()


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

    (green.path / "senapathy-classpath.txt").write_text("/some/jar:/another/jar\n")
    (green.path / "generated-report.txt").write_text("noise\n")

    assert green.source_diff() == before, "a build artifact changed the patch identity"
    # ...while the unrestricted diff does see it, which is why identity uses
    # source_diff and not diff.
    assert "senapathy-classpath.txt" in green.diff()


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


def test_a_mismatched_old_text_shows_the_current_file_content(
    green: GreenWorktree,
) -> None:
    """COL-05: repeated apply_patch failures with no way to see what the
    file actually contained now. The refusal has to deliver the content it
    was already pointing at, not just say it exists somewhere else."""
    tools = make_tools(green)
    result = tools.apply_patch(SERVICE, "not here", "x")
    assert "REFUSED" in result
    assert "class CustomerService {" in result
    assert "void enrich() { outstanding(); }" in result
    assert "1| class CustomerService {" in result.replace("    1|", "1|")


# ─────────────────────────────  read_file (patch)  ─────────────────────────


def test_the_patch_registry_exposes_a_read_only_read_file(
    green: GreenWorktree,
) -> None:
    """COL-05 died reproducing old_text from memory; COL-03 hallucinated
    print_tree reaching for the same capability. The registry needs a real,
    read-only read_file -- scoped by the same containment rule the
    comprehension tools use, not by the write allowlist, since reading is
    not writing."""
    from senapathy.patch.agent import READ_FILE_TOOL
    from senapathy.tools import SourceRoot
    from senapathy.tools.source import make_read_file

    assert READ_FILE_TOOL["function"]["name"] == "read_file"

    read_file = make_read_file(SourceRoot.of(green.path))
    result = read_file(SERVICE)
    assert "class CustomerService {" in result
    assert "void enrich() { outstanding(); }" in result
    # The write allowlist is unaffected: reading a file is not touching it.
    assert not (green.path / SERVICE).read_text().startswith("REFUSED")


def test_read_file_refuses_a_path_outside_the_source_root(
    green: GreenWorktree,
) -> None:
    from senapathy.errors import GuardrailError
    from senapathy.tools import SourceRoot
    from senapathy.tools.source import make_read_file

    read_file = make_read_file(SourceRoot.of(green.path))
    with pytest.raises(GuardrailError, match="outside the repository"):
        read_file("../../../../../../etc/passwd")


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
    """Three DISTINCT edits, each followed by a build -- a repeat with no
    edit between is now free (see the no-op-retry tests below), so this has
    to exercise three real calls to prove the shared budget still counts
    them."""
    tools = make_tools(green, max_iterations=3)
    stub_maven(tools, 0, "")
    edits = [
        ("void enrich() { outstanding(); }", "void enrich() { outstanding(); } // 1"),
        ("void other() { outstanding(); }", "void other() { outstanding(); } // 2"),
        ("class CustomerService {", "class CustomerService { // 3"),
    ]
    for old, new in edits:
        tools.apply_patch(SERVICE, old, new)
        tools.build()
    assert tools.iterations == 3

    # One more edit, so this build is a real call and not a free repeat —
    # the budget must still refuse it.
    tools.apply_patch(SERVICE, "class CustomerService { // 3",
                      "class CustomerService { // 4")
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
    root = Path(__file__).resolve().parents[1] / "senapathy" / "patch"
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "senapathy.oracle" not in source and "from ..oracle" not in source
        assert "oracle.json" not in source


# ─────────────────────────────  F3: standards  ───────────────────────────


FIELD_AUTOWIRED_DIFF = """--- a/src/main/java/com/ex/service/CustomerService.java
+++ b/src/main/java/com/ex/service/CustomerService.java
@@ -10,6 +10,9 @@
 public class CustomerService {
+    @Autowired
+    private InvoiceService invoices;
+
     public Page<CustomerDto> search(SearchRequest req) {
"""


def test_a_field_autowired_fails_the_lint_and_the_message_names_the_rule() -> None:
    from senapathy.patch.standards import lint_diff, render_violations

    violations = lint_diff(FIELD_AUTOWIRED_DIFF, managed_types=["InvoiceService"])
    assert len(violations) == 1
    assert "constructor injection only" in violations[0].rule.lower()
    rendered = render_violations(violations)
    assert "No field @Autowired" in rendered
    assert "CustomerService.java" in rendered


def test_constructor_injection_passes_the_same_rule() -> None:
    from senapathy.patch.standards import lint_diff

    diff = """--- a/src/main/java/com/ex/service/CustomerService.java
+++ b/src/main/java/com/ex/service/CustomerService.java
@@ -10,6 +10,9 @@
+    @Autowired
+    public CustomerService(InvoiceService invoices) {
+        this.invoices = invoices;
"""
    assert lint_diff(diff, managed_types=["InvoiceService"]) == []


def test_a_wildcard_import_fails_the_lint() -> None:
    from senapathy.patch.standards import lint_diff

    diff = """--- a/src/main/java/com/ex/A.java
+++ b/src/main/java/com/ex/A.java
@@
+import java.util.*;
+import java.util.List;
"""
    violations = lint_diff(diff)
    assert len(violations) == 1 and "wildcard" in violations[0].rule.lower()


def test_new_on_a_spring_bean_fails_the_lint(tmp_path) -> None:
    """Read from the source, not guessed from the name: a @Component called
    Clock is container-managed and `new Clock()` is a bug."""
    from senapathy.patch.standards import lint_diff, spring_managed_types

    root = tmp_path / "repo" / "src/main/java/com/ex"
    root.mkdir(parents=True)
    (root / "Clock.java").write_text(
        "package com.ex;\n@Component\npublic class Clock {}\n")
    (root / "Plain.java").write_text("package com.ex;\npublic class Plain {}\n")

    managed = spring_managed_types(tmp_path / "repo")
    assert managed == {"Clock"}

    diff = """--- a/src/main/java/com/ex/A.java
+++ b/src/main/java/com/ex/A.java
@@
+        var c = new Clock();
+        var p = new Plain();
"""
    violations = lint_diff(diff, repo=tmp_path / "repo")
    assert len(violations) == 1
    assert "Spring-managed" in violations[0].rule
    assert "Clock" in violations[0].detail


def test_removing_a_controller_method_fails_the_lint() -> None:
    from senapathy.patch.standards import lint_diff

    diff = """--- a/src/main/java/com/ex/web/CustomerController.java
+++ b/src/main/java/com/ex/web/CustomerController.java
@@
 @RestController
-    public Page<CustomerDto> search(@RequestBody SearchRequest req) {
+    public Page<CustomerDto> find(@RequestBody SearchRequest req) {
"""
    violations = lint_diff(diff)
    assert any("public method signatures on controllers" in v.rule
               for v in violations)


def test_only_the_four_mechanical_rules_are_enforced() -> None:
    """The advisory half needs taste. A lint that rejects correct code is a
    lint someone switches off."""
    from senapathy.patch.standards import lint_diff

    advisory = """--- a/src/main/java/com/ex/service/A.java
+++ b/src/main/java/com/ex/service/A.java
@@
+    public String find(long id) { return null; }
+    public List<X> all() { return repo.findAll().stream().toList(); }
"""
    assert lint_diff(advisory, managed_types=["A"]) == []


def test_the_standards_document_is_loaded_from_disk() -> None:
    from senapathy.patch.standards import load_standards

    text = load_standards()
    assert "Constructor injection only" in text
    assert "ENFORCED" in text


# ───────────────────────  F3: the per-candidate allowlist  ───────────────


def test_a_write_outside_files_to_touch_is_refused(green) -> None:
    tools = PatchTools(
        green=green, baseline=TestCounts(),
        allowed_files=("src/main/java/com/ex/service/CustomerService.java",))

    refusal = tools.apply_patch(
        file="src/main/java/com/ex/web/CustomerController.java",
        old_text="x", new_text="y")

    assert refusal.startswith("REFUSED")
    assert "not in this plan's files_to_touch" in refusal
    assert "CustomerService.java" in refusal, "it must name what IS allowed"
    assert tools.refusals and tools.log[-1]["ok"] is False


def test_a_migration_cannot_be_written_with_apply_patch(green) -> None:
    tools = PatchTools(green=green, baseline=TestCounts())
    refusal = tools.apply_patch(
        file="src/main/resources/db/migration/V2__x.sql",
        old_text="a", new_text="b")
    assert "add_migration" in refusal


# ───────────────  COL-06: the bounded request_file_access escape  ────────


def test_request_file_access_grants_a_file_on_the_call_path(green) -> None:
    """COL-06: a query_rewrite scoped to SERVICE needed CONTROLLER, its
    caller, to adjust the call when a signature changed. CONTROLLER is on
    this endpoint's own traced call path, so it is granted."""
    tools = PatchTools(
        green=green, baseline=TestCounts(),
        allowed_files=(SERVICE,), call_path_files=(SERVICE, CONTROLLER))

    result = tools.request_file_access(CONTROLLER)
    assert result.startswith("GRANTED")
    assert tools.log[-1]["tool"] == "request_file_access"
    assert tools.log[-1]["ok"] is True

    # And CONTROLLER is now actually writable, on top of allowed_files.
    outcome = tools.apply_patch(
        file=CONTROLLER, old_text="class CustomerController {}",
        new_text="class CustomerController { }")
    assert outcome.startswith("OK")


def test_request_file_access_refuses_a_file_off_the_call_path(green) -> None:
    tools = PatchTools(
        green=green, baseline=TestCounts(),
        allowed_files=(SERVICE,), call_path_files=(SERVICE,))

    result = tools.request_file_access(CONTROLLER)
    assert result.startswith("REFUSED")
    assert "not on this endpoint's own traced call path" in result
    assert result in tools.refusals

    # Refusal is final: it did not become writable.
    outcome = tools.apply_patch(
        file=CONTROLLER, old_text="x", new_text="y")
    assert outcome.startswith("REFUSED")
    assert "not in this plan's files_to_touch" in outcome


def test_request_file_access_never_grants_a_test_file(green) -> None:
    """Even if a call chain somehow named a test file, immutability wins —
    this hard limit does not move for anything on the call path."""
    tools = PatchTools(
        green=green, baseline=TestCounts(),
        allowed_files=(SERVICE,), call_path_files=(SERVICE, TEST_FILE))

    result = tools.request_file_access(TEST_FILE)
    assert result.startswith("REFUSED")
    assert "immutable" in result


def test_request_file_access_never_grants_a_schema_file_outside_migration(
    green
) -> None:
    schema_file = "src/main/resources/schema.sql"
    tools = PatchTools(
        green=green, baseline=TestCounts(),
        allowed_files=(SERVICE,), call_path_files=(SERVICE, schema_file))

    result = tools.request_file_access(schema_file)
    assert result.startswith("REFUSED")
    assert "schema file" in result


def test_request_file_access_never_grants_outside_the_source_root(green) -> None:
    """A hard limit, independent of call-path membership: even a file the
    call chain somehow named outside src/main/java is refused."""
    resource = "src/main/resources/application.yml"
    tools = PatchTools(
        green=green, baseline=TestCounts(),
        allowed_files=(SERVICE,), call_path_files=(SERVICE, resource))

    result = tools.request_file_access(resource)
    assert result.startswith("REFUSED")
    assert "outside the detected source root" in result


def test_request_file_access_refuses_an_absolute_path(green) -> None:
    tools = PatchTools(green=green, baseline=TestCounts(),
                       call_path_files=("/etc/passwd",))
    result = tools.request_file_access("/etc/passwd")
    assert result.startswith("REFUSED") and "absolute path" in result


def test_request_file_access_refuses_a_path_escape(green) -> None:
    tools = PatchTools(green=green, baseline=TestCounts())
    result = tools.request_file_access("../../etc/passwd")
    assert result.startswith("REFUSED") and "outside the repository" in result


# ──────────────────────────  F3: add_migration  ──────────────────────────


def flyway_green(green):
    (green.path / "src/main/resources/db/migration").mkdir(
        parents=True, exist_ok=True)
    (green.path / "src/main/resources/db/migration/V1__base.sql"
     ).write_text("CREATE TABLE x (id BIGINT);\n")
    (green.path / "pom.xml").write_text("<project>flyway-core</project>")
    return green


def test_an_index_migration_is_written_and_numbered(green) -> None:
    tools = PatchTools(green=flyway_green(green), baseline=TestCounts(),
                       allowed_tables=("invoice",))
    result = tools.add_migration(
        sql="CREATE INDEX idx_invoice_customer_id_status ON invoice "
            "(customer_id, status);",
        rationale="the aggregate filters on both columns",
        evidence="rows examined 2,645 -> 571")

    assert result.startswith("OK")
    assert "V2__add_invoice_customer_id_status_index.sql" in result
    assert tools.migrations_written == [
        "src/main/resources/db/migration/V2__add_invoice_customer_id_status_index.sql"]
    # A migration is an edit: build and tests must run again.
    assert tools.edits_since_verify == 1
    assert tools.last_build_ok is False and tools.last_tests_ok is False


@pytest.mark.parametrize("sql", [
    "ALTER TABLE invoice ADD COLUMN total DECIMAL(19,2)",
    "DROP INDEX idx_invoice_customer_id ON invoice",
    "CREATE TABLE shadow (id BIGINT)",
    "CREATE INDEX idx_a ON invoice (status); DROP TABLE invoice;",
    "UPDATE invoice SET status = 'PAID'",
])
def test_anything_that_is_not_one_create_index_is_refused(green,
                                                          sql: str) -> None:
    """Parsed, not trusted: 'additive index only' is a property of the code."""
    tools = PatchTools(green=flyway_green(green), baseline=TestCounts(),
                       allowed_tables=("invoice",))
    result = tools.add_migration(sql=sql, rationale="why")

    assert result.startswith("REFUSED")
    assert "CREATE INDEX" in result
    assert tools.migrations_written == []
    written = list((tools.green.path / "src/main/resources/db/migration"
                    ).glob("V2*.sql"))
    assert written == [], "nothing may reach the disk"


def test_an_index_on_a_table_off_the_path_is_refused(green) -> None:
    tools = PatchTools(green=flyway_green(green), baseline=TestCounts(),
                       allowed_tables=("invoice",))
    result = tools.add_migration(
        sql="CREATE INDEX idx_u ON users (email);", rationale="why")
    assert result.startswith("REFUSED") and "users" in result


# ────────────────────────  F3: the in-loop benchmark  ────────────────────


def test_the_benchmark_delta_reaches_the_agent_when_tests_pass(green) -> None:
    from senapathy.patch.bench import QuickResult

    tools = PatchTools(
        green=green, baseline=TestCounts(executed=1, files=1),
        benchmark=lambda: QuickResult(
            p50_ms=640.0, p95_ms=700.0, sql_per_request=2.0, requests=20,
            baseline_p50_ms=1022.0, baseline_sql_per_request=6.0),
    )
    tools.last_build_ok = True
    stub_maven(tools, 0, "Tests run: 1, Failures: 0, Errors: 0, Skipped: 0")

    result = tools.run_tests()
    assert "TESTS OK" in result
    assert "1,022 ms -> 640 ms (-37%, faster)" in result
    assert "SQL statements per request 6.0 -> 2.0" in result
    assert "not the real measurement" in result, "it must say it is heuristic"
    assert tools.benchmark_regressed is False


def test_a_regression_is_visible_to_the_agent_before_it_emits(green) -> None:
    from senapathy.patch.bench import QuickResult

    tools = PatchTools(
        green=green, baseline=TestCounts(executed=1, files=1),
        benchmark=lambda: QuickResult(
            p50_ms=1500.0, p95_ms=1600.0, sql_per_request=8.0, requests=20,
            baseline_p50_ms=1000.0, baseline_sql_per_request=6.0),
    )
    tools.last_build_ok = True
    stub_maven(tools, 0, "Tests run: 1, Failures: 0, Errors: 0, Skipped: 0")

    result = tools.run_tests()
    assert "SLOWER" in result
    assert "IT GOT SLOWER" in result
    assert "Do not emit this patch as it stands" in result
    assert tools.benchmark_regressed is True


def test_a_benchmark_that_cannot_run_does_not_fail_the_phase(green) -> None:
    def explode():
        raise RuntimeError("no port free")

    tools = PatchTools(green=green,
                       baseline=TestCounts(executed=1, files=1),
                       benchmark=explode)
    tools.last_build_ok = True
    stub_maven(tools, 0, "Tests run: 1, Failures: 0, Errors: 0, Skipped: 0")

    result = tools.run_tests()
    assert "TESTS OK" in result
    assert "could not run" in result
    assert tools.last_tests_ok is True


# ───────────────────  F3: the phase implements the PLAN  ─────────────────


def test_the_patch_phase_refuses_to_start_without_a_plan(tmp_path) -> None:
    from senapathy.errors import ContractViolation
    from senapathy.patch.agent import load_plan_inputs

    (tmp_path / "hypotheses.json").write_text("{}")
    with pytest.raises(ContractViolation, match="Run `senapathy plan`"):
        load_plan_inputs(tmp_path)


def test_the_prompt_is_built_from_the_candidate_not_the_top_hypothesis(
    tmp_path
) -> None:
    """Acceptance 1, as a property of the prompt: the candidate's approach and
    predicted effect are in it, and the plan's files_to_touch are the
    allowlist."""
    import json as _json

    from senapathy.contracts import CandidateFix, EndpointModel, RiskTier
    from senapathy.patch.agent import build_task

    green = tmp_path / "green"
    service = green / "src/main/java/com/ex/service/CustomerService.java"
    service.parent.mkdir(parents=True)
    service.write_text("public class CustomerService { }\n")

    candidate = CandidateFix(
        id="c2", targets_hypothesis="h2", approach="fetch_join",
        description="replace the per-row sum with a join fetch",
        mechanism_changed="SQL statements per request",
        predicted_effect="queries per request drop from 22 to 2",
        files_to_touch=["src/main/java/com/ex/service/CustomerService.java"],
        what_could_break=["duplicate rows if the join is not distinct"],
        preserves_response_bytes=True, risk_tier=RiskTier.LOW,
    )
    model = EndpointModel.model_validate({
        "route": "POST /api/customers/search",
        "call_chain": [{"method": "CustomerService.search",
                        "file": "src/main/java/com/ex/service/CustomerService.java",
                        "line": 1, "does": "searches"}],
        "data_access": [],
        "response_shape": {"dto": "CustomerDto", "fields_returned": ["tags"],
                           "unbounded_collections": ["tags"]},
        "per_row_work": [], "uncertainties": [],
    })

    task = build_task(candidate=candidate, model=model, green=green,
                      fixability_reason="batchable: fetch join or @EntityGraph",
                      standards="Constructor injection only.")

    assert "c2" in task and "fetch_join" in task
    assert "queries per request drop from 22 to 2" in task
    assert "batchable: fetch join or @EntityGraph" in task
    assert "duplicate rows if the join is not distinct" in task
    # The frozen fields are NAMED, not described.
    assert "UNBOUNDED COLLECTIONS IN THE RESPONSE: tags" in task
    # The standards travel with it.
    assert "Constructor injection only." in task
    # And the allowlist is the plan's.
    assert "src/main/java/com/ex/service/CustomerService.java" in task
    assert "ranked" not in task.lower().split("=== how to work")[0]


def test_the_test_count_guardrail_fires_on_real_maven_output(green) -> None:
    """Acceptance 5's other half, against the strings maven actually prints.

    With the test file present, surefire reports a count. With it deleted,
    `mvn test` prints NO "Tests run" line at all and still says BUILD SUCCESS —
    which is exactly why the count is checked here rather than trusted to the
    build's exit code.
    """
    with_test = """[INFO] Running com.ex.CustomerSearchIT
[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 2.750 s
[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0
[INFO] BUILD SUCCESS"""
    without_test = "[INFO] BUILD SUCCESS"

    baseline = parse_test_counts(with_test, green.path)
    assert baseline.executed == 1, "surefire must actually run the IT now"

    tools = PatchTools(green=green, baseline=baseline)
    tools.last_build_ok = True
    stub_maven(tools, 0, without_test)

    result = tools.run_tests()
    assert "GUARDRAIL VIOLATION" in result
    assert "tests executed dropped from 1 to 0" in result
    assert tools.guardrail_violations, "the run is over, not warned about"
    assert tools.last_tests_ok is False


# ──────────────────────  input-hash provenance  ───────────────────────────


def test_persist_stamps_the_current_baseline_hash(tmp_path: Path) -> None:
    """Resume's skip check compares this against the run's CURRENT
    baseline.json -- a patch built against a baseline that has since been
    re-measured is not evidence for the current one."""
    import json as _json

    from senapathy.measure.baseline import baseline_hash
    from senapathy.patch.agent import _persist

    directory = tmp_path / "run"
    directory.mkdir()
    baseline_doc = {"derived": {"total_sql_statements": 3}}
    (directory / "baseline.json").write_text(_json.dumps(baseline_doc))

    document = {"outcome": "PATCHED", "patch": {"diff": ""}}
    _persist(directory, document, green=None)

    saved = _json.loads((directory / "patch.json").read_text())
    assert saved["baseline_hash"] == baseline_hash(baseline_doc)


def test_build_document_targets_the_selected_candidates_hypothesis(
) -> None:
    """The exact bug found in run INV-03: plan selected candidate c1
    (targeting h3), but patch.json recorded targets_hypothesis=h1 --
    `ranked_ids[0]` is the DIAGNOSIS's own ranking by attributed
    milliseconds, something this phase is specifically supposed to have
    stopped trusting (see build_task's docstring). The selected candidate,
    not the hypothesis ranking, is the source of truth here."""
    from senapathy.contracts import CandidateFix, RiskTier
    from senapathy.patch.agent import _build_document
    from senapathy.patch.tools import TestCounts

    candidate = CandidateFix(
        id="c1", targets_hypothesis="h3", approach="dto_projection",
        description="project a DTO instead of mapping the full entity",
        files_to_touch=["src/main/java/com/ex/repo/InvoiceRepository.java"],
        mechanism_changed="eliminates the per-row customer lookup",
        predicted_effect="fewer statements per request",
        what_could_break=[], preserves_response_bytes=True,
        risk_tier=RiskTier.LOW,
    )
    hypotheses = {"ranked_ids": ["h1", "h2", "h3"]}  # h1 ranked highest

    class FakeGreen:
        path = Path("/green")
        branch = "b"
        base_commit = "a" * 40
        carried_over: list[str] = []
        skipped: list[str] = []
        checkpoints: list[str] = []

        def diff_stat(self):
            return [], 0, 0

        def diff(self):
            return ""

        def blue_status(self):
            return "clean"

    class FakeTools:
        iterations = 1
        max_iterations = 8
        log: list[dict] = []
        refusals: list[str] = []
        guardrail_violations: list[str] = []
        last_error = None
        _granted_files: set[str] = set()

    document = _build_document(
        green=FakeGreen(), tools=FakeTools(), candidate=candidate,
        baseline=TestCounts(), accepted={}, rejections=[],
        loop_error=None, blue_status_before="clean",
    )

    assert document["targets_hypothesis"] == "h3"
    assert document["patch"]["plan_id"] == "c1"
    # Proof this is reading the candidate, not falling back to the ranking:
    assert document["targets_hypothesis"] != hypotheses["ranked_ids"][0]


# ──────────────────  surefire-read failures and no-op retries  ────────────


def test_run_tests_failure_returns_failing_test_names_not_the_banner(
    green: GreenWorktree,
) -> None:
    """The actual regression: three COL-03 patches each got back 40 lines of
    `[INFO] Scanning for projects...` and no failing test name anywhere.
    The detail has to come from surefire's XML, which is unaffected by a
    head-of-output line budget."""
    reports = green.path / "target" / "surefire-reports"
    reports.mkdir(parents=True)
    (reports / "TEST-com.ex.CustomerSearchIT.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="com.ex.CustomerSearchIT" tests="1" failures="1" errors="0" skipped="0">
  <testcase name="searchReturnsOk" classname="com.ex.CustomerSearchIT" time="0.1">
    <failure message="expected: &lt;200&gt; but was: &lt;500&gt;" type="org.opentest4j.AssertionFailedError">org.opentest4j.AssertionFailedError: expected: &lt;200&gt; but was: &lt;500&gt;
	at org.junit.jupiter.api.Assertions.assertEquals(Assertions.java:200)
	at com.ex.CustomerSearchIT.searchReturnsOk(CustomerSearchIT.java:28)
</failure>
  </testcase>
</testsuite>
""")

    banner = (
        "[INFO] Scanning for projects...\n"
        + "\n".join(f"[INFO] -----< some line {i} >-----" for i in range(50))
        + "\n[INFO] Tests run: 1, Failures: 1, Errors: 0, Skipped: 0\n"
        "[INFO] BUILD FAILURE"
    )
    tools = make_tools(green, max_iterations=5)
    tools.last_build_ok = True
    stub_maven(tools, 1, banner)

    result = tools.run_tests()

    assert "TESTS FAILED" in result
    assert "com.ex.CustomerSearchIT.searchReturnsOk" in result
    assert "expected: <200> but was: <500>" in result
    assert "CustomerSearchIT.java:28" in result
    # The banner a 40-line head-of-output truncation would have shown
    # instead is gone from the detail entirely.
    assert "Scanning for projects" not in result
    assert "searchReturnsOk" in tools.last_error


def test_a_crash_before_any_report_falls_back_to_raw_output(
    green: GreenWorktree,
) -> None:
    """No target/surefire-reports/ at all -- a JVM-level crash before any
    test ran -- still has to produce something, via the old behaviour."""
    tools = make_tools(green, max_iterations=5)
    tools.last_build_ok = True
    stub_maven(tools, 1, "[INFO] Scanning for projects...\nOutOfMemoryError")

    result = tools.run_tests()
    assert "TESTS FAILED" in result
    assert "Scanning for projects" in result


def test_a_repeated_build_with_no_edit_costs_no_iteration(
    green: GreenWorktree,
) -> None:
    tools = make_tools(green, max_iterations=5)
    stub_maven(tools, 0, "")

    first = tools.build()
    assert tools.iterations == 1

    second = tools.build()
    assert tools.iterations == 1, "a repeat with no edit must not spend an iteration"
    assert "SAME AS LAST TIME" in second
    assert "BUILD OK" in second


def test_a_repeated_run_tests_with_no_edit_costs_no_iteration(
    green: GreenWorktree,
) -> None:
    with_test = (
        "[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0, "
        "Time elapsed: 0.1 s -- in com.ex.CustomerSearchIT\n"
        "[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0\n"
        "[INFO] BUILD SUCCESS"
    )
    tools = make_tools(green, max_iterations=5)
    tools.last_build_ok = True
    stub_maven(tools, 0, with_test)

    first = tools.run_tests()
    assert tools.iterations == 1
    assert "TESTS OK" in first

    second = tools.run_tests()
    assert tools.iterations == 1, "a repeat with no edit must not spend an iteration"
    assert "SAME AS LAST TIME" in second
    assert "TESTS OK" in second


def test_an_edit_between_calls_makes_the_next_one_real(
    green: GreenWorktree,
) -> None:
    tools = make_tools(green, max_iterations=5)
    stub_maven(tools, 0, "")

    tools.build()
    assert tools.iterations == 1

    tools.apply_patch(SERVICE, "void other() { outstanding(); }",
                      "void other() { outstanding(); } // changed")
    result = tools.build()
    assert tools.iterations == 2, "an edit between calls must make the next build real"
    assert "SAME AS LAST TIME" not in result


# ───────────────────────  trace wiring and abandonment  ───────────────────


def test_abandon_patch_requires_a_reason() -> None:
    """The schema is the enforcement: a provider that honours `required`
    cannot generate a call missing it, and `emit_patch`'s own handler trusts
    the same mechanism rather than re-checking its fields in Python."""
    from senapathy.patch.agent import ABANDON_PATCH_TOOL

    params = ABANDON_PATCH_TOOL["function"]["parameters"]
    assert params["required"] == ["reason"]
    assert "reason" in params["properties"]


class _ScriptedLLM:
    """A minimal scripted client: each call pops the next canned response,
    in the same shape run_agent expects from a real provider."""

    def __init__(self, *turns: tuple[str, dict]) -> None:
        self._turns = list(turns)

    def complete(self, **kwargs):
        from senapathy.contracts import LLMResponse, Usage

        if not self._turns:
            return LLMResponse(
                content=[{"type": "text", "text": "done"}],
                stop_reason="end_turn", usage=Usage(), latency_ms=1.0,
                raw_message={"role": "assistant", "content": "done"})
        name, args = self._turns.pop(0)
        return LLMResponse(
            content=[{"type": "tool_use", "id": "c1", "name": name, "input": args}],
            stop_reason="tool_use", usage=Usage(), latency_ms=1.0,
            raw_message={"role": "assistant", "tool_calls": [{"id": "c1"}]})

    def forced_tool_choice(self, name: str) -> dict:
        return {"type": "function", "function": {"name": name}}

    def tool_result_messages(self, results):
        return [{"role": "tool", "tool_call_id": r.id, "content": r.content}
                for r in results]


def test_a_patch_run_traces_tool_call_and_tool_result_under_phase_patch(
    blue: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual regression: patch wrote phase_start/phase_end and nothing
    else, because run_patch never passed its trace handle into run_agent.
    Driving one real apply_patch + build + abandon_patch through run_patch
    must produce the same llm_request/llm_response/tool_call/tool_result
    shape understand/plan/diagnose already get -- tagged phase="patch" --
    plus a note for the build iteration, and an abandoned_reason in the
    artifact naming the reason the model actually gave.
    """
    import json as _json

    from senapathy.contracts import CandidateFix, EndpointModel, RiskTier
    from senapathy.contracts.plan import Plan
    from senapathy.patch import agent as patch_agent
    from senapathy.trace import EventType, Trace

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    candidate = CandidateFix(
        id="c1", targets_hypothesis="h1", approach="other",
        description="rewrite enrich", mechanism_changed="fewer calls",
        predicted_effect="latency drops",
        files_to_touch=[SERVICE], what_could_break=[],
        preserves_response_bytes=True, risk_tier=RiskTier.LOW,
    )
    plan = Plan(
        endpoint="GET /x", measured_total_ms=100.0, ranked=[],
        considered=["h1"], candidates=[candidate], selected_id="c1",
        selection_reason="only candidate", base_commit="a" * 40,
    )
    (run_dir / "plan.json").write_text(_json.dumps(plan.model_dump(mode="json")))
    (run_dir / "hypotheses.json").write_text(_json.dumps({
        "measured_total_ms": 100.0, "ranked_ids": ["h1"],
        "hypotheses": [{"id": "h1", "claim": "c", "mechanism": "m",
                        "layer": "application_logic", "evidence": [],
                        "attributed_ms": 50.0, "prediction": "p",
                        "confidence": 0.8}],
    }))
    (run_dir / "endpoint_model.json").write_text(_json.dumps(
        EndpointModel.model_validate({
            "route": "GET /x",
            "entry": {"file": "x", "line": 1}, "path": [{"file": "x", "line": 1}],
            "tables": [], "uncertainties": [],
        }).model_dump(mode="json")))

    monkeypatch.setattr(
        patch_agent, "_measure_baseline",
        lambda green, java_home, profile=None: TestCounts(files=1))
    monkeypatch.setattr(
        PatchTools, "_maven",
        lambda self, args, timeout_s: subprocess.CompletedProcess(
            ["mvn", *args], returncode=0, stdout="BUILD SUCCESS", stderr=""))

    client = _ScriptedLLM(
        ("apply_patch", {"file": SERVICE,
                         "old_text": "void enrich() { outstanding(); }",
                         "new_text": "void enrich() { outstandingBatched(); }"}),
        ("build", {}),
        ("abandon_patch", {"reason": "run_tests keeps timing out in this sandbox"}),
    )

    trace = Trace(run_dir, "r1")
    from senapathy.agent import LoopBudget

    document = patch_agent.run_patch(
        run_dir, run_id="r1", repo=blue, green_dir=tmp_path / "green-traced",
        budget=LoopBudget(max_steps=10), client=client, trace=trace,
        keep_worktree=False,
    )

    assert document["outcome"] == "ABANDONED"
    assert "run_tests keeps timing out" in document["abandoned_reason"]

    events = [
        _json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text().splitlines()
    ]
    patch_events = [e for e in events if e["phase"] == "patch"]
    kinds = {e["type"] for e in patch_events}
    assert EventType.LLM_REQUEST.value in kinds
    assert EventType.LLM_RESPONSE.value in kinds
    assert EventType.TOOL_CALL.value in kinds
    assert EventType.TOOL_RESULT.value in kinds

    tool_calls = {e["tool"] for e in patch_events if e["type"] == "tool_call"}
    assert tool_calls == {"apply_patch", "build", "abandon_patch"}

    build_notes = [e for e in patch_events
                  if e["type"] == "note" and "build iteration" in e.get("note", "")]
    assert build_notes, "no per-build-iteration note was emitted"
    assert "OK" in build_notes[0]["note"]


def test_request_file_access_outcomes_both_reach_patch_json(
    blue: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """COL-06, end to end: a candidate scoped to SERVICE requests CONTROLLER
    (on the call path — granted) and an unrelated file (not on the call
    path — refused). patch.json must show both, so the report can compare
    the scope actually used against the scope the plan declared."""
    import json as _json

    from senapathy.agent import LoopBudget
    from senapathy.contracts import CandidateFix, EndpointModel, RiskTier
    from senapathy.contracts.plan import Plan
    from senapathy.patch import agent as patch_agent
    from senapathy.trace import Trace

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    candidate = CandidateFix(
        id="c1", targets_hypothesis="h1", approach="query_rewrite",
        description="widen the comparison on the parameter side",
        mechanism_changed="rows examined", predicted_effect="latency drops",
        files_to_touch=[SERVICE], what_could_break=[],
        preserves_response_bytes=True, risk_tier=RiskTier.LOW,
    )
    plan = Plan(
        endpoint="GET /x", measured_total_ms=100.0, ranked=[],
        considered=["h1"], candidates=[candidate], selected_id="c1",
        selection_reason="only candidate", base_commit="a" * 40,
    )
    (run_dir / "plan.json").write_text(_json.dumps(plan.model_dump(mode="json")))
    (run_dir / "hypotheses.json").write_text(_json.dumps({
        "measured_total_ms": 100.0, "ranked_ids": ["h1"],
        "hypotheses": [{"id": "h1", "claim": "c", "mechanism": "m",
                        "layer": "database_query", "evidence": [],
                        "attributed_ms": 50.0, "prediction": "p",
                        "confidence": 0.8}],
    }))
    (run_dir / "endpoint_model.json").write_text(_json.dumps(
        EndpointModel.model_validate({
            "route": "GET /x",
            "entry": {"file": CONTROLLER, "line": 1},
            "path": [{"file": CONTROLLER, "line": 1}],
            "call_chain": [
                {"method": "CustomerController.search", "file": CONTROLLER,
                 "line": 1, "does": "delegates"},
                {"method": "CustomerService.enrich", "file": SERVICE,
                 "line": 2, "does": "runs the query"},
            ],
            "tables": [], "uncertainties": [],
        }).model_dump(mode="json")))

    monkeypatch.setattr(
        patch_agent, "_measure_baseline",
        lambda green, java_home, profile=None: TestCounts(files=1))
    monkeypatch.setattr(
        PatchTools, "_maven",
        lambda self, args, timeout_s: subprocess.CompletedProcess(
            ["mvn", *args], returncode=0, stdout="BUILD SUCCESS", stderr=""))

    client = _ScriptedLLM(
        ("request_file_access", {"file": CONTROLLER}),
        ("request_file_access",
         {"file": "src/main/java/com/ex/other/Unrelated.java"}),
        ("apply_patch", {"file": CONTROLLER,
                         "old_text": "class CustomerController {}",
                         "new_text": "class CustomerController { }"}),
        ("build", {}),
        ("abandon_patch", {"reason": "stopping once the grant is proven"}),
    )

    document = patch_agent.run_patch(
        run_dir, run_id="r1", repo=blue, green_dir=tmp_path / "green-access",
        budget=LoopBudget(max_steps=10),
        client=client, trace=Trace(run_dir, "r1"), keep_worktree=False,
    )

    requests = [e for e in document["tool_log"] if e["tool"] == "request_file_access"]
    assert len(requests) == 2
    granted = next(e for e in requests if e["args"]["file"] == CONTROLLER)
    refused = next(e for e in requests
                   if e["args"]["file"] == "src/main/java/com/ex/other/Unrelated.java")
    assert granted["ok"] is True and "GRANTED" in granted["result"]
    assert refused["ok"] is False and "REFUSED" in refused["result"]
    assert any("not on this endpoint's own traced call path" in r
              for r in document["refusals"])


# ───────────────  COL-04: a prose abandonment must not vanish  ────────────


def test_abandoned_reason_captures_the_models_own_final_words() -> None:
    """COL-04: the model wrote a coherent, specific explanation as plain
    text instead of calling abandon_patch (stop_reason end_turn, iteration
    0). Before this fix, patch.json recorded only the generic "the model
    stopped calling tools" and the actual explanation lived only in
    trace.jsonl -- invisible to the person the report is for."""
    from senapathy.patch.agent import _abandoned_reason

    final_text = (
        "Reason for abandonment: Implementing the proposed performance "
        "improvement requires adding a @SqlResultSetMapping... The "
        "repository file ... is the only file permitted for modification. "
        "... the necessary mapping cannot be done within this single file."
    )

    class FakeTools:
        guardrail_violations: list[str] = []
        last_error = ""

    reason = _abandoned_reason(
        abandonment={}, loop_error=None, tools=FakeTools(),
        loop_stopped_because="the model stopped calling tools",
        loop_final_text=final_text,
    )

    assert "the necessary mapping cannot be done within this single file" in reason
    assert reason != ("the agent loop stopped on its own: "
                      "the model stopped calling tools")


def test_abandoned_reason_ignores_final_text_from_an_unrelated_stop() -> None:
    """final_text is only trustworthy as a stopping explanation when the
    model's LAST turn genuinely had no tool call in it -- using it for a
    budget-exhaustion or no-progress stop would surface stale content from
    an earlier, unrelated turn."""
    from senapathy.patch.agent import _abandoned_reason

    class FakeTools:
        guardrail_violations: list[str] = []
        last_error = ""

    reason = _abandoned_reason(
        abandonment={}, loop_error=None, tools=FakeTools(),
        loop_stopped_because="reached the step ceiling of 80",
        loop_final_text="some stale text from three turns ago",
    )

    assert "stale text" not in reason
    assert "reached the step ceiling of 80" in reason
