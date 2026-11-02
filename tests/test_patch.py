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
    from perfagent.patch.standards import lint_diff, render_violations

    violations = lint_diff(FIELD_AUTOWIRED_DIFF, managed_types=["InvoiceService"])
    assert len(violations) == 1
    assert "constructor injection only" in violations[0].rule.lower()
    rendered = render_violations(violations)
    assert "No field @Autowired" in rendered
    assert "CustomerService.java" in rendered


def test_constructor_injection_passes_the_same_rule() -> None:
    from perfagent.patch.standards import lint_diff

    diff = """--- a/src/main/java/com/ex/service/CustomerService.java
+++ b/src/main/java/com/ex/service/CustomerService.java
@@ -10,6 +10,9 @@
+    @Autowired
+    public CustomerService(InvoiceService invoices) {
+        this.invoices = invoices;
"""
    assert lint_diff(diff, managed_types=["InvoiceService"]) == []


def test_a_wildcard_import_fails_the_lint() -> None:
    from perfagent.patch.standards import lint_diff

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
    from perfagent.patch.standards import lint_diff, spring_managed_types

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
    from perfagent.patch.standards import lint_diff

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
    from perfagent.patch.standards import lint_diff

    advisory = """--- a/src/main/java/com/ex/service/A.java
+++ b/src/main/java/com/ex/service/A.java
@@
+    public String find(long id) { return null; }
+    public List<X> all() { return repo.findAll().stream().toList(); }
"""
    assert lint_diff(advisory, managed_types=["A"]) == []


def test_the_standards_document_is_loaded_from_disk() -> None:
    from perfagent.patch.standards import load_standards

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
    from perfagent.patch.bench import QuickResult

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
    from perfagent.patch.bench import QuickResult

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
    from perfagent.errors import ContractViolation
    from perfagent.patch.agent import load_plan_inputs

    (tmp_path / "hypotheses.json").write_text("{}")
    with pytest.raises(ContractViolation, match="Run `perfagent plan`"):
        load_plan_inputs(tmp_path)


def test_the_prompt_is_built_from_the_candidate_not_the_top_hypothesis(
    tmp_path
) -> None:
    """Acceptance 1, as a property of the prompt: the candidate's approach and
    predicted effect are in it, and the plan's files_to_touch are the
    allowlist."""
    import json as _json

    from perfagent.contracts import CandidateFix, EndpointModel, RiskTier
    from perfagent.patch.agent import build_task

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
