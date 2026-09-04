"""The tool layer, against a fixture repo. No network, no database."""

from __future__ import annotations

from pathlib import Path

import pytest

from perfagent.errors import GuardrailError, ToolError
from perfagent.tools import SourceRoot, ToolRegistry, ToolSpec
from perfagent.tools.routes import (build_route_table, find_matches,
                                    make_find_route, route_table)
from perfagent.tools.source import (make_list_dir, make_read_file,
                                    make_search_code, source_tools)


@pytest.fixture
def root(fixture_repo: Path) -> SourceRoot:
    return SourceRoot.of(fixture_repo)


# ────────────────────────────  containment  ──────────────────────────────


def test_the_root_is_injected_not_a_module_constant(fixture_repo: Path,
                                                    tmp_path: Path) -> None:
    """Two repositories must be usable in one process."""
    other = tmp_path / "other"
    (other / "src").mkdir(parents=True)
    (other / "README.md").write_text("different\n")

    first = make_read_file(SourceRoot.of(fixture_repo))
    second = make_read_file(SourceRoot.of(other))
    assert "fixture repo" in first("README.md")
    assert "different" in second("README.md")


def test_a_path_escape_is_refused_as_a_guardrail(root: SourceRoot) -> None:
    with pytest.raises(GuardrailError) as caught:
        root.resolve("../../etc/passwd")
    assert "outside the repository" in str(caught.value)


def test_a_dotted_path_that_stays_inside_is_allowed(root: SourceRoot) -> None:
    inside = root.resolve("src/main/../main/java/com/ex/web/OrderController.java")
    assert inside.exists()


def test_the_escape_refusal_is_a_tool_error_subclass(root: SourceRoot) -> None:
    """It used to be ValueError, which the loop could only treat as a crash."""
    assert issubclass(GuardrailError, ToolError)
    with pytest.raises(ToolError):
        make_read_file(root)("../../../etc/passwd")


# ─────────────────────────────  read_file  ───────────────────────────────


def test_read_file_numbers_every_line(root: SourceRoot) -> None:
    out = make_read_file(root)("src/main/java/com/ex/web/OrderController.java")
    assert "    1| package com.ex.web;" in out
    assert "of 25)" in out


def test_read_file_suggests_a_spelling_for_a_typo(root: SourceRoot) -> None:
    out = make_read_file(root)("src/main/java/com/ex/web/OrdreController.java")
    assert "File not found" in out
    assert "OrderController.java" in out


def test_read_file_points_at_list_dir_for_a_directory(root: SourceRoot) -> None:
    assert "list_dir" in make_read_file(root)("src/main")


def test_read_file_says_how_to_continue_a_truncated_file(
    fixture_repo: Path
) -> None:
    long_file = fixture_repo / "src/main/java/com/ex/Long.java"
    long_file.write_text("\n".join(f"// line {n}" for n in range(1, 900)))
    out = make_read_file(SourceRoot.of(fixture_repo))(
        "src/main/java/com/ex/Long.java")
    assert "truncated at line 400 of 899" in out
    assert "start_line=401" in out


# ────────────────────────────  search_code  ──────────────────────────────


def test_search_code_states_the_per_file_cap(root: SourceRoot) -> None:
    """Capping at N per file silently hid matches. The model has to be told
    that 'three matches in Foo.java' is a display cap, not a count."""
    out = make_search_code(root)("import")
    assert "PER FILE" in out


def test_the_per_file_cap_is_a_parameter(root: SourceRoot) -> None:
    search = make_search_code(root)
    assert "at most 1 match(es) shown PER FILE" in search("public", max_per_file=1)
    assert "at most 9 match(es) shown PER FILE" in search("public", max_per_file=9)


def test_no_matches_is_a_fact_not_an_error(root: SourceRoot) -> None:
    assert make_search_code(root)("zzz-not-here") == "No matches for 'zzz-not-here'."


def test_search_results_are_repo_relative(root: SourceRoot) -> None:
    out = make_search_code(root)("class OrderController")
    assert str(root.path) not in out
    assert "src/main/java/com/ex/web/OrderController.java" in out


# ──────────────────────────────  list_dir  ───────────────────────────────


def test_list_dir_marks_directories(root: SourceRoot) -> None:
    out = make_list_dir(root)("src/main/java/com/ex")
    assert "web/" in out and "service/" in out


# ─────────────────────────────  find_route  ──────────────────────────────


def test_find_route_composes_class_and_method_mappings(root: SourceRoot) -> None:
    """The whole reason this tool exists: '/api/orders' is on the class and
    '/search' is on the method, and a model composing them from grep gets it
    wrong often enough to cost several turns."""
    out = make_find_route(root)("POST", "/api/orders/search")
    assert "POST /api/orders/search" in out
    assert "OrderController.search" in out
    assert "OrderController.java:12" in out
    assert "SearchRequest req" in out
    assert "Page<OrderDto>" in out


def test_find_route_reports_a_template_match_as_exact(root: SourceRoot) -> None:
    out = make_find_route(root)("GET", "/api/orders/{id}")
    assert "Matched the template exactly" in out


def test_a_concrete_path_matches_a_template_and_says_so(root: SourceRoot) -> None:
    out = make_find_route(root)("GET", "/api/orders/42")
    assert "CONCRETE path" in out
    assert "/api/orders/{id}" in out


def test_an_ambiguous_match_returns_every_candidate(root: SourceRoot) -> None:
    """Two controllers declare GET /api/orders/{id}. Picking one silently is
    the failure mode this tool exists to remove."""
    out = make_find_route(root)("GET", "/api/orders/{id}")
    assert "AMBIGUOUS" in out
    assert "OrderController.findOne" in out and "HealthController.shadow" in out


def test_no_match_lists_the_closest_routes(root: SourceRoot) -> None:
    with pytest.raises(ToolError) as caught:
        make_find_route(root)("GET", "/api/order/search")
    message = str(caught.value)
    assert "No route matches GET /api/order/search" in message
    assert "/api/orders/search" in message


def test_bare_keyed_and_array_path_forms_are_all_parsed(root: SourceRoot) -> None:
    labels = set(route_table(root).labels())
    assert "GET /health" in labels                     # path = "..."
    assert "POST /api/orders/search" in labels          # bare string
    assert "GET /api/orders/legacy" in labels           # value = {..}, method =
    assert "GET /api/orders/old" in labels


def test_the_route_table_is_cached_and_invalidated_on_mtime(
    fixture_repo: Path
) -> None:
    root = SourceRoot.of(fixture_repo)
    assert route_table(root) is route_table(root), "second call should be cached"

    controller = fixture_repo / "src/main/java/com/ex/web/HealthController.java"
    controller.write_text(controller.read_text().replace('"/health"', '"/ping"'))
    labels = set(route_table(root).labels())
    assert "GET /ping" in labels and "GET /health" not in labels


def test_a_repo_with_no_controllers_says_so(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    (empty / "src").mkdir(parents=True)
    with pytest.raises(ToolError, match="no Spring routes"):
        make_find_route(SourceRoot.of(empty))("GET", "/x")


def test_matching_is_whitespace_and_trailing_slash_tolerant(root: SourceRoot) -> None:
    table = build_route_table(root)
    for spelling in ("/api/orders/search", "/api/orders/search/",
                     "api/orders/search"):
        matches, how = find_matches(table, "post", spelling)
        assert how == "exact" and matches


# ─────────────────────────────  the registry  ────────────────────────────


def test_the_registry_reports_its_terminal_tool(root: SourceRoot) -> None:
    registry = ToolRegistry().extend(source_tools(root))
    assert registry.terminal_name() is None
    registry.add(ToolSpec(name="submit", description="d", parameters={},
                          handler=lambda: "ok", terminal=True))
    assert registry.terminal_name() == "submit"
    assert set(registry.handlers()) == set(registry.names())


def test_search_results_are_deterministically_ordered(root: SourceRoot) -> None:
    """ripgrep is multithreaded by default, so the same search returns the same
    lines in a different order run to run. That makes an agent run
    unreproducible and turns every recorded cassette into a coin flip."""
    search = make_search_code(root)
    runs = {search("public") for _ in range(5)}
    assert len(runs) == 1
