"""find_route — the route table, built from the AST rather than from grep.

This tool exists to delete the search-and-guess step that opened every run.
`@RequestMapping("/api/customers")` on the class and `@GetMapping("/{id}")` on
the method are two facts in two places, and a model asked to compose them from
grep output gets it wrong often enough to burn several turns before it even
starts tracing.

Parsing is tree-sitter, not regex: `@GetMapping` inside a comment, inside a
string, or on a nested class are all things a regex gets wrong and a parser
does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Callable

from ..errors import ToolError
from . import SourceRoot, ToolSpec
from .javaparse import (ANY_METHOD, ClassMappings, class_declarations,
                        class_mappings, parser)

MAX_CANDIDATES_SHOWN = 12


@dataclass(frozen=True)
class Route:
    http_method: str
    path: str
    controller: str
    method: str
    file: str
    line: int
    parameter_types: list[str]
    return_type: str
    annotation: str

    def label(self) -> str:
        return f"{self.http_method} {self.path}"

    def render(self) -> str:
        params = ", ".join(self.parameter_types) or "(none)"
        return (
            f"{self.http_method} {self.path}\n"
            f"  handler:    {self.controller}.{self.method}\n"
            f"  location:   {self.file}:{self.line}\n"
            f"  annotation: {self.annotation}\n"
            f"  parameters: {params}\n"
            f"  returns:    {self.return_type}"
        )


@dataclass
class RouteTable:
    routes: list[Route] = field(default_factory=list)
    files_scanned: int = 0

    def labels(self) -> list[str]:
        return [r.label() for r in self.routes]


def _join(base: str, leaf: str) -> str:
    combined = f"/{base.strip('/')}/{leaf.strip('/')}" if base or leaf else "/"
    collapsed = re.sub(r"/+", "/", combined)
    return collapsed.rstrip("/") or "/"


def _routes_of(mappings: ClassMappings, rel_path: str) -> list[Route]:
    """Compose class-level and method-level mappings into concrete routes.

    This is the composition step the tool exists for: two facts in two places,
    joined once, in code, instead of by a model reading grep output.
    """
    return [
        Route(http_method=verb, path=_join(base, leaf),
              controller=mappings.class_name, method=method.method,
              file=rel_path, line=method.line,
              parameter_types=method.parameter_types,
              return_type=method.return_type, annotation=method.annotation)
        for method in mappings.methods
        for base in mappings.base_paths
        for leaf in (method.paths or [""])
        for verb in method.verbs
    ]


# ────────────────────────────  the table  ────────────────────────────────


def _candidate_files(root: SourceRoot) -> list[Path]:
    """Java files that mention a mapping annotation at all.

    A text prefilter, not a parse decision: it only decides which files are
    worth handing to tree-sitter, and a false positive costs one parse.
    """
    search_root = root.path / "src" / "main" / "java"
    if not search_root.is_dir():
        search_root = root.path
    out: list[Path] = []
    for path in sorted(search_root.rglob("*.java")):
        try:
            if "Mapping" in path.read_text(encoding="utf-8", errors="replace"):
                out.append(path)
        except OSError:
            continue
    return out


def build_route_table(root: SourceRoot) -> RouteTable:
    java = parser()
    table = RouteTable()
    for path in _candidate_files(root):
        try:
            source = path.read_bytes()
        except OSError:
            continue
        tree = java.parse(source)
        rel = root.relative(path)
        table.files_scanned += 1
        for class_node in class_declarations(tree.root_node):
            table.routes.extend(_routes_of(class_mappings(source, class_node), rel))
    table.routes.sort(key=lambda r: (r.path, r.http_method))
    return table


# Per-repo cache, invalidated on the mtime of any candidate file. Parsing a
# real controller tree on every call would make the tool the slow step in a
# loop whose whole point is to be the fast one.
_CACHE: dict[Path, tuple[tuple[tuple[str, int], ...], RouteTable]] = {}


def _fingerprint(root: SourceRoot) -> tuple[tuple[tuple[str, int], ...], list[Path]]:
    files = _candidate_files(root)
    stamps = tuple((str(p), p.stat().st_mtime_ns) for p in files)
    return stamps, files


def route_table(root: SourceRoot, *, use_cache: bool = True) -> RouteTable:
    if not use_cache:
        return build_route_table(root)
    stamps, _ = _fingerprint(root)
    cached = _CACHE.get(root.path)
    if cached is not None and cached[0] == stamps:
        return cached[1]
    table = build_route_table(root)
    _CACHE[root.path] = (stamps, table)
    return table


def clear_cache() -> None:
    _CACHE.clear()


# ────────────────────────────  matching  ─────────────────────────────────


def _normalise(path: str) -> str:
    cleaned = re.sub(r"/+", "/", "/" + (path or "").strip())
    return cleaned.rstrip("/") or "/"


def _template_regex(template: str) -> re.Pattern[str]:
    """`/api/customers/{id}` → a regex matching `/api/customers/2`.

    `{id:[0-9]+}` (Spring's regex form) keeps only the variable name for
    matching purposes; narrowing it would reject concrete paths that Spring
    itself accepts.
    """
    parts = re.split(r"(\{[^/]*\})", template)
    out = "".join(r"[^/]+" if p.startswith("{") else re.escape(p) for p in parts)
    return re.compile(f"^{out}$")


def _method_matches(route: Route, wanted: str) -> bool:
    return route.http_method in (wanted, ANY_METHOD)


def find_matches(table: RouteTable, method: str,
                 path: str) -> tuple[list[Route], str]:
    """Matching routes and how they were matched.

    A request may be given as the template (`/api/customers/{id}`) or as a
    concrete path (`/api/customers/2`). Both are legitimate; which one was
    used is reported rather than assumed, because a concrete match means the
    tool inferred a path variable and the model should know that.
    """
    wanted_method = (method or "").strip().upper() or ANY_METHOD
    wanted_path = _normalise(path)

    exact = [r for r in table.routes
             if _normalise(r.path) == wanted_path and _method_matches(r, wanted_method)]
    if exact:
        return exact, "exact"

    concrete = [r for r in table.routes
                if "{" in r.path
                and _template_regex(_normalise(r.path)).match(wanted_path)
                and _method_matches(r, wanted_method)]
    if concrete:
        return concrete, "concrete"

    return [], "none"


def _no_match_message(table: RouteTable, method: str, path: str) -> str:
    wanted = f"{method.upper()} {_normalise(path)}"
    labels = table.labels()
    close = get_close_matches(wanted, labels, n=5, cutoff=0.4)
    if not close:
        close = labels[:MAX_CANDIDATES_SHOWN]
    listing = "\n".join(f"  {label}" for label in close) or "  (none found)"
    return (
        f"No route matches {wanted}.\n"
        f"{len(labels)} route(s) were found across {table.files_scanned} "
        f"controller file(s). Closest:\n{listing}\n"
        f"Check the verb and the path, then call find_route again."
    )


def make_find_route(root: SourceRoot) -> Callable[..., str]:
    def find_route(method: str, path: str) -> str:
        table = route_table(root)
        if not table.routes:
            raise ToolError(
                f"no Spring routes were found under {root.path}. Either this "
                f"is not a Spring MVC repository or the controllers live "
                f"outside src/main/java.",
                suggestion="Use search_code for '@RestController' to check.",
            )

        matches, how = find_matches(table, method, path)
        if not matches:
            raise ToolError(_no_match_message(table, method, path))

        header = (
            f"Matched the template exactly."
            if how == "exact" else
            f"'{_normalise(path)}' is a CONCRETE path; it was matched against "
            f"the template below, so the trailing segment is a path variable."
        )
        if len(matches) == 1:
            return f"{header}\n\n{matches[0].render()}"

        rendered = "\n\n".join(r.render() for r in matches[:MAX_CANDIDATES_SHOWN])
        return (
            f"{header}\n\nAMBIGUOUS: {len(matches)} routes match. They are all "
            f"listed below — decide which one you mean and cite that "
            f"controller; do not assume the first.\n\n{rendered}"
        )

    return find_route


FIND_ROUTE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "method": {"type": "string",
                   "description": "HTTP verb, e.g. GET or POST."},
        "path": {"type": "string",
                 "description": "Request path. Either the template "
                                "('/api/customers/{id}') or a concrete path "
                                "('/api/customers/2')."},
    },
    "required": ["method", "path"],
}


def route_tools(root: SourceRoot) -> list[ToolSpec]:
    return [ToolSpec(
        name="find_route",
        description=(
            "START HERE. Resolve an HTTP method and path to the exact "
            "controller method that handles it. Class-level @RequestMapping "
            "and method-level @GetMapping/@PostMapping/... are already "
            "composed for you. Returns the controller class, the method, "
            "file:line, parameter types and the return type."
        ),
        parameters=FIND_ROUTE_PARAMETERS,
        handler=make_find_route(root),
    )]
