"""Reading Spring mappings out of Java source, with a real parser.

Split out of `routes.py` so the route table is about routing and this file is
about Java. Regex is not an option here: `@GetMapping` inside a comment,
inside a string literal, or on a nested class are all things a regex gets
wrong and a parser does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator

from ..errors import ToolError

HTTP_ANNOTATIONS = {
    "GetMapping": "GET",
    "PostMapping": "POST",
    "PutMapping": "PUT",
    "DeleteMapping": "DELETE",
    "PatchMapping": "PATCH",
}

# @RequestMapping with no method= matches every verb. Saying ANY out loud is
# better than silently picking GET.
ANY_METHOD = "ANY"

PATH_KEYS = ("value", "path")


# ─────────────────────────────  parsing  ─────────────────────────────────


def parser() -> Any:
    try:
        import tree_sitter_java
        from tree_sitter import Language, Parser
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ToolError(
            "tree-sitter-java is not installed, so find_route cannot build a "
            "route table.",
            suggestion="pip install tree-sitter tree-sitter-java, or use "
                       "search_code for '@PostMapping' instead.",
        ) from exc
    return Parser(Language(tree_sitter_java.language()))


def _text(source: bytes, node: Any) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _children(node: Any, kind: str) -> Iterator[Any]:
    for child in node.children:
        if child.type == kind:
            yield child


def _annotations(node: Any) -> Iterator[Any]:
    """Annotations attached to a class or method declaration."""
    for modifiers in _children(node, "modifiers"):
        for child in modifiers.children:
            if child.type in ("annotation", "marker_annotation"):
                yield child


def _annotation_name(source: bytes, node: Any) -> str:
    name = node.child_by_field_name("name")
    return _text(source, name) if name is not None else ""


def _string_literals(source: bytes, node: Any) -> list[str]:
    """Every string literal in a subtree, in order.

    Covers the bare form `@GetMapping("/x")`, the keyed form
    `@GetMapping(path = "/x")` and the array form `@GetMapping({"/a", "/b"})`
    with one traversal, because all three put the paths in string literals and
    the caller has already narrowed to the right argument node.
    """
    out: list[str] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "string_literal":
            raw = _text(source, current)
            out.append(raw[1:-1] if len(raw) >= 2 else raw)
            continue
        stack.extend(reversed(current.children))
    return out


def _annotation_paths(source: bytes, node: Any) -> list[str]:
    """The path(s) an annotation declares. Empty list means "no path given"."""
    arguments = node.child_by_field_name("arguments")
    if arguments is None:
        return []

    pairs = list(_children(arguments, "element_value_pair"))
    if not pairs:
        # Bare form: @GetMapping("/x") or @GetMapping({"/a", "/b"}).
        return _string_literals(source, arguments)

    for pair in pairs:
        key = pair.child_by_field_name("key")
        if key is not None and _text(source, key) in PATH_KEYS:
            value = pair.child_by_field_name("value")
            return _string_literals(source, value) if value is not None else []
    return []


def _request_mapping_methods(source: bytes, node: Any) -> list[str]:
    """The verbs a @RequestMapping(method = ...) names, or [] for all of them."""
    arguments = node.child_by_field_name("arguments")
    if arguments is None:
        return []
    for pair in _children(arguments, "element_value_pair"):
        key = pair.child_by_field_name("key")
        if key is None or _text(source, key) != "method":
            continue
        value = pair.child_by_field_name("value")
        if value is None:
            return []
        # RequestMethod.GET, {RequestMethod.GET, RequestMethod.POST}, GET
        return re.findall(r"\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b",
                          _text(source, value))
    return []


def _mapping(source: bytes, annotation: Any) -> tuple[list[str], list[str]] | None:
    """(http methods, paths) for a mapping annotation, or None if it isn't one."""
    name = _annotation_name(source, annotation)
    if name in HTTP_ANNOTATIONS:
        return [HTTP_ANNOTATIONS[name]], _annotation_paths(source, annotation)
    if name == "RequestMapping":
        verbs = _request_mapping_methods(source, annotation) or [ANY_METHOD]
        return verbs, _annotation_paths(source, annotation)
    return None


def _parameter_types(source: bytes, node: Any) -> list[str]:
    parameters = node.child_by_field_name("parameters")
    if parameters is None:
        return []
    out: list[str] = []
    for parameter in parameters.children:
        if parameter.type not in ("formal_parameter", "spread_parameter"):
            continue
        type_node = parameter.child_by_field_name("type")
        name_node = parameter.child_by_field_name("name")
        rendered = _text(source, type_node) if type_node is not None else "?"
        if name_node is not None:
            rendered += f" {_text(source, name_node)}"
        out.append(" ".join(rendered.split()))
    return out


@dataclass(frozen=True)
class ClassMappings:
    """One controller class, as far as routing is concerned."""

    class_name: str
    base_paths: list[str]          # from a class-level @RequestMapping; [""] if none
    methods: list["MethodMapping"]


@dataclass(frozen=True)
class MethodMapping:
    """One handler method and the mapping annotation on it."""

    method: str
    line: int
    parameter_types: list[str]
    return_type: str
    annotation: str                # "@GetMapping"
    verbs: list[str]               # ["GET"], or [ANY_METHOD]
    paths: list[str]               # [] when the annotation names no path


def class_mappings(source: bytes, class_node: Any) -> ClassMappings:
    """Everything routing needs from one class declaration.

    Deliberately stops short of building routes: composing a class base path
    with a method path is a routing decision, and this module only reads Java.
    """
    class_name_node = class_node.child_by_field_name("name")
    class_name = _text(source, class_name_node) if class_name_node else "?"

    bases: list[str] = []
    for annotation in _annotations(class_node):
        if _annotation_name(source, annotation) == "RequestMapping":
            bases = _annotation_paths(source, annotation) or [""]
    if not bases:
        bases = [""]

    body = class_node.child_by_field_name("body")
    methods: list[MethodMapping] = []
    if body is not None:
        for member in body.children:
            if member.type != "method_declaration":
                continue
            name_node = member.child_by_field_name("name")
            type_node = member.child_by_field_name("type")
            line = (name_node.start_point[0] if name_node
                    else member.start_point[0]) + 1
            for annotation in _annotations(member):
                mapping = _mapping(source, annotation)
                if mapping is None:
                    continue
                verbs, paths = mapping
                methods.append(MethodMapping(
                    method=_text(source, name_node) if name_node else "?",
                    line=line,
                    parameter_types=_parameter_types(source, member),
                    return_type=(" ".join(_text(source, type_node).split())
                                 if type_node is not None else "void"),
                    annotation=f"@{_annotation_name(source, annotation)}",
                    verbs=verbs,
                    paths=paths,
                ))

    return ClassMappings(class_name=class_name, base_paths=bases,
                         methods=methods)


def class_declarations(node: Any) -> Iterator[Any]:
    """Every class in the file, nested ones included."""
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in ("class_declaration", "record_declaration"):
            yield current
        stack.extend(current.children)
