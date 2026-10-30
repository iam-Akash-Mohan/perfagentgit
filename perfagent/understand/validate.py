"""Citation validation: every claim is checked against the filesystem.

A model that cannot be caught inventing a line number will invent line
numbers. This module is what makes the comprehension artifact trustworthy —
not the prompt asking nicely, and not the model's confidence.

The rules are deliberately mechanical and the failures are deliberately
verbose: the message goes straight back to the agent as a tool result, and one
retry should be able to fix everything at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..contracts import Citation, EndpointModel
from ..tools import SourceRoot

# Java puts annotations above a method and wraps long signatures across lines,
# so a strict single-line check rejects citations that are correct in every
# way that matters. Two lines either side is the smallest window that covers
# @Transactional / @Query / @Override above a declaration.
SYMBOL_WINDOW = 2

MAX_NEARBY_REPORTED = 3


@dataclass(frozen=True)
class CitationProblem:
    """One rejected citation, and enough detail to fix it in one retry."""

    where: str                      # "call_chain[2].line"
    file: str
    line: int
    symbol: str | None
    problem: str
    found_instead: str | None = None
    nearest_line: int | None = None

    def render(self) -> str:
        out = f"  {self.where}: {self.file}:{self.line} — {self.problem}"
        if self.found_instead is not None:
            out += f"\n      found instead: {self.found_instead}"
        if self.nearest_line is not None:
            out += (f"\n      nearest line containing '{self.symbol}': "
                    f"{self.nearest_line}")
        return out


def labelled_citations(model: EndpointModel) -> list[tuple[str, Citation]]:
    """Every citation the agent actually supplied, with where it came from.

    `entry`, `path` and `sql_sources` are derived from `call_chain` and
    `data_access`, so validating them too would report the same mistake three
    times under names the model never typed.
    """
    out: list[tuple[str, Citation]] = []
    for i, hop in enumerate(model.call_chain):
        out.append((f"call_chain[{i}] ({hop.method})", hop.citation()))
    for i, access in enumerate(model.data_access):
        out.append((f"data_access[{i}] ({access.trigger})", access.citation()))
    for i, entity in enumerate(model.entities):
        out.append((f"entities[{i}] ({entity.name})", entity.citation))
    for i, transaction in enumerate(model.transactions):
        out.append((f"transactions[{i}] ({transaction.method})",
                    transaction.citation))
    for i, call in enumerate(model.external_calls):
        out.append((f"external_calls[{i}] ({call.method})", call.citation))
    for i, work in enumerate(model.per_row_work):
        out.append((f"per_row_work[{i}] ({work.kind})", work.citation()))
    for i, cache in enumerate(model.caching):
        out.append((f"caching[{i}] ({cache.on})", cache.citation))
    for i, trigger in enumerate(model.hidden_query_triggers):
        out.append((f"hidden_query_triggers[{i}] ({trigger.on})",
                    trigger.citation))
    if model.response_shape is not None and model.response_shape.citation:
        out.append(("response_shape", model.response_shape.citation))
    if model.pagination is not None and model.pagination.citation:
        out.append(("pagination", model.pagination.citation))
    if not model.call_chain:
        out.append(("entry", model.entry))
        out.extend((f"path[{i}]", c) for i, c in enumerate(model.path))
    return out


def _token(symbol: str) -> str:
    """The part of a symbol that can appear in source.

    `CustomerService.search` never appears verbatim on the declaration of
    `search`; the tail does. Matching the tail is what makes the check useful
    rather than a source of false rejections.
    """
    return symbol.rsplit(".", 1)[-1].strip("()")


@lru_cache(maxsize=256)
def _lines(path: Path) -> tuple[str, ...]:
    return tuple(path.read_text(encoding="utf-8", errors="replace").splitlines())


def validate_citations(model: EndpointModel,
                       root: SourceRoot) -> list[CitationProblem]:
    """Check every citation. Returns EVERY problem, not the first one."""
    problems: list[CitationProblem] = []

    for where, citation in labelled_citations(model):
        try:
            target = root.resolve(citation.file)
        except Exception as exc:  # noqa: BLE001 — containment refusal is a problem, not a crash
            problems.append(CitationProblem(
                where=where, file=citation.file, line=citation.line,
                symbol=citation.symbol,
                problem=f"path was refused: {exc}"))
            continue

        if not target.exists() or not target.is_file():
            problems.append(CitationProblem(
                where=where, file=citation.file, line=citation.line,
                symbol=citation.symbol,
                problem="this file does not exist in the repository",
                found_instead=_nearby_files(root, citation.file)))
            continue

        lines = _lines(target)
        if citation.line < 1 or citation.line > len(lines):
            problems.append(CitationProblem(
                where=where, file=citation.file, line=citation.line,
                symbol=citation.symbol,
                problem=f"line {citation.line} is outside the file, which has "
                        f"{len(lines)} lines"))
            continue

        if not citation.symbol:
            continue

        token = _token(citation.symbol)
        low = max(citation.line - 1 - SYMBOL_WINDOW, 0)
        high = min(citation.line + SYMBOL_WINDOW, len(lines))
        window = lines[low:high]
        if any(token in line for line in window):
            continue

        matches = [i + 1 for i, line in enumerate(lines) if token in line]
        nearest = (min(matches, key=lambda n: abs(n - citation.line))
                   if matches else None)
        problems.append(CitationProblem(
            where=where, file=citation.file, line=citation.line,
            symbol=citation.symbol,
            problem=f"'{token}' does not appear within {SYMBOL_WINDOW} lines "
                    f"of line {citation.line}",
            found_instead=lines[citation.line - 1].strip()[:160] or "(blank line)",
            nearest_line=nearest))

    return problems


# How many files to consider when suggesting a spelling. A monorepo has
# hundreds of thousands; the suggestion is a nicety and must not become the
# slow part of a rejection.
MAX_FILES_SCANNED = 5000


def _nearby_files(root: SourceRoot, wanted: str) -> str | None:
    """A spelling suggestion for a file that does not exist.

    Matched on the BASENAME first: the model usually gets the directory right
    and the class name slightly wrong, and comparing whole paths buries that
    signal under twenty identical path segments.
    """
    from difflib import get_close_matches

    suffix = Path(wanted).suffix or ".java"
    paths: list[Path] = []
    for path in root.path.rglob(f"*{suffix}"):
        paths.append(path)
        if len(paths) >= MAX_FILES_SCANNED:
            break
    if not paths:
        return None

    by_name: dict[str, str] = {}
    for path in paths:
        by_name.setdefault(path.name, root.relative(path))

    close = get_close_matches(Path(wanted).name, list(by_name),
                              n=MAX_NEARBY_REPORTED, cutoff=0.5)
    if not close:
        return None
    return ", ".join(by_name[name] for name in close)


def validate_claims(model: EndpointModel) -> list[str]:
    """Semantic rules that no citation check can catch. These REJECT.

    Only one rule so far, and it earns its place: `bounded=True` with no
    `bound` is not a claim, it is a shrug. "Something limits this, I decline to
    say what" is exactly the sentence a later phase would read as permission to
    ignore the work.
    """
    problems: list[str] = []
    for i, work in enumerate(model.per_row_work):
        if work.bounded and not (work.bound or "").strip():
            problems.append(
                f"  per_row_work[{i}] ({work.description[:60]}) says "
                f"bounded=true but gives no `bound`. Name the limit — a page "
                f"size, a hard cap, a fixed-size collection — or set "
                f"bounded=false."
            )
    return problems


def completeness_warnings(model: EndpointModel) -> list[str]:
    """Suspicions worth recording. These NEVER reject.

    This is a HEURISTIC, so it must not fail a submission: a loop that makes a
    query per row almost always does other work per row too, but "almost
    always" is not "always", and an agent that is right about an unusual
    endpoint must not be blocked by a rule of thumb. The suspicion is written
    into `uncertainties`, where a human reading the artifact will see it.
    """
    warnings: list[str] = []

    shape = model.response_shape
    if shape is not None and shape.fields_returned:
        stray = [name for name in shape.unbounded_collections
                 if name not in shape.fields_returned]
        if stray:
            warnings.append(
                f"AUTOMATED CHECK: response_shape.unbounded_collections names "
                f"{', '.join(repr(s) for s in stray)}, which do not appear in "
                f"fields_returned. A later phase matches these against the "
                f"response body, so a description or a renamed field will not "
                f"match anything."
            )

    unmeasured = [t.table for t in model.tables
                  if not t.all_columns and t.approx_rows is None]
    if unmeasured:
        warnings.append(
            f"AUTOMATED CHECK: {', '.join(unmeasured)} carry no measured "
            f"facts, so get_schema was never called for them. Their column "
            f"types, row counts and EXISTING INDEXES are unknown — not "
            f"absent. A later phase must not read an empty index list here as "
            f"'no index exists'."
        )

    per_row_queries = model.per_row_accesses()
    if per_row_queries and not model.per_row_work:
        triggers = ", ".join(a.trigger for a in per_row_queries[:3])
        warnings.append(
            f"AUTOMATED CHECK: {len(per_row_queries)} data access(es) run per "
            f"result row ({triggers}) but per_row_work is empty. A loop that "
            f"queries per row almost always does other work per row too, so "
            f"this may be an incomplete map rather than an endpoint with no "
            f"per-row work."
        )
    return warnings


def render_problems(problems: list[CitationProblem]) -> str:
    """The rejection message. Every problem at once, so one retry can fix all."""
    lines = [
        f"{len(problems)} citation(s) could not be verified against the "
        f"repository. ALL of them are listed — fix every one, then call "
        f"submit_endpoint_analysis again:",
    ]
    lines.extend(problem.render() for problem in problems)
    lines.append(
        "Re-read the file with read_file if you are unsure of a line number; "
        "the numbers it prints are the ones checked here."
    )
    return "\n".join(lines)
