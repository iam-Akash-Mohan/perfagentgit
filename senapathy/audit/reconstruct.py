"""Turning a digest back into something EXPLAIN can run.

`performance_schema.events_statements_summary_by_digest` normalises every
literal to `?` — that is what makes digests comparable across calls, and
it is also why a digest alone is not an executable statement. Real values
are needed, and there are exactly two honest sources for them:

1. `performance_schema.events_statements_history` — if the exact
   connection that ran this digest is still tracked and its retention
   has not rolled over, its own `SQL_TEXT` may already show the literal
   values the driver actually sent (true for a driver using client-side
   prepared-statement emulation, which is the common default; false for
   one using true server-side prepared statements, where SQL_TEXT still
   shows `?`). This is the BEST source when it works, because the values
   are the ones a real request actually used — but it is inherently
   timing-dependent: the arm that produced them is normally gone by the
   time this phase runs (see `audit/run.py`), so in practice this often
   finds nothing and the fallback below is what actually resolves most
   statements.
2. The existing sampler (`tools/sample.py::ValueSampler`) — real column
   values pulled directly from the table the placeholder's own predicate
   names, found by parsing the digest text itself. Always available,
   independent of any live connection's history.

PROJ-01's own failure is the one this module is built to never repeat: a
submitted `sql_params` entry that was the literal NAME of a parameter
("pageSize", "offset") rather than a value. Every value this module hands
back comes from the sampler or from a live connection's own SQL_TEXT —
never from parsing an identifier out of the statement and treating the
identifier itself as the value — and `_looks_like_an_identifier_not_a_value`
is a defensive check against exactly that shape of mistake, even though
the two paths above cannot produce it by construction.

A THIRD source applies specifically to a statement's own LIMIT/OFFSET: the
run's own corpus (`corpus.json`) records the real page/size every benchmark
request actually used. Page depth changes the plan — `LIMIT 0, 50` and
`LIMIT 5000, 50` examine very different numbers of rows on an indexed scan —
so a fixed synthetic pair can report a rows-examined figure the endpoint
never produced. `_resolve_limit_offset` prefers a real corpus value over the
synthetic fallback whenever one is available, and every caller downstream
(`audit/run.py`) records WHICH of the three sources supplied it
(`AuditedStatement.limit_offset_source`) rather than presenting a corpus
value, a history value and a guess as the same kind of number.
"""

from __future__ import annotations

import re
from typing import Any

from ..contracts import RequestCorpus
from ..measure.sql import MySQLTarget, connect
from ..plan.fixability import WRAPPING_FUNCTION_NAMES
from ..tools.db import bind_params

# `col = ?`, `alias.col = ?`, `col IN (?`, `alias.col >= ?`, and the SAME
# shapes with the column wrapped in one of the functions fixability.py's own
# non_sargable_marker already flags (`DATE(cp.paid_on) >= ?`) -- confirmed
# live against COL-06's real query, which is exactly this shape: a date
# range filter is the common case for a non-sargable predicate, not a rare
# one, and the value worth sampling is the column's real value regardless
# of the function wrapped around it at comparison time. Captures the
# table/alias prefix (optional) and the column immediately before a
# comparison operator that ends in the placeholder we are resolving.
_COL_BEFORE_PLACEHOLDER = re.compile(
    r"(?:\b(?:" + "|".join(WRAPPING_FUNCTION_NAMES) + r")\s*\(\s*)?"
    r"(?:`?(?P<table>\w+)`?\s*\.\s*)?`?(?P<col>\w+)`?\s*\)?\s*"
    r"(?:=|>=|<=|<>|!=|>|<|\bIN\b)\s*\(?\s*\?",
    re.I,
)

# A bare word immediately after a table reference, with no comma or AS
# before it, is ambiguous between "an alias" and "the next clause's own
# keyword" -- `FROM invoice WHERE ...` and `FROM invoice i WHERE ...` are
# both legal, and only the SECOND one's "i" is really an alias. Excluding
# these from the alias group is what keeps `table_reference_end` from
# splicing a hint after "invoice where" instead of after "invoice".
_NOT_AN_ALIAS = (
    "where", "order", "group", "having", "limit", "join", "on", "union",
    "inner", "left", "right", "outer", "using", "for",
)
_NOT_AN_ALIAS_LOOKAHEAD = "(?!(?:" + "|".join(_NOT_AN_ALIAS) + r")\b)"

# `FROM table alias`, `FROM table AS alias`, `JOIN table alias` -- the digest
# text strips backticks inconsistently depending on MySQL version, so both
# quoted and bare identifiers are matched.
_FROM_OR_JOIN = re.compile(
    r"\b(?:FROM|JOIN)\s+`?(?P<table>\w+)`?"
    r"(?:\s+(?:AS\s+)?`?" + _NOT_AN_ALIAS_LOOKAHEAD + r"(?P<alias>\w+)`?)?",
    re.I,
)

# Old-style comma-separated FROM list (`FROM a x, b y, c z` rather than
# explicit JOINs) -- confirmed live against COL-06's real query, which
# Hibernate rendered exactly this way. `_FROM_OR_JOIN` only ever captures
# the FIRST table (the one immediately after the FROM/JOIN keyword); every
# table after it is separated by a comma, not another keyword. Matched only
# within the FROM clause's own span (see `table_alias_map`) so a comma
# inside the SELECT list or inside a WHERE predicate's `IN (...)` list is
# never mistaken for another table.
_COMMA_TABLE = re.compile(
    r",\s*`?(?P<table>\w+)`?"
    r"(?:\s+(?:AS\s+)?`?" + _NOT_AN_ALIAS_LOOKAHEAD + r"(?P<alias>\w+)`?)?",
    re.I,
)
_FROM_CLAUSE_END = re.compile(
    r"\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|\bHAVING\b", re.I,
)
_FROM_KEYWORD = re.compile(r"\bFROM\b", re.I)

# LIMIT ?  /  LIMIT ?, ?  /  LIMIT ? OFFSET ? -- these placeholders are
# always a bare integer, never a column value, and are safe to synthesise
# (bind_params already converts a digit string to int).
_LIMIT_OFFSET = re.compile(
    r"\bLIMIT\s+\?(?:\s*,\s*\?|\s+OFFSET\s+\?)?", re.I,
)

# MySQL's OWN digest computation has an internal token-buffer ceiling
# (performance_schema_max_digest_length) independent of the printed text's
# length -- a statement joining several aliased tables with several WHERE
# predicates and a multi-column ORDER BY routinely exceeds it before
# reaching its own LIMIT clause, which MySQL then renders as a bare "LIMIT
# ?, ...". Confirmed live, not theoretical: 9 of this project's own 10
# frozen eval scenarios hit exactly this on their endpoint's real paginated
# query (`cat eval/scenarios/*/baseline.json | grep 'LIMIT ?, \.\.\.'`) --
# this is the common case for a paginated multi-table query, not a corner
# case. Anchored at the END of the text on purpose: "..." means MySQL
# stopped computing the digest entirely at that point, so anything that
# could still follow is genuinely unknown and must still refuse; nothing
# ever follows it in practice because LIMIT is the last clause of a
# pagination query, but the anchor is the difference between relying on
# that and asserting it.
_TRUNCATED_LIMIT = re.compile(r"\bLIMIT\s+\?\s*,\s*\.\.\.\s*$", re.I)

_PLACEHOLDER_COUNT = re.compile(r"\?")

# A synthesised LIMIT/OFFSET is a fixed, small, always-safe pair -- large
# enough to be a realistic page, small enough that EXPLAIN never pays for a
# pathological scan just to analyse one statement. Used only when no corpus
# request supplies real values (`_resolve_limit_offset`'s own fallback).
_SYNTHETIC_LIMIT = "50"
_SYNTHETIC_OFFSET = "0"


class UnresolvableStatement(Exception):
    """Why this digest could not be turned into something runnable. Never
    raised to end a run — `resolve_statement` catches this and returns the
    reason as data, because an unanalysable statement is a finding, not a
    failure (see the module docstring and `audit/run.py`)."""


def table_alias_map(digest_text: str) -> dict[str, str]:
    """{alias_or_bare_table_name: real_table_name}, so both `cp.id` and a
    bare `id` (single-table query, no alias) resolve the same way."""
    mapping: dict[str, str] = {}
    for match in _FROM_OR_JOIN.finditer(digest_text):
        table = match.group("table")
        alias = match.group("alias")
        mapping[table.lower()] = table
        if alias:
            mapping[alias.lower()] = table

    from_match = _FROM_KEYWORD.search(digest_text)
    if from_match:
        end_match = _FROM_CLAUSE_END.search(digest_text, from_match.end())
        span_end = end_match.start() if end_match else len(digest_text)
        for match in _COMMA_TABLE.finditer(digest_text, from_match.end(), span_end):
            table = match.group("table")
            alias = match.group("alias")
            mapping.setdefault(table.lower(), table)
            if alias:
                mapping.setdefault(alias.lower(), table)
    return mapping


def table_reference_end(sql: str, table: str) -> int | None:
    """The character offset immediately after `table`'s OWN reference in
    `sql`'s FROM clause — where `candidates.py` splices an `IGNORE INDEX
    (...)` hint to force the statement off a just-created temp index
    without touching the schema again. None when the table cannot be
    found as a FROM/JOIN/comma-list reference at all.

    Returns the end of the table's OWN token (plus its alias, if any) —
    never an earlier alias-only match — so the hint lands in the one
    syntactic position MySQL accepts it: right after the table reference
    it modifies, before any join condition or the next comma.
    """
    wanted = table.lower()
    for match in _FROM_OR_JOIN.finditer(sql):
        if match.group("table").lower() == wanted:
            return match.end()

    from_match = _FROM_KEYWORD.search(sql)
    if from_match:
        end_match = _FROM_CLAUSE_END.search(sql, from_match.end())
        span_end = end_match.start() if end_match else len(sql)
        for match in _COMMA_TABLE.finditer(sql, from_match.end(), span_end):
            if match.group("table").lower() == wanted:
                return match.end()
    return None


def extract_tables(digest_text: str) -> list[str]:
    """Every real table name this statement's own FROM/JOIN clauses name,
    aliases resolved — the tokens `diagnose/validate.py`'s SQL-audit
    reconciliation checks a hypothesis's own text for, to decide whether a
    NON_SARGABLE/MISSING_INDEX finding was addressed at all rather than left
    unmentioned (COL-04's shape: a full scan sat in the evidence and nothing
    in the diagnosis named the table it hit)."""
    return sorted(set(table_alias_map(digest_text).values()))


def extract_equality_columns(digest_text: str) -> dict[str, list[str]]:
    """{table: [column, ...]}, in the order each first appears in a `col =
    ?`/`col IN (?)`-style predicate — the natural column ORDER for a
    composite index candidate (equality columns lead a range or sort
    column, never the reverse). Used by `audit/candidates.py` to propose a
    MISSING_INDEX candidate; shares the same parsing `resolve_statement`
    uses, so a column this audit cannot resolve a VALUE for is still
    available here for an index proposal that needs only the column name.
    """
    aliases = table_alias_map(digest_text)
    out: dict[str, list[str]] = {}
    for match in _COL_BEFORE_PLACEHOLDER.finditer(digest_text):
        table_or_alias = (match.group("table") or "").lower()
        column = match.group("col")
        table = aliases.get(table_or_alias) if table_or_alias else (
            next(iter(set(aliases.values())), None)
            if len(set(aliases.values())) == 1 else None
        )
        if table is None:
            continue
        columns = out.setdefault(table, [])
        if column not in columns:
            columns.append(column)
    return out


def _looks_like_an_identifier_not_a_value(value: str, column: str) -> bool:
    """PROJ-01's shape, defensively refused even though neither path above
    can actually produce it: a 'value' that is really the column's own
    name, or an obviously-unbound placeholder, rather than real data."""
    stripped = value.strip()
    if not stripped:
        return True
    if stripped.lower() == column.lower():
        return True
    if stripped in ("?", "%s"):
        return True
    return False


def _corpus_page_size(path: str) -> tuple[str, str] | None:
    """(offset, limit) as decimal strings, parsed from ONE corpus request's
    own path -- Spring Data's `page`/`size` convention (offset = page *
    size, confirmed against every one of this project's own frozen
    scenarios' real corpus.json) or a bare `offset`/`limit` pair. None when
    neither is present (an unpaginated request, or one this convention does
    not cover)."""
    from urllib.parse import parse_qs, urlsplit

    query = parse_qs(urlsplit(path).query)

    def _one(name: str) -> int | None:
        values = query.get(name)
        if not values:
            return None
        try:
            return int(values[0])
        except ValueError:
            return None

    page, size = _one("page"), _one("size")
    if page is not None and size is not None:
        return str(page * size), str(size)
    offset, limit = _one("offset"), _one("limit")
    if offset is not None and limit is not None:
        return str(offset), str(limit)
    return None


def _corpus_limit_offset(
    corpus: RequestCorpus | None,
) -> tuple[str, str, str] | None:
    """(request_id, offset, limit) from the FIRST benchmark request whose
    own path carries real page/size -- `held_back` is excluded on purpose,
    the same split that keeps it out of every other measured number (see
    `RequestCorpus.held_back`'s own docstring). None when the corpus has no
    such request at all (no corpus, or an endpoint that does not paginate)."""
    if corpus is None:
        return None
    for request in corpus.benchmark:
        parsed = _corpus_page_size(request.path)
        if parsed is not None:
            return request.id, parsed[0], parsed[1]
    return None


def _inline_literals(executable: str, params: list[str]) -> str:
    """`executable`'s own `%s` placeholders, filled in with `params` as SQL
    literals -- for a `STATEMENT_DIGEST()` confirmation call only, never
    for execution (the real run always goes through a parameterised
    `cursor.execute`, never this string). Uses `bind_params`' own
    int-vs-string distinction so a LIMIT/OFFSET value renders bare the same
    way the real bound parameter would."""
    out = executable
    for value in bind_params(params):
        literal = (str(value) if isinstance(value, int) else
                  "'" + str(value).replace("'", "''") + "'")
        out = out.replace("%s", literal, 1)
    return out


def _digest_of(target: MySQLTarget, sql: str) -> str | None:
    """MySQL's own `STATEMENT_DIGEST()` of a complete SQL string, computed
    without executing it -- safe to call on a string built from made-up
    bound values. None on any connection or parse failure; the caller
    degrades to unconfirmed rather than refusing on a failure that is this
    function's own, not the reconstruction's."""
    from mysql.connector import Error as MySQLError

    try:
        with connect(target) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(f"USE `{target.database}`")
                cursor.execute("SELECT STATEMENT_DIGEST(%s)", (sql,))
                row = cursor.fetchone()
            finally:
                cursor.close()
    except MySQLError:
        return None
    return str(row[0]) if row and row[0] else None


def _resolve_limit_offset(
    corpus: RequestCorpus | None,
) -> tuple[str, str, str, str | None]:
    """(offset, limit, source, request_id). Prefers a real benchmark
    request's own page/size over the fixed synthetic pair -- page depth
    changes the plan (`LIMIT 0, 50` and `LIMIT 5000, 50` examine very
    different numbers of rows), so a synthetic value is reported as the
    weaker claim it is (`source == "fallback"`), never presented the same
    as a real one."""
    found = _corpus_limit_offset(corpus)
    if found is not None:
        request_id, offset, limit = found
        return offset, limit, "corpus", request_id
    return _SYNTHETIC_OFFSET, _SYNTHETIC_LIMIT, "fallback", None


def _from_history(target: MySQLTarget, digest: str) -> str | None:
    """A live connection's own SQL_TEXT for this digest, if one is still
    tracked and shows literal values rather than `?`. See the module
    docstring for why this is best-effort, not load-bearing."""
    from mysql.connector import Error as MySQLError

    try:
        with connect(target) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT SQL_TEXT FROM performance_schema.events_statements_history "
                    "WHERE DIGEST = %s AND CURRENT_SCHEMA = %s "
                    "ORDER BY TIMER_START DESC LIMIT 5",
                    (digest, target.database),
                )
                rows = cursor.fetchall()
            finally:
                cursor.close()
    except MySQLError:
        return None

    for (sql_text,) in rows:
        text = str(sql_text or "")
        if text and "?" not in text:
            return text
    return None


def _limit_shape(span: str) -> str:
    """'comma' for `LIMIT ?, ?` (binds (offset, count), in that order),
    'offset_kw' for `LIMIT ? OFFSET ?` (binds (count, offset) -- the
    OPPOSITE order for the same two numbers), 'bare' for `LIMIT ?` (count
    only; offset is implicitly 0 and not a placeholder at all)."""
    if "," in span:
        return "comma"
    if re.search(r"OFFSET", span, re.I):
        return "offset_kw"
    return "bare"


def _from_sampler(
    digest_text: str, sampler: Any, *, target: MySQLTarget, digest: str,
    corpus: RequestCorpus | None,
) -> tuple[str, list[str], str | None, str | None]:
    """(executable_sql with %s placeholders, params, limit_offset_source,
    limit_offset_request_id). Raises UnresolvableStatement, never guesses,
    when any placeholder cannot be confidently mapped, OR when a
    `STATEMENT_DIGEST()` confirmation comes back and DEFINITIVELY
    disagrees with the digest being resolved."""
    limit_offset_source: str | None = None
    limit_offset_request_id: str | None = None

    truncated = _TRUNCATED_LIMIT.search(digest_text)
    if truncated:
        # The LIMIT/OFFSET VALUES were always going to be filled in here,
        # never read off the truncated text itself -- so losing the
        # placeholder count MySQL's own digest computation already
        # discarded loses nothing that was ever going to come FROM the
        # digest. Splice in real corpus values when available (see
        # `_resolve_limit_offset`), the fixed synthetic pair otherwise, as
        # literal text rather than a placeholder: there is no `?` left
        # here to bind against.
        offset, limit, limit_offset_source, limit_offset_request_id = \
            _resolve_limit_offset(corpus)
        digest_text = (digest_text[:truncated.start()]
                       + f"LIMIT {offset}, {limit}")
    elif "..." in digest_text:
        # A truncation marker NOT in the one shape just handled (end-of-text,
        # immediately after a LIMIT clause) means something genuinely unknown
        # was lost, possibly a whole extra clause -- refuse rather than treat
        # the literal dots as inert text, which would otherwise silently
        # reach the connector as part of the executed statement.
        raise UnresolvableStatement(
            "the digest text contains MySQL's own truncation marker "
            "('...') somewhere this module does not know how to recover "
            "from -- only an end-of-statement LIMIT truncation is handled."
        )

    total_placeholders = len(_PLACEHOLDER_COUNT.findall(digest_text))
    aliases = table_alias_map(digest_text)

    # Mark LIMIT/OFFSET placeholders first and remove them from the text
    # the column-matcher sees, so `LIMIT ?` is never mistaken for a
    # column comparison.
    limit_match = _LIMIT_OFFSET.search(digest_text)
    working_text = digest_text
    limit_values: list[str] = []
    if limit_match:
        offset, limit, limit_offset_source, limit_offset_request_id = \
            _resolve_limit_offset(corpus)
        shape = _limit_shape(limit_match.group(0))
        if shape == "bare":
            limit_values = [limit]
        elif shape == "comma":
            limit_values = [offset, limit]          # LIMIT offset, count
        else:
            limit_values = [limit, offset]          # LIMIT count OFFSET offset
        working_text = (digest_text[:limit_match.start()]
                       + digest_text[limit_match.end():])

    column_matches = list(_COL_BEFORE_PLACEHOLDER.finditer(working_text))
    resolved_non_limit = len(column_matches)
    if resolved_non_limit + len(limit_values) != total_placeholders:
        raise UnresolvableStatement(
            f"{total_placeholders} placeholder(s) in the digest text, but "
            f"only {resolved_non_limit + len(limit_values)} could be "
            f"mapped to a column or a LIMIT/OFFSET position -- refusing "
            f"rather than guessing at the rest."
        )

    params: list[str] = []
    for match in column_matches:
        table_or_alias = (match.group("table") or "").lower()
        column = match.group("col")
        table = aliases.get(table_or_alias) if table_or_alias else (
            next(iter(set(aliases.values())), None)
            if len(set(aliases.values())) == 1 else None
        )
        if table is None:
            raise UnresolvableStatement(
                f"placeholder for column {column!r} has no unambiguous "
                f"table -- {len(set(aliases.values()))} table(s) in this "
                f"statement's FROM/JOIN clauses, and no alias prefix to "
                f"disambiguate."
            )
        values = sampler.sample(table, column, limit=1)
        if not values:
            raise UnresolvableStatement(
                f"no real value could be sampled for {table}.{column} -- "
                f"either the table is empty or the column holds only NULLs."
            )
        value = str(values[0])
        if _looks_like_an_identifier_not_a_value(value, column):
            raise UnresolvableStatement(
                f"the value sampled for {table}.{column} ({value!r}) looks "
                f"like an identifier, not real data -- refusing rather "
                f"than binding something that is not a value."
            )
        params.append(value)

    # Re-merge LIMIT/OFFSET params back in at their ORIGINAL position (the
    # order placeholders appear in the final params list must match the
    # order %s appears in the rewritten SQL, not the order they were
    # resolved in).
    executable = digest_text.replace("?", "%s")
    if limit_match:
        ordered_params: list[str] = []
        col_iter = iter(params)
        limit_iter = iter(limit_values)
        for m in re.finditer(r"\?", digest_text):
            if limit_match.start() <= m.start() < limit_match.end():
                ordered_params.append(next(limit_iter))
            else:
                ordered_params.append(next(col_iter))
        params = ordered_params

    # A real benchmark request's page/size is only trustworthy if it is
    # still spliced into a statement whose SHAPE matches what was actually
    # recorded -- STATEMENT_DIGEST() confirms that without executing
    # anything. Unreachable or unparsable is a failure of THIS check, not
    # of the reconstruction, so it degrades to unconfirmed rather than
    # refusing; a DEFINITE disagreement is the reconstruction measuring the
    # wrong statement under the right digest's identity, which is refused.
    computed = _digest_of(target, _inline_literals(executable, params))
    if computed is not None and computed != digest:
        raise UnresolvableStatement(
            f"STATEMENT_DIGEST() of the reconstructed statement "
            f"({computed}) does not match the recorded digest ({digest}) "
            f"-- this reconstruction does not match what was actually "
            f"measured, and refusing is safer than analysing the wrong "
            f"statement under the right digest's identity."
        )

    return executable, params, limit_offset_source, limit_offset_request_id


def resolve_statement(
    target: MySQLTarget, digest: str, digest_text: str, sampler: Any,
    *, corpus: RequestCorpus | None = None,
) -> tuple[str | None, list[str], str, str | None, str | None]:
    """(executable_sql, params, note, limit_offset_source,
    limit_offset_request_id). executable_sql is None, with note explaining
    why, when this digest could not be reconstructed at all -- recorded as
    UNANALYSABLE, never skipped.

    `executable_sql` uses `%s` placeholders with a separate `params` list
    when resolved via the sampler (the project's existing binding
    convention, see `tools/db.py::bind_params`), or is a complete,
    parameter-free statement (`params` empty) when resolved via a live
    connection's own history -- in which case `limit_offset_source` is
    `"digest"` whenever the recovered text has a LIMIT clause at all: every
    value in it, LIMIT included, is the real one a real connection sent,
    nothing spliced in.

    `corpus` is this run's own `corpus.json` (`RequestCorpus`), read for
    real page/size values; optional so a caller without one (an older run,
    a hand-built test) still gets a resolvable statement, just with
    `limit_offset_source == "fallback"` for anything paginated.
    """
    historical = _from_history(target, digest)
    if historical is not None:
        limit_offset_source = "digest" if re.search(
            r"\bLIMIT\b", historical, re.I) else None
        return (historical, [], "resolved from a live connection's own SQL_TEXT",
                limit_offset_source, None)

    try:
        executable, params, limit_offset_source, limit_offset_request_id = \
            _from_sampler(digest_text, sampler, target=target, digest=digest,
                         corpus=corpus)
    except UnresolvableStatement as exc:
        return None, [], str(exc), None, None
    return (executable, params, "resolved by sampling real column values",
            limit_offset_source, limit_offset_request_id)
