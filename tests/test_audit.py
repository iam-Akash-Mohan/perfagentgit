"""The SQL audit sweep: digest reconstruction, classification, candidates,
and the end-to-end run. The whole phase is ordinary code — every test here
is about getting a deterministic answer out of a fixed input, not about a
model's behaviour.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

from senapathy.audit.candidates import (propose_index_candidate,
                                        propose_rewrite_candidate)
from senapathy.audit.classify import classify
from senapathy.audit.reconstruct import (UnresolvableStatement,
                                         extract_equality_columns,
                                         extract_tables, resolve_statement,
                                         table_reference_end)
from senapathy.audit.run import run_sql_audit
from senapathy.contracts import (IndexCandidate, RequestCorpus, RequestSpec,
                                 SqlDigestStat)
from senapathy.measure.sql import MySQLTarget
from senapathy.tools.db import IndexBenefitResult, PlanReading
from tests.test_db import FAST_TREE, SQL, TREE, connected, script, toolbox

# ───────────────────────────────  reconstruct  ────────────────────────────


def _target() -> MySQLTarget:
    return MySQLTarget(host="127.0.0.1", port=3307, user="root",
                       password="root", database="mini")


class FakeSampler:
    """{(table, column): value}. Missing keys sample as empty (no row)."""

    def __init__(self, values: dict[tuple[str, str], Any]) -> None:
        self.values = values

    def sample(self, table: str, column: str, *, limit: int = 1) -> list[Any]:
        key = (table, column)
        return [self.values[key]] if key in self.values else []


class FakeDigestConnection:
    """A `connect()` stand-in answering `events_statements_history` with no
    rows (history always "unavailable", so every test using this reaches
    the sampler path) and `STATEMENT_DIGEST(%s)` with a FIXED digest
    string -- for testing the corpus-matching confirmation step without a
    real MySQL server. Records every string passed to STATEMENT_DIGEST so
    a test can assert on exactly what was confirmed."""

    def __init__(self, digest_to_return: str | None) -> None:
        self.digest_to_return = digest_to_return
        self.statement_digest_calls: list[str] = []

    @contextmanager
    def __call__(self, target: MySQLTarget) -> Iterator[Any]:
        outer = self

        class Cursor:
            def __init__(self) -> None:
                self._rows: list[tuple[Any, ...]] = []

            def execute(self, sql: str, params: Any = ()) -> None:
                if "STATEMENT_DIGEST" in sql:
                    outer.statement_digest_calls.append(params[0])
                    self._rows = ([(outer.digest_to_return,)]
                                 if outer.digest_to_return is not None else [])
                elif "events_statements_history" in sql:
                    self._rows = []
                else:
                    self._rows = []

            def fetchall(self) -> list[tuple[Any, ...]]:
                return self._rows

            def fetchone(self) -> tuple[Any, ...] | None:
                return self._rows[0] if self._rows else None

            def close(self) -> None:
                pass

        class Conn:
            def cursor(self) -> Cursor:
                return Cursor()

        yield Conn()


def _corpus(*requests: tuple[str, str]) -> RequestCorpus:
    """[(request_id, path), ...] -> a RequestCorpus with those as the
    benchmark set."""
    return RequestCorpus(
        benchmark=[RequestSpec(id=rid, method="GET", path=path)
                  for rid, path in requests],
        hash="x",
    )


def test_extract_equality_columns_reads_where_order_per_table() -> None:
    digest_text = ("SELECT * FROM invoice i JOIN customer c ON c.id = i.customer_id "
                   "WHERE i.customer_id = ? AND i.status = ? AND c.region = ?")
    by_table = extract_equality_columns(digest_text)
    assert by_table["invoice"] == ["customer_id", "status"]
    assert by_table["customer"] == ["region"]


def test_extract_equality_columns_handles_a_bare_unaliased_table() -> None:
    by_table = extract_equality_columns(
        "SELECT * FROM invoice WHERE customer_id = ? AND status = ?")
    assert by_table["invoice"] == ["customer_id", "status"]


def test_extract_tables_resolves_an_old_style_comma_separated_from_list() -> None:
    """Live-confirmed against COL-06's real query: Hibernate rendered its
    join as `FROM a x, b y, c z`, not explicit JOINs -- the FROM/JOIN
    regex alone only ever finds the first table (`a x`); the rest need
    their own comma-aware pass, scoped to the FROM clause only."""
    tables = extract_tables(
        "SELECT c1_0.name FROM payment_allocation pa1_0 , customer_payment "
        "cp1_0 , invoice i1_0 , customer c1_0 WHERE pa1_0.payment_id = "
        "cp1_0.id AND cp1_0.method = ?"
    )
    assert tables == ["customer", "customer_payment", "invoice",
                      "payment_allocation"]


def test_extract_tables_does_not_mistake_a_select_list_comma_for_a_table() -> None:
    """The comma-list pass is scoped to the FROM clause's own span -- a
    comma in the SELECT list (before FROM even starts) must never be read
    as another table."""
    tables = extract_tables(
        "SELECT a , b , c FROM invoice WHERE customer_id = ?")
    assert tables == ["invoice"]


def test_extract_tables_does_not_cross_into_an_in_list_after_where() -> None:
    """Nor a comma inside a WHERE predicate's own IN (...) list, which sits
    after the FROM clause's end boundary."""
    tables = extract_tables(
        "SELECT * FROM invoice WHERE status IN (?, ?, ?)")
    assert tables == ["invoice"]


def test_table_reference_end_finds_a_bare_table_with_no_alias() -> None:
    sql = "select * from invoice where customer_id = %s"
    end = table_reference_end(sql, "invoice")
    assert end is not None
    assert sql[:end] == "select * from invoice"
    assert sql[end:].startswith(" where")


def test_table_reference_end_finds_an_aliased_join() -> None:
    sql = "select * from invoice i join customer c on c.id = i.customer_id"
    end = table_reference_end(sql, "invoice")
    assert sql[:end] == "select * from invoice i"


def test_table_reference_end_finds_a_comma_joined_table() -> None:
    sql = ("select * from payment_allocation pa1_0 , customer_payment cp1_0 "
          "where pa1_0.payment_id = cp1_0.id")
    end = table_reference_end(sql, "customer_payment")
    assert sql[:end] == ("select * from payment_allocation pa1_0 , "
                         "customer_payment cp1_0")


def test_table_reference_end_is_none_for_a_table_not_in_the_from_clause() -> None:
    sql = "select * from invoice where customer_id = %s"
    assert table_reference_end(sql, "customer") is None


def test_resolve_statement_prefers_a_live_connections_own_sql_text(
    monkeypatch,
) -> None:
    """When `events_statements_history` still shows literal values (no `?`
    left), that is the best source -- it reflects what a real request
    actually used -- and the sampler is never even consulted."""
    @contextmanager
    def fake_connect(target: MySQLTarget) -> Iterator[Any]:
        class Cursor:
            def execute(self, sql, params=()):
                pass

            def fetchall(self):
                return [("select * from invoice where customer_id = 42",)]

            def close(self):
                pass

        class Conn:
            def cursor(self):
                return Cursor()

        yield Conn()

    monkeypatch.setattr("senapathy.audit.reconstruct.connect", fake_connect)
    sampler = FakeSampler({})  # never touched if history wins
    executable, params, note, limit_source, limit_request_id = resolve_statement(
        _target(), "abc123", "select * from invoice where customer_id = ?", sampler)
    assert executable == "select * from invoice where customer_id = 42"
    assert params == []
    assert "live connection" in note
    assert limit_source is None      # no LIMIT clause in this statement at all


def test_resolve_statement_labels_a_paginated_history_hit_as_digest_sourced(
    monkeypatch,
) -> None:
    """When history recovers the whole statement, LIMIT included, every
    value in it -- LIMIT included -- is the real one a real connection
    sent. Nothing was spliced in, so the source is 'digest', not 'corpus'
    or 'fallback'."""
    @contextmanager
    def fake_connect(target: MySQLTarget) -> Iterator[Any]:
        class Cursor:
            def execute(self, sql, params=()):
                pass

            def fetchall(self):
                return [("select * from invoice where customer_id = 42 "
                        "LIMIT 200, 50",)]

            def close(self):
                pass

        class Conn:
            def cursor(self):
                return Cursor()

        yield Conn()

    monkeypatch.setattr("senapathy.audit.reconstruct.connect", fake_connect)
    sampler = FakeSampler({})
    executable, params, note, limit_source, limit_request_id = resolve_statement(
        _target(), "abc123",
        "select * from invoice where customer_id = ? LIMIT ?, ?", sampler)
    assert limit_source == "digest"
    assert limit_request_id is None


def test_resolve_statement_falls_back_to_the_sampler_when_history_is_unavailable(
    monkeypatch,
) -> None:
    @contextmanager
    def fake_connect(target: MySQLTarget) -> Iterator[Any]:
        raise RuntimeError("should never be reached directly")
        yield  # pragma: no cover

    def raising_connect(target: MySQLTarget):
        from mysql.connector import Error as MySQLError
        raise MySQLError("performance_schema not enabled")

    monkeypatch.setattr("senapathy.audit.reconstruct.connect", raising_connect)
    sampler = FakeSampler({("invoice", "customer_id"): 42})
    executable, params, note, limit_source, limit_request_id = resolve_statement(
        _target(), "abc123", "select * from invoice where customer_id = ?", sampler)
    assert executable == "select * from invoice where customer_id = %s"
    assert params == ["42"]
    assert "sampling" in note


def test_resolve_statement_samples_a_function_wrapped_column_predicate(
    monkeypatch,
) -> None:
    """Live-confirmed against COL-06's real query: `DATE(cp.paid_on) >= ?`
    -- the column-before-placeholder matcher must see past the DATE()
    wrapper to sample a real `paid_on` value, the same value it would
    sample for a bare `paid_on >= ?` predicate."""
    def raising_connect(target: MySQLTarget):
        from mysql.connector import Error as MySQLError
        raise MySQLError("not enabled")

    monkeypatch.setattr("senapathy.audit.reconstruct.connect", raising_connect)
    sampler = FakeSampler({("customer_payment", "paid_on"): "2024-01-01"})
    executable, params, note, limit_source, limit_request_id = resolve_statement(
        _target(), "abc123",
        "SELECT * FROM customer_payment cp WHERE DATE ( cp . paid_on ) >= ?",
        sampler)
    assert executable == (
        "SELECT * FROM customer_payment cp WHERE DATE ( cp . paid_on ) >= %s")
    assert params == ["2024-01-01"]


def test_resolve_statement_resolves_limit_offset_as_synthetic_integers(
    monkeypatch,
) -> None:
    def raising_connect(target: MySQLTarget):
        from mysql.connector import Error as MySQLError
        raise MySQLError("not enabled")

    monkeypatch.setattr("senapathy.audit.reconstruct.connect", raising_connect)
    sampler = FakeSampler({("invoice", "customer_id"): 42})
    executable, params, note, limit_source, limit_request_id = resolve_statement(
        _target(), "abc123",
        "select * from invoice where customer_id = ? LIMIT ? OFFSET ?", sampler)
    assert executable == "select * from invoice where customer_id = %s LIMIT %s OFFSET %s"
    assert params == ["42", "50", "0"]


def test_resolve_statement_binds_the_comma_form_offset_before_count(
    monkeypatch,
) -> None:
    """`LIMIT ?, ?` binds (offset, count) in that order -- the OPPOSITE of
    `LIMIT ? OFFSET ?`'s (count, offset). Binding the fallback pair
    positionally without checking which grammar matched would put the
    synthetic LIMIT value where OFFSET belongs and vice versa."""
    def raising_connect(target: MySQLTarget):
        from mysql.connector import Error as MySQLError
        raise MySQLError("not enabled")

    monkeypatch.setattr("senapathy.audit.reconstruct.connect", raising_connect)
    sampler = FakeSampler({("invoice", "customer_id"): 42})
    executable, params, note, limit_source, limit_request_id = resolve_statement(
        _target(), "abc123",
        "select * from invoice where customer_id = ? LIMIT ?, ?", sampler)
    assert executable == "select * from invoice where customer_id = %s LIMIT %s, %s"
    assert params == ["42", "0", "50"]       # offset=0 first, count=50 second
    assert limit_source == "fallback"


def test_resolve_statement_prefers_a_real_corpus_page_depth_when_digest_confirms(
    monkeypatch,
) -> None:
    """Live-confirmed against COL-06: the SAME digest that a fixed
    synthetic LIMIT would have produced is also produced using the run's
    own corpus page/size -- STATEMENT_DIGEST() ignores literal values
    entirely, so confirming the shape once is enough to trust corpus
    values over an invented pair."""
    digest_text = "select * from invoice where customer_id = ? LIMIT ?, ?"
    fake_target = _target()
    fake_connect = FakeDigestConnection(digest_to_return="abc123")
    monkeypatch.setattr("senapathy.audit.reconstruct.connect", fake_connect)
    sampler = FakeSampler({("invoice", "customer_id"): 42})
    corpus = _corpus(("dense-p2-s100", "/api/invoices?page=2&size=100"))

    executable, params, note, limit_source, limit_request_id = resolve_statement(
        fake_target, "abc123", digest_text, sampler, corpus=corpus)

    assert executable == "select * from invoice where customer_id = %s LIMIT %s, %s"
    assert params == ["42", "200", "100"]    # page 2 * size 100 = offset 200
    assert limit_source == "corpus"
    assert limit_request_id == "dense-p2-s100"
    assert fake_connect.statement_digest_calls    # confirmation was attempted


def test_resolve_statement_falls_back_when_corpus_has_no_pagination_info(
    monkeypatch,
) -> None:
    def raising_connect(target: MySQLTarget):
        from mysql.connector import Error as MySQLError
        raise MySQLError("not enabled")

    monkeypatch.setattr("senapathy.audit.reconstruct.connect", raising_connect)
    sampler = FakeSampler({("invoice", "customer_id"): 42})
    corpus = _corpus(("req-1", "/api/invoices/1"))     # no page/size at all

    executable, params, note, limit_source, limit_request_id = resolve_statement(
        _target(), "abc123",
        "select * from invoice where customer_id = ? LIMIT ?, ?", sampler,
        corpus=corpus)

    assert params == ["42", "0", "50"]
    assert limit_source == "fallback"
    assert limit_request_id is None


def test_resolve_statement_refuses_when_statement_digest_disagrees(
    monkeypatch,
) -> None:
    """A corpus value is only trustworthy if the reconstruction it was
    spliced into still matches what was actually recorded -- a definite
    STATEMENT_DIGEST() disagreement means this reconstruction measures the
    wrong statement under the right digest's identity, which is refused
    rather than silently analysed."""
    fake_connect = FakeDigestConnection(digest_to_return="some-other-digest")
    monkeypatch.setattr("senapathy.audit.reconstruct.connect", fake_connect)
    sampler = FakeSampler({("invoice", "customer_id"): 42})

    executable, params, note, limit_source, limit_request_id = resolve_statement(
        _target(), "abc123",
        "select * from invoice where customer_id = ? LIMIT ?, ?", sampler)

    assert executable is None
    assert "does not match" in note


def test_resolve_statement_proceeds_unconfirmed_when_statement_digest_is_unavailable(
    monkeypatch,
) -> None:
    """STATEMENT_DIGEST() being unreachable is a failure of the
    confirmation step itself, not of the reconstruction -- it must degrade
    to unconfirmed, never refuse a statement over its OWN inability to
    check."""
    fake_connect = FakeDigestConnection(digest_to_return=None)
    monkeypatch.setattr("senapathy.audit.reconstruct.connect", fake_connect)
    sampler = FakeSampler({("invoice", "customer_id"): 42})

    executable, params, note, limit_source, limit_request_id = resolve_statement(
        _target(), "abc123",
        "select * from invoice where customer_id = ? LIMIT ?, ?", sampler)

    assert executable == "select * from invoice where customer_id = %s LIMIT %s, %s"
    assert params == ["42", "0", "50"]


def test_resolve_statement_splices_a_synthetic_limit_over_a_truncated_one(
    monkeypatch,
) -> None:
    """Live-confirmed, not theoretical: 9 of this project's own 10 frozen
    scenarios hit MySQL's own digest truncation on their real paginated
    query's LIMIT clause (COL-06 among them). The LIMIT/OFFSET values were
    ALWAYS going to be synthesised rather than sampled -- losing the exact
    numbers MySQL itself already discarded loses nothing real, so this
    resolves instead of refusing."""
    def raising_connect(target: MySQLTarget):
        from mysql.connector import Error as MySQLError
        raise MySQLError("not enabled")

    monkeypatch.setattr("senapathy.audit.reconstruct.connect", raising_connect)
    sampler = FakeSampler({("invoice", "customer_id"): 42})
    executable, params, note, limit_source, limit_request_id = resolve_statement(
        _target(), "abc123",
        "select * from invoice where customer_id = ? ORDER BY id LIMIT ?, ...",
        sampler)
    assert executable == ("select * from invoice where customer_id = %s "
                          "ORDER BY id LIMIT 0, 50")
    assert params == ["42"]
    assert limit_source == "fallback"


def test_resolve_statement_splices_a_real_corpus_page_depth_over_a_truncated_limit(
    monkeypatch,
) -> None:
    """The exact COL-06 shape, end to end: a truncated LIMIT clause,
    resolved from the run's own corpus rather than a fixed guess."""
    digest_text = "select * from invoice where customer_id = ? ORDER BY id LIMIT ?, ..."
    fake_connect = FakeDigestConnection(digest_to_return="abc123")
    monkeypatch.setattr("senapathy.audit.reconstruct.connect", fake_connect)
    sampler = FakeSampler({("invoice", "customer_id"): 42})
    corpus = _corpus(("dense-p0-s100", "/api/invoices?page=0&size=100"))

    executable, params, note, limit_source, limit_request_id = resolve_statement(
        _target(), "abc123", digest_text, sampler, corpus=corpus)

    assert executable == ("select * from invoice where customer_id = %s "
                          "ORDER BY id LIMIT 0, 100")
    assert limit_source == "corpus"
    assert limit_request_id == "dense-p0-s100"


def test_resolve_statement_still_refuses_when_something_follows_the_truncation(
    monkeypatch,
) -> None:
    """"..." means MySQL stopped computing entirely at that point in
    practice -- but the anchor is checked, not assumed: anything that is
    not at the very end after a truncation marker is genuinely unknown and
    must still refuse rather than silently dropping it."""
    def raising_connect(target: MySQLTarget):
        from mysql.connector import Error as MySQLError
        raise MySQLError("not enabled")

    monkeypatch.setattr("senapathy.audit.reconstruct.connect", raising_connect)
    sampler = FakeSampler({})
    executable, params, note, limit_source, limit_request_id = resolve_statement(
        _target(), "abc123",
        "select * from invoice LIMIT ?, ... FOR UPDATE", sampler)
    assert executable is None


def test_resolve_statement_refuses_an_ambiguous_unaliased_join(monkeypatch) -> None:
    """Two tables, no alias prefix on the predicate -- PROJ-01's shape of
    mistake would be to guess; this refuses by name instead."""
    def raising_connect(target: MySQLTarget):
        from mysql.connector import Error as MySQLError
        raise MySQLError("not enabled")

    monkeypatch.setattr("senapathy.audit.reconstruct.connect", raising_connect)
    sampler = FakeSampler({("invoice", "id"): 1, ("customer", "id"): 1})
    executable, params, note, limit_source, limit_request_id = resolve_statement(
        _target(), "abc123",
        "SELECT * FROM invoice i JOIN customer c ON c.id = i.customer_id "
        "WHERE id = ?", sampler)
    assert executable is None
    assert "no unambiguous table" in note


def test_resolve_statement_refuses_a_sampled_value_shaped_like_an_identifier(
    monkeypatch,
) -> None:
    """PROJ-01's own failure, reproduced on purpose: a 'value' that is
    really the parameter's own name ('pageSize') must never be bound."""
    def raising_connect(target: MySQLTarget):
        from mysql.connector import Error as MySQLError
        raise MySQLError("not enabled")

    monkeypatch.setattr("senapathy.audit.reconstruct.connect", raising_connect)
    sampler = FakeSampler({("invoice", "pageSize"): "pageSize"})
    executable, params, note, limit_source, limit_request_id = resolve_statement(
        _target(), "abc123", "select * from invoice where pageSize = ?", sampler)
    assert executable is None
    assert "looks like an identifier" in note


def test_resolve_statement_refuses_when_the_table_has_no_rows_to_sample(
    monkeypatch,
) -> None:
    def raising_connect(target: MySQLTarget):
        from mysql.connector import Error as MySQLError
        raise MySQLError("not enabled")

    monkeypatch.setattr("senapathy.audit.reconstruct.connect", raising_connect)
    sampler = FakeSampler({})
    executable, params, note, limit_source, limit_request_id = resolve_statement(
        _target(), "abc123", "select * from invoice where customer_id = ?", sampler)
    assert executable is None
    assert "no real value could be sampled" in note


# ───────────────────────────────  classify  ───────────────────────────────


def _reading(*, examined: int, returned: int, access_types=None,
            keys_used=None, tree: str = "") -> PlanReading:
    return PlanReading(
        tree=tree, access_types=access_types or {"invoice": "ref"},
        keys_used=keys_used or {"invoice": "idx_invoice_customer_id"},
        rows_examined=examined, rows_returned=returned, actual_ms=1.0,
    )


def test_a_tight_ratio_is_optimal() -> None:
    reading = _reading(examined=10, returned=5)
    classification, marker = classify(reading, "select * from invoice where customer_id = %s")
    assert classification == "OPTIMAL"
    assert marker is None


def test_a_function_wrapped_column_is_non_sargable_even_with_an_index() -> None:
    reading = _reading(examined=5000, returned=1)
    classification, marker = classify(
        reading, "select * from invoice where DATE(created_at) >= %s")
    assert classification == "NON_SARGABLE"
    assert "function-wrapped column" in marker


def test_a_bad_ratio_with_no_index_and_one_table_is_missing_index() -> None:
    reading = _reading(examined=1_500_000, returned=3, keys_used={"invoice": None})
    classification, marker = classify(reading, "select * from invoice where region = %s")
    assert classification == "MISSING_INDEX"
    assert marker is None


def test_a_clustered_primary_key_scan_is_missing_index_not_over_fetching() -> None:
    """Live-confirmed against mini-spring's own `payment` table: no index
    on `paid_date`, yet `EXPLAIN` reports `type=index, key=PRIMARY,
    possible_keys=NULL` for `WHERE paid_date = ? ORDER BY id LIMIT ?` --
    InnoDB walks the clustered primary key's own order to serve the ORDER
    BY + LIMIT, filtering as it goes. A key NAME is present (PRIMARY), but
    nothing actually serves the predicate: this is a scan, not a seek, and
    `any_index_used` alone cannot tell the difference."""
    reading = _reading(
        examined=51_301, returned=50,
        access_types={"payment": "index"}, keys_used={"payment": "PRIMARY"},
    )
    classification, marker = classify(
        reading, "select * from payment where paid_date = %s order by id limit 50")
    assert classification == "MISSING_INDEX"
    assert marker is None


def test_a_genuine_primary_key_seek_is_not_mistaken_for_a_scan() -> None:
    """The same key name (PRIMARY) with a SEEK access type (an id lookup,
    say) must not be swept into the scan fix -- only ALL/index access
    types count as a scan."""
    reading = _reading(
        examined=1, returned=1,
        access_types={"invoice": "const"}, keys_used={"invoice": "PRIMARY"},
    )
    classification, marker = classify(reading, "select * from invoice where id = %s")
    assert classification == "OPTIMAL"


def test_a_join_product_marker_is_bad_join_order() -> None:
    reading = _reading(
        examined=900_000, returned=4,
        access_types={"invoice": "ALL", "customer": "ALL"},
        keys_used={"invoice": None, "customer": None},
        tree="-> Nested loop inner join (using join buffer)",
    )
    classification, marker = classify(reading, "select * from invoice, customer")
    assert classification == "BAD_JOIN_ORDER"
    assert marker == "using join buffer"


def test_a_multi_table_scan_with_no_marker_is_still_bad_join_order() -> None:
    reading = _reading(
        examined=900_000, returned=4,
        access_types={"invoice": "ALL", "customer": "ALL"},
        keys_used={"invoice": None, "customer": None},
        tree="-> Nested loop inner join",
    )
    classification, marker = classify(reading, "select * from invoice, customer")
    assert classification == "BAD_JOIN_ORDER"
    assert marker is None


def test_a_bad_ratio_with_an_index_already_used_is_over_fetching() -> None:
    reading = _reading(examined=50_000, returned=1)
    classification, marker = classify(reading, "select * from invoice where customer_id = %s")
    assert classification == "OVER_FETCHING"
    assert marker is None


def test_zero_rows_returned_does_not_divide_by_zero() -> None:
    reading = _reading(examined=1000, returned=0, keys_used={"invoice": None})
    classification, _marker = classify(reading, "select * from invoice where region = %s")
    assert classification == "MISSING_INDEX"


# ───────────────────────────────  candidates  ─────────────────────────────


class FakeStatement:
    def __init__(self, *, digest_text, executable_sql, access_types,
                keys_used, rows_examined=1_500_000, actual_ms=900.0,
                resolved_params=None):
        self.digest_text = digest_text
        self.executable_sql = executable_sql
        self.access_types = access_types
        self.keys_used = keys_used
        self.rows_examined = rows_examined
        self.actual_ms = actual_ms
        self.resolved_params = resolved_params or []


class FakeIndexBox:
    def __init__(self, result: IndexBenefitResult,
                write_volume: dict[str, int] | None = None) -> None:
        self.result = result
        self.write_volume = write_volume or {}
        self.calls: list[tuple[str, list[str], str, list[str] | None]] = []
        self.hint_positions: list[int | None] = []

    def index_benefit_measured(self, table, columns, sql, params=None, *,
                               benchmark_hint_position=None,
                               benchmark_rounds=0):
        self.calls.append((table, columns, sql, params))
        self.hint_positions.append(benchmark_hint_position)
        return self.result

    def table_write_volume(self, table):
        return self.write_volume


def test_benchmark_rounds_is_large_enough_for_the_gates_own_p_value_bar() -> None:
    """Live-confirmed, not a hypothetical: mini-spring's own IDX-01
    scenario measured 5/5 rounds faster -- the best possible outcome for 5
    paired samples -- and STATEMENT_FASTER still failed, because Wilcoxon
    signed-rank's exact two-sided p-value over n non-tied paired samples
    has a FLOOR of 2 * 0.5**n regardless of effect size, and that floor
    sits ABOVE gate.rules.MAX_INDEX_P_VALUE at n=5. This test ties the two
    constants together across modules so neither can drift out of a
    mathematically achievable relationship again."""
    from senapathy.audit.candidates import BENCHMARK_ROUNDS
    from senapathy.gate.rules import MAX_INDEX_P_VALUE

    floor = 2 * 0.5 ** BENCHMARK_ROUNDS
    assert floor < MAX_INDEX_P_VALUE, (
        f"{BENCHMARK_ROUNDS} round(s) floors at p={floor}, which cannot "
        f"clear MAX_INDEX_P_VALUE={MAX_INDEX_P_VALUE} even in the best "
        f"possible case"
    )


def test_propose_index_candidate_uses_the_unindexed_table_and_its_columns() -> None:
    before = PlanReading(tree=TREE, rows_examined=1_500_000, rows_returned=3,
                        actual_ms=900.0, access_types={"invoice": "ALL"})
    after = PlanReading(tree=FAST_TREE, rows_examined=3, rows_returned=3,
                       actual_ms=0.9, access_types={"invoice": "ref"})
    box = FakeIndexBox(
        IndexBenefitResult(
            table="invoice", columns=["customer_id", "status"],
            index_name="senapathy_tmp_x", before=before, after=after,
            size_bytes=4096,
            without_index_ms_samples=[20.0, 21.0, 19.5, 20.5, 20.0],
            with_index_ms_samples=[2.0, 2.1, 1.9, 2.2, 2.0],
        ),
        write_volume={"insert": 10, "update": 2, "delete": 0, "fetch": 5000},
    )
    statement = FakeStatement(
        digest_text="select * from invoice where customer_id = ? and status = ?",
        executable_sql="select * from invoice where customer_id = %s and status = %s",
        access_types={"invoice": "ALL"}, keys_used={"invoice": None},
        resolved_params=["42", "OPEN"])

    candidate = propose_index_candidate(box, statement)

    assert candidate.table == "invoice"
    assert candidate.columns == ["customer_id", "status"]
    assert candidate.rows_examined_before == 1_500_000
    assert candidate.rows_examined_after == 3
    assert candidate.already_covered_by is None
    assert box.calls == [("invoice", ["customer_id", "status"],
                         statement.executable_sql, ["42", "OPEN"])]
    # The hint position lands right after "invoice" in the FROM clause.
    assert box.hint_positions == [len("select * from invoice")]

    assert "customer_id, status" in candidate.column_rationale
    assert "leading" in candidate.column_rationale
    assert candidate.plan_before.startswith("invoice:")
    assert candidate.plan_after.startswith("invoice:")
    assert candidate.size_bytes == 4096
    assert candidate.table_writes_since_reset == {
        "insert": 10, "update": 2, "delete": 0, "fetch": 5000}
    assert candidate.benchmark_rounds == 5
    assert candidate.improvement_pct is not None and candidate.improvement_pct > 0
    assert candidate.benchmark_p_value is not None
    assert candidate.access_type_before == "ALL"
    assert candidate.access_type_after == "ref"


def test_propose_index_candidate_resolves_access_type_through_an_alias() -> None:
    """Live-confirmed against mini-spring's own IDX-01 scenario: a JPQL
    constructor query aliases its table (`p1_0`), so EXPLAIN's own rows
    come back keyed by that alias, not the bare table name -- looking
    access_type up by the real table name silently returned "" for both
    sides, and PLAN_IMPROVED failed every time regardless of the real
    plan change underneath."""
    before = PlanReading(tree=TREE, rows_examined=51_301, rows_returned=50,
                        actual_ms=7.85, access_types={"p1_0": "index"},
                        keys_used={"p1_0": "PRIMARY"})
    after = PlanReading(tree=FAST_TREE, rows_examined=50, rows_returned=50,
                       actual_ms=0.12, access_types={"p1_0": "ref"},
                       keys_used={"p1_0": "senapathy_tmp_x"})
    box = FakeIndexBox(IndexBenefitResult(
        table="payment", columns=["paid_date"],
        index_name="senapathy_tmp_x", before=before, after=after))
    statement = FakeStatement(
        digest_text="select p1_0.id from payment p1_0 where p1_0.paid_date = ?",
        executable_sql="select p1_0.id from payment p1_0 where p1_0.paid_date = %s",
        access_types={"p1_0": "index"}, keys_used={"p1_0": "PRIMARY"},
        resolved_params=["2023-07-02"])

    candidate = propose_index_candidate(box, statement)

    assert candidate.table == "payment"
    assert candidate.access_type_before == "index"
    assert candidate.access_type_after == "ref"


def test_propose_index_candidate_reports_an_already_covered_refusal() -> None:
    box = FakeIndexBox(IndexBenefitResult(
        table="invoice", columns=["customer_id"], index_name=None,
        covered_by="idx_invoice_customer_id_status",
        covered_by_columns=["customer_id", "status"]))
    statement = FakeStatement(
        digest_text="select * from invoice where customer_id = ?",
        executable_sql="select * from invoice where customer_id = %s",
        access_types={"invoice": "ALL"}, keys_used={"invoice": None})

    candidate = propose_index_candidate(box, statement)

    assert candidate.already_covered_by == "idx_invoice_customer_id_status"
    assert candidate.before is None if hasattr(candidate, "before") else True


def test_propose_index_candidate_returns_none_without_a_resolvable_column() -> None:
    box = FakeIndexBox(IndexBenefitResult(table="invoice", columns=[], index_name=None))
    statement = FakeStatement(
        digest_text="select count(*) from invoice",
        executable_sql="select count(*) from invoice",
        access_types={"invoice": "ALL"}, keys_used={"invoice": None})

    assert propose_index_candidate(box, statement) is None


class FakeRewriteBox:
    def __init__(self, *, equivalence: str, a_times: list[float],
                b_times: list[float]) -> None:
        self.equivalence = equivalence
        self.a_times = a_times
        self.b_times = b_times
        self.equivalence_calls: list[tuple[str, str, list[str] | None]] = []

    def query_equivalent(self, original_sql, candidate_sql, params=None):
        self.equivalence_calls.append((original_sql, candidate_sql, params))
        return self.equivalence

    def plan_delta(self, sql_before, sql_after, params=None):
        return (PlanReading(tree=TREE, rows_examined=1_500_000, rows_returned=3),
                PlanReading(tree=FAST_TREE, rows_examined=3, rows_returned=3))

    def benchmark_interleaved(self, sql_a, sql_b, params=None, *, rounds=5):
        return list(self.a_times[:rounds]), list(self.b_times[:rounds])


def test_propose_rewrite_candidate_drops_the_date_wrapper_and_proves_it() -> None:
    box = FakeRewriteBox(equivalence="EQUIVALENT\n  original 3 row(s)",
                         a_times=[20.0] * 5, b_times=[2.0] * 5)
    statement = FakeStatement(
        digest_text="select * from invoice where DATE(created_at) >= ?",
        executable_sql="select * from invoice where DATE(created_at) >= %s",
        access_types={"invoice": "ALL"}, keys_used={"invoice": None},
        resolved_params=["2024-01-01"])

    candidate = propose_rewrite_candidate(box, statement)

    assert candidate is not None
    assert candidate.sql_after == "select * from invoice where created_at >= %s"
    assert candidate.equivalence_verified is True
    assert candidate.improvement_pct is not None and candidate.improvement_pct > 0
    assert box.equivalence_calls


def test_propose_rewrite_candidate_handles_mysqls_own_digest_spacing() -> None:
    r"""Live-confirmed against COL-06's real query: a digest's own
    `executable_sql` carries backticks and spaces around every token
    (`DATE ( \`cp1_0\` . \`paid_on\` ) >= %s`), not the compact hand-written
    form `DATE(col) >= %s` -- the rewrite regex must match this shape too,
    not just the style a human would type."""
    box = FakeRewriteBox(equivalence="EQUIVALENT", a_times=[20.0] * 5,
                         b_times=[2.0] * 5)
    statement = FakeStatement(
        digest_text="select * from `customer_payment` `cp1_0` where "
                    "DATE ( `cp1_0` . `paid_on` ) >= ?",
        executable_sql="select * from `customer_payment` `cp1_0` where "
                       "DATE ( `cp1_0` . `paid_on` ) >= %s",
        access_types={"customer_payment": "ALL"},
        keys_used={"customer_payment": None}, resolved_params=["2024-01-01"])

    candidate = propose_rewrite_candidate(box, statement)

    assert candidate is not None
    assert candidate.sql_after == (
        "select * from `customer_payment` `cp1_0` where "
        "`cp1_0` . `paid_on` >= %s")


def test_propose_rewrite_candidate_refuses_a_rewrite_that_is_not_equivalent() -> None:
    box = FakeRewriteBox(equivalence="NOT EQUIVALENT\n  DIFFERENCE x",
                         a_times=[20.0] * 5, b_times=[2.0] * 5)
    statement = FakeStatement(
        digest_text="select * from invoice where DATE(created_at) >= ?",
        executable_sql="select * from invoice where DATE(created_at) >= %s",
        access_types={"invoice": "ALL"}, keys_used={"invoice": None},
        resolved_params=["2024-01-01"])

    assert propose_rewrite_candidate(box, statement) is None


def test_propose_rewrite_candidate_handles_patterns_it_does_not_know_how_to_rewrite() -> None:
    box = FakeRewriteBox(equivalence="EQUIVALENT", a_times=[1.0], b_times=[1.0])
    statement = FakeStatement(
        digest_text="select * from invoice where name like '%smith'",
        executable_sql="select * from invoice where name like '%smith'",
        access_types={"invoice": "ALL"}, keys_used={"invoice": None})

    assert propose_rewrite_candidate(box, statement) is None
    assert box.equivalence_calls == []


# ─────────────────────────────  run_sql_audit  ────────────────────────────


def test_run_sql_audit_skips_session_control_and_classifies_the_rest(
    connected, tmp_path: Path, monkeypatch,
) -> None:
    """End to end against the real DbToolbox and a fake server -- the same
    full-scan shape COL-04 hit (a `select` digest with no index) is the
    proof this phase exists for."""
    server = script(indexes=[])
    box = toolbox(server, connected)

    def fake_resolve(target, digest, digest_text, sampler, corpus=None):
        return ("select sum(amount) from invoice where customer_id = %s "
                "and status = 'OPEN'", ["42"], "resolved by sampling real column values",
                None, None)

    monkeypatch.setattr("senapathy.audit.run.resolve_statement", fake_resolve)
    monkeypatch.setattr("senapathy.audit.run.DbToolbox", lambda *a, **k: box)

    digests = [
        SqlDigestStat(digest="d-set", sample_sql="SET autocommit=0", calls=1,
                      total_ms=0.1, mean_ms=0.1),
        SqlDigestStat(digest="d-scan", sample_sql=SQL, calls=10, total_ms=9000.0,
                      mean_ms=900.0),
    ]

    audit = run_sql_audit(
        endpoint="GET /invoices", measured_total_ms=9500.0, digests=digests,
        target=_target(), allowed_tables=["invoice", "customer"],
        run_dir=tmp_path)

    assert len(audit.statements) == 1
    statement = audit.statements[0]
    assert statement.digest == "d-scan"
    # The fake server's plan reads an index (key=idx_invoice_customer_id)
    # but still examines 2,645 rows for the single row returned -- an
    # index that exists but is not selective enough, not a missing one.
    assert statement.classification == "OVER_FETCHING"
    assert statement.rows_examined == 2645
    assert (tmp_path / "sql_audit.json").exists()


def test_run_sql_audit_records_an_unresolvable_digest_as_unanalysable_not_skipped(
    tmp_path: Path, monkeypatch,
) -> None:
    def fake_resolve(target, digest, digest_text, sampler, corpus=None):
        return None, [], "no real value could be sampled for invoice.customer_id", None, None

    monkeypatch.setattr("senapathy.audit.run.resolve_statement", fake_resolve)
    monkeypatch.setattr("senapathy.audit.run.DbToolbox", lambda *a, **k: object())
    monkeypatch.setattr("senapathy.audit.run.ValueSampler", lambda *a, **k: object())

    digests = [SqlDigestStat(digest="d1", sample_sql=SQL, calls=1, total_ms=1.0, mean_ms=1.0)]

    audit = run_sql_audit(
        endpoint="GET /invoices", measured_total_ms=1.0, digests=digests,
        target=_target(), allowed_tables=["invoice"], run_dir=tmp_path)

    assert len(audit.statements) == 1
    assert audit.statements[0].classification == "UNANALYSABLE"
    assert "no real value" in audit.statements[0].unanalysable_reason


def test_load_sql_audit_returns_none_when_no_artifact_was_ever_written(
    tmp_path: Path,
) -> None:
    from senapathy.audit.run import load_sql_audit
    assert load_sql_audit(tmp_path / "sql_audit.json") is None


def test_load_sql_audit_round_trips_what_run_sql_audit_wrote(
    tmp_path: Path, monkeypatch,
) -> None:
    from senapathy.audit.run import load_sql_audit

    def fake_resolve(target, digest, digest_text, sampler, corpus=None):
        return None, [], "unresolved", None, None

    monkeypatch.setattr("senapathy.audit.run.resolve_statement", fake_resolve)
    monkeypatch.setattr("senapathy.audit.run.DbToolbox", lambda *a, **k: object())
    monkeypatch.setattr("senapathy.audit.run.ValueSampler", lambda *a, **k: object())

    digests = [SqlDigestStat(digest="d1", sample_sql=SQL, calls=1, total_ms=1.0, mean_ms=1.0)]
    run_sql_audit(endpoint="GET /x", measured_total_ms=1.0, digests=digests,
                  target=_target(), allowed_tables=["invoice"], run_dir=tmp_path)

    reloaded = load_sql_audit(tmp_path / "sql_audit.json")
    assert reloaded is not None
    assert reloaded.endpoint == "GET /x"
    assert reloaded.statements[0].digest == "d1"
