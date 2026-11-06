"""Unit tests for the measurement harness. No network, no database.

Every test here drives the code with fabricated samples or a stub transport,
which is the point: the arithmetic that turns samples into a p95 is the part a
report depends on, so it has to be checkable without a live app.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from perfagent.contracts import RequestCorpus, RequestSpec, SqlDigestStat
from perfagent.errors import ContractViolation, ToolError
from perfagent.measure.baseline import derive_facts, is_session_control, render_summary
from perfagent.measure.bench import latency_stats, percentile, run_benchmark
from perfagent.measure.corpus import corpus_hash, load_corpus, save_corpus


# ────────────────────────────  percentiles  ─────────────────────────────

def test_percentile_endpoints_and_interpolation() -> None:
    samples = [float(i) for i in range(1, 101)]  # 1..100
    assert percentile(samples, 0) == 1.0
    assert percentile(samples, 100) == 100.0
    assert percentile(samples, 50) == pytest.approx(50.5)
    assert percentile(samples, 95) == pytest.approx(95.05)


def test_percentile_single_sample() -> None:
    assert percentile([7.5], 99) == 7.5


def test_percentile_rejects_empty() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)


def test_percentiles_come_from_raw_samples_not_averages() -> None:
    """Ninety fast requests and ten slow ones.

    The mean hides the tail; p95 must not. If this ever starts reporting
    something near the mean, someone has averaged per-request means somewhere.
    """
    samples = [10.0] * 90 + [2000.0] * 10
    stats = latency_stats(samples)
    assert stats.n == 100
    assert stats.p50_ms == pytest.approx(10.0)
    assert stats.p95_ms == pytest.approx(2000.0)
    assert stats.mean_ms == pytest.approx(209.0)


def test_latency_stats_stdev_needs_two_samples() -> None:
    assert latency_stats([42.0]).stdev_ms == 0.0


# ──────────────────────────────  corpus  ────────────────────────────────

def _spec(request_id: str, **body: object) -> RequestSpec:
    return RequestSpec(
        id=request_id, method="POST", path="/api/customers/search", body=dict(body)
    )


def test_there_is_no_shipped_corpus_to_fall_back_to() -> None:
    """The deliberate inversion of the two tests that used to live here.

    `perfagent/measure/corpus.json` held 12 hand-written requests against
    one endpoint of one repo and was the DEFAULT for every run, so a run
    against any other endpoint silently baselined and gated that one. The
    corpus is per-run evidence now; the absence of a default is the feature.
    """
    from perfagent.measure.corpus import DEFAULT_CORPUS_PATH

    assert DEFAULT_CORPUS_PATH is None
    assert not (Path(__file__).resolve().parent.parent
                / "perfagent/measure/corpus.json").exists()


def test_loading_a_corpus_with_no_path_refuses_loudly() -> None:
    with pytest.raises(ContractViolation, match="no corpus path"):
        load_corpus(None)


def test_corpus_hash_is_content_addressed_and_order_sensitive() -> None:
    a, b = _spec("a", name="x"), _spec("b", name="y")
    assert corpus_hash([a, b]) == corpus_hash([a, b])
    assert corpus_hash([a, b]) != corpus_hash([b, a])
    assert corpus_hash([a, b]) != corpus_hash([a])


def test_load_corpus_rejects_a_stale_stated_hash(tmp_path: Path) -> None:
    path = tmp_path / "corpus.json"
    path.write_text(
        json.dumps(
            {
                "benchmark": [_spec("a", name="x").model_dump(mode="json")],
                "hash": "deadbeefdeadbeef",
            }
        )
    )
    with pytest.raises(ContractViolation, match="states hash"):
        load_corpus(path)


def test_load_corpus_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "corpus.json"
    path.write_text(
        json.dumps(
            {"benchmark": [_spec("dup").model_dump(mode="json")] * 2}
        )
    )
    with pytest.raises(ContractViolation, match="duplicate request ids"):
        load_corpus(path)


def test_load_corpus_rejects_empty_and_missing(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"benchmark": []}))
    with pytest.raises(ContractViolation, match="empty"):
        load_corpus(empty)
    with pytest.raises(ContractViolation, match="not found"):
        load_corpus(tmp_path / "nope.json")


def test_save_corpus_round_trips(tmp_path: Path) -> None:
    original = RequestCorpus(
        benchmark=[_spec("a", name="x"), _spec("b", name="y")],
        held_back=[_spec("c", name="z")],
        hash=corpus_hash([_spec("a", name="x"), _spec("b", name="y")],
                         [_spec("c", name="z")]),
    )
    out = save_corpus(original, tmp_path / "copy.json")
    assert load_corpus(out).hash == original.hash


# ────────────────────────────  benchmark  ───────────────────────────────

def _corpus(*specs: RequestSpec) -> RequestCorpus:
    return RequestCorpus(benchmark=list(specs), hash=corpus_hash(list(specs)))


def _stub_client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler, timeout=5.0)


def test_warmup_is_discarded_and_cycles_the_corpus() -> None:
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["name"])
        return httpx.Response(200, json={"ok": True})

    corpus = _corpus(_spec("one", name="one"), _spec("two", name="two"))
    with _stub_client(httpx.MockTransport(handle)) as client:
        result = run_benchmark(
            "http://app", corpus, warmup=4, iterations=6, client=client
        )

    assert len(seen) == 10  # 4 warmup + 6 measured, all really sent
    assert result["warmup_discarded"]["n"] == 4
    assert result["latency"].n == 6  # but only the measured ones are counted
    assert len(result["samples"]) == 6
    assert [s["request_id"] for s in result["samples"]] == [
        "one", "two", "one", "two", "one", "two",
    ]


def test_on_measure_start_fires_once_between_warmup_and_measurement() -> None:
    calls: list[int] = []
    sent: list[int] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(1)
        return httpx.Response(200, json={})

    corpus = _corpus(_spec("one", name="one"))
    with _stub_client(httpx.MockTransport(handle)) as client:
        run_benchmark(
            "http://app",
            corpus,
            warmup=5,
            iterations=3,
            client=client,
            on_measure_start=lambda: calls.append(len(sent)),
        )

    # Exactly once, after all 5 warmups and before any measured request.
    assert calls == [5]


def test_error_responses_stay_in_the_distribution() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        name = json.loads(request.content)["name"]
        return httpx.Response(400 if name == "bad" else 200, json={})

    corpus = _corpus(_spec("ok", name="ok"), _spec("bad", name="bad"))
    with _stub_client(httpx.MockTransport(handle)) as client:
        result = run_benchmark("http://app", corpus, warmup=0, iterations=4, client=client)

    assert result["status_counts"] == {"200": 2, "400": 2}
    assert result["latency"].n == 4
    assert result["transport_failures"] == 0


def test_transport_failures_are_recorded_but_excluded_from_latency() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        name = json.loads(request.content)["name"]
        if name == "dead":
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json={})

    corpus = _corpus(_spec("ok", name="ok"), _spec("dead", name="dead"))
    with _stub_client(httpx.MockTransport(handle)) as client:
        result = run_benchmark("http://app", corpus, warmup=0, iterations=4, client=client)

    assert result["transport_failures"] == 2
    assert result["latency"].n == 2  # a refused connection is not a 0ms request
    assert len(result["errors"]) == 2


def test_total_failure_raises_a_tool_error() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    corpus = _corpus(_spec("one", name="one"))
    with _stub_client(httpx.MockTransport(handle)) as client:
        with pytest.raises(ToolError, match="all 3 measured requests failed"):
            run_benchmark("http://app", corpus, warmup=0, iterations=3, client=client)


def test_run_benchmark_rejects_nonsense_arguments() -> None:
    corpus = _corpus(_spec("one", name="one"))
    with pytest.raises(ValueError):
        run_benchmark("http://app", corpus, warmup=-1, iterations=1)
    with pytest.raises(ValueError):
        run_benchmark("http://app", corpus, warmup=0, iterations=0)


# ──────────────────────────  derived facts  ─────────────────────────────

def _digest(sql: str, calls: int, total_ms: float, **kw: object) -> SqlDigestStat:
    return SqlDigestStat(
        digest=sql[:16],
        sample_sql=sql,
        calls=calls,
        total_ms=total_ms,
        mean_ms=total_ms / calls,
        **kw,
    )


def test_session_control_classification() -> None:
    assert is_session_control(_digest("SET `autocommit` = ?", 1, 1.0))
    assert is_session_control(_digest("COMMIT", 1, 1.0))
    assert is_session_control(_digest("SELECT @@SESSION . `tx_isolation`", 1, 1.0))
    assert not is_session_control(_digest("SELECT `c1_0` . `id` FROM `customer`", 1, 1.0))


def test_derive_facts_arithmetic_and_n_plus_one_selection() -> None:
    bench = {
        "samples": [
            {"i": 0, "request_id": "a", "status": 200, "latency_ms": 100.0, "error": None},
            {"i": 1, "request_id": "a", "status": 200, "latency_ms": 300.0, "error": None},
            {"i": 2, "request_id": "b", "status": None, "latency_ms": 5000.0, "error": "boom"},
        ]
    }
    digests = [
        _digest("SELECT SUM ( `amount` ) FROM `invoice`", 20, 40.0, rows_examined=2000),
        _digest("SELECT `id` FROM `customer`", 2, 8.0, rows_examined=500, no_index_used=True),
        _digest("SET `autocommit` = ?", 60, 0.5),
    ]
    facts = derive_facts(bench, digests)

    assert facts["measured_requests"] == 2          # the failure is not a request
    assert facts["measured_latency_ms"] == 400.0
    assert facts["total_sql_statements"] == 82
    assert facts["app_sql_statements"] == 22
    assert facts["session_control_statements"] == 60
    assert facts["app_statements_per_request"] == 11.0
    assert facts["total_sql_ms"] == 48.5
    assert facts["sql_share_of_latency_pct"] == pytest.approx(12.125, abs=0.006)
    assert facts["digests_without_index"] == 1

    # The N+1 candidate must be application SQL, never `SET autocommit`, even
    # though that has by far the highest call count.
    assert facts["busiest_digest"]["calls"] == 20
    assert facts["busiest_digest"]["calls_per_request"] == 10.0
    assert "invoice" in facts["busiest_digest"]["sample_sql"]


def test_derive_facts_survives_an_empty_digest_set() -> None:
    bench = {"samples": [{"i": 0, "request_id": "a", "status": 200, "latency_ms": 1.0, "error": None}]}
    facts = derive_facts(bench, [])
    assert facts["total_sql_statements"] == 0
    assert facts["sql_share_of_latency_pct"] == 0.0
    assert facts["busiest_digest"] is None


# ───────────────────────────  rendering  ────────────────────────────────

def test_render_summary_quotes_only_recorded_numbers() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    corpus = _corpus(_spec("one", name="one"))
    with _stub_client(httpx.MockTransport(handle)) as client:
        bench = run_benchmark("http://app", corpus, warmup=0, iterations=4, client=client)

    digests = [_digest("SELECT SUM ( `amount` ) FROM `invoice`", 8, 40.0, rows_examined=99)]
    document = {
        "schema_version": 1,
        "kind": "baseline",
        "arm": "blue",
        "captured_at": "20260101T000000Z",
        "target": {
            "base_url": "http://app",
            "health": {"status": "UP"},
            "mysql": "root@127.0.0.1:3307/mini",
        },
        "corpus": {"path": "x.json", "hash": corpus.hash, "requests": [{"id": "one"}]},
        "benchmark": bench,
        "sql_digests": [d.model_dump(mode="json") for d in digests],
        "derived": derive_facts(bench, digests),
    }

    text = render_summary(document)
    assert "LATENCY (ms, from raw samples)" in text
    assert "TOP 1 SQL DIGESTS" in text
    assert "DERIVED" in text
    assert "8 calls (2.0 per request)" in text
    assert corpus.hash in text
