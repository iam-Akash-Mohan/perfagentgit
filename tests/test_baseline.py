"""run_baseline's substance checks: a baseline that never ran is refused.

Every network and database call `run_baseline` makes is monkeypatched at the
module boundary, so what's under test is the refusal logic itself, not
`health_check` or `run_benchmark` (both covered elsewhere).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from senapathy.contracts import SqlDigestStat
from senapathy.errors import ToolError
from senapathy.measure import baseline as baseline_mod
from senapathy.measure.corpus import save_corpus


def _corpus_path(tmp_path: Path) -> Path:
    from senapathy.contracts import RequestCorpus, RequestSpec
    from senapathy.measure.corpus import corpus_hash

    specs = [RequestSpec(id="r0", method="GET", path="/api/invoices?status=OPEN")]
    corpus = RequestCorpus(benchmark=specs, held_back=[],
                           hash=corpus_hash(specs, []))
    return save_corpus(corpus, tmp_path / "corpus.json")


def _bench(status_counts: dict[str, int], *, first_non_2xx: dict[str, Any] | None = None) -> dict[str, Any]:
    from senapathy.contracts import LatencyStats

    n = sum(status_counts.values())
    return {
        "base_url": "http://blue",
        "corpus_hash": "h",
        "requests_in_corpus": 1,
        "warmup": 0,
        "iterations": n,
        "started_at": "t0",
        "finished_at": "t1",
        "wall_ms": 1.0,
        "latency": LatencyStats(n=n, p50_ms=1, p95_ms=1, p99_ms=1,
                                mean_ms=1, stdev_ms=0),
        "status_counts": status_counts,
        "first_non_2xx": first_non_2xx,
        "transport_failures": 0,
        "warmup_discarded": {"n": 0, "failures": 0, "mean_ms": 0.0},
        "per_request": {},
        "samples": [{"i": 0, "request_id": "r0", "status": 200,
                    "latency_ms": 1.0, "error": None}],
        "errors": [],
    }


def _install(monkeypatch: pytest.MonkeyPatch, *, status_counts: dict[str, int],
            first_non_2xx: dict[str, Any] | None = None,
            digests: list[SqlDigestStat] | None = None) -> None:
    monkeypatch.setattr(baseline_mod, "health_check",
                        lambda url, timeout_s=10.0: {"url": url, "http_status": 200,
                                                      "status": "UP", "body": {}})
    monkeypatch.setattr(baseline_mod, "run_benchmark",
                        lambda *a, **k: _bench(status_counts, first_non_2xx=first_non_2xx))
    monkeypatch.setattr(baseline_mod, "reset_digests", lambda target: None)
    monkeypatch.setattr(baseline_mod, "collect_digests",
                        lambda target: list(digests or []))


def test_a_baseline_with_no_2xx_response_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The exact defect: a 400 on every request must not be written as if it
    measured the endpoint."""
    _install(monkeypatch, status_counts={"400": 100},
            first_non_2xx={"request_id": "r0", "status": 400,
                           "error": None, "body": '{"error":"status is required"}'})
    corpus_path = _corpus_path(tmp_path)

    with pytest.raises(ToolError) as exc:
        baseline_mod.run_baseline(
            base_url="http://blue", corpus_path=corpus_path,
            runs_dir=tmp_path / "runs", collect_sql=False,
        )
    message = str(exc.value)
    assert "no measured request" in message
    assert "400" in message
    assert "status is required" in message


def test_a_baseline_with_some_2xx_is_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, status_counts={"200": 90, "404": 10})
    corpus_path = _corpus_path(tmp_path)

    document, path = baseline_mod.run_baseline(
        base_url="http://blue", corpus_path=corpus_path,
        runs_dir=tmp_path / "runs", collect_sql=False,
    )
    assert path.exists()
    assert document["benchmark"]["status_counts"] == {"200": 90, "404": 10}


def test_zero_sql_for_a_db_touching_endpoint_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """'I measured something that never ran' -- the comprehension model says
    this endpoint reads the database and not one SQL statement was seen."""
    from senapathy.measure.sql import MySQLTarget

    _install(monkeypatch, status_counts={"200": 100}, digests=[])
    corpus_path = _corpus_path(tmp_path)
    target = MySQLTarget(host="127.0.0.1", port=3307, user="root",
                         password="root", database="mini")

    with pytest.raises(ToolError) as exc:
        baseline_mod.run_baseline(
            base_url="http://blue", corpus_path=corpus_path,
            runs_dir=tmp_path / "runs", collect_sql=True, mysql=target,
            expects_sql=True,
        )
    assert "zero SQL statements" in str(exc.value)


def test_zero_sql_is_fine_when_the_model_expects_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An endpoint that genuinely touches no table must not be refused for
    having no SQL -- that would be a false positive on every DB-less repo."""
    from senapathy.measure.sql import MySQLTarget

    _install(monkeypatch, status_counts={"200": 100}, digests=[])
    corpus_path = _corpus_path(tmp_path)
    target = MySQLTarget(host="127.0.0.1", port=3307, user="root",
                         password="root", database="mini")

    document, path = baseline_mod.run_baseline(
        base_url="http://blue", corpus_path=corpus_path,
        runs_dir=tmp_path / "runs", collect_sql=True, mysql=target,
        expects_sql=False,
    )
    assert path.exists()


# ──────────────────────  input-hash provenance  ───────────────────────────


def test_baseline_hash_is_deterministic_and_content_based() -> None:
    a = {"corpus_hash": "c", "derived": {"total_sql_statements": 3}}
    b = {"derived": {"total_sql_statements": 3}, "corpus_hash": "c"}  # reordered
    c = {"corpus_hash": "c", "derived": {"total_sql_statements": 4}}

    assert baseline_mod.baseline_hash(a) == baseline_mod.baseline_hash(b)
    assert baseline_mod.baseline_hash(a) != baseline_mod.baseline_hash(c)


def test_run_baseline_stamps_a_top_level_corpus_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A resume's skip check reads this top-level field directly as JSON --
    it must be there, not only nested under document['corpus']['hash']."""
    _install(monkeypatch, status_counts={"200": 100})
    corpus_path = _corpus_path(tmp_path)

    document, path = baseline_mod.run_baseline(
        base_url="http://blue", corpus_path=corpus_path,
        runs_dir=tmp_path / "runs", collect_sql=False,
    )
    assert document["corpus_hash"] == document["corpus"]["hash"]
    on_disk = json.loads(path.read_text())
    assert on_disk["corpus_hash"] == document["corpus"]["hash"]
