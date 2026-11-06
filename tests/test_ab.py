"""P14 tests: interleaving, statistics, the mechanism check, and the refusal.

No network, no database, no JVM. Requests go through httpx.MockTransport and
SQL comes from hand-built SqlDigestStat fixtures, so the arithmetic that a
report depends on is checkable without two running apps.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from perfagent.contracts import (EquivalenceVerdict, RequestCorpus, RequestSpec,
                                 SqlDigestStat)
from perfagent.errors import ContractViolation, ToolError
from perfagent.measure.ab import (ArmSql, Pair, build_report, delta_pct,
                                  mann_whitney_p, mechanism_verdict,
                                  run_interleaved, wilcoxon_p)
from perfagent.measure.arms import (JVM_FLAGS, Arm, arm_argv,
                                    assert_build_current,
                                    assert_flags_identical)
from perfagent.measure.corpus import corpus_hash
from perfagent.measure.verify import (patch_identity, path_shape,
                                      require_verified, render_verdict,
                                      summarise_diffs)


def spec(request_id: str, **body) -> RequestSpec:
    return RequestSpec(
        id=request_id, method="POST", path="/api/customers/search", body=dict(body)
    )


def corpus_of(*specs: RequestSpec, held_back: tuple[RequestSpec, ...] = ()) -> RequestCorpus:
    return RequestCorpus(
        benchmark=list(specs), held_back=list(held_back),
        hash=corpus_hash(list(specs), list(held_back)),
    )


def arm_aware_client(blue_ms: float, green_ms: float) -> httpx.Client:
    """A transport that answers instantly but lets tests assert on the URL."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"port": request.url.port})
    return httpx.Client(transport=httpx.MockTransport(handler))


# ────────────────────────────  interleaving  ─────────────────────────────


def test_the_firing_order_strictly_alternates_blue_green() -> None:
    order: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        run = run_interleaved(
            "http://blue:1", "http://green:2",
            corpus_of(spec("a", name="a"), spec("b", name="b")),
            warmup=0, iterations=3, client=client,
        )

    arms = [s.arm for s in run.samples]
    # blue,green,blue,green ... for the whole run, never all-blue-then-all-green.
    assert arms == ["blue", "green"] * 6
    assert [s.seq for s in run.samples] == list(range(12))


def test_each_pair_is_the_same_request_on_both_arms() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        run = run_interleaved(
            "http://blue:1", "http://green:2",
            corpus_of(spec("a", name="a"), spec("b", name="b")),
            warmup=0, iterations=2, client=client,
        )

    assert [p.request_id for p in run.pairs] == ["a", "b", "a", "b"]
    assert [p.iteration for p in run.pairs] == [0, 0, 1, 1]
    # A pair's two samples are adjacent in the firing order.
    for index, pair in enumerate(run.pairs):
        blue_sample, green_sample = run.samples[index * 2], run.samples[index * 2 + 1]
        assert blue_sample.arm == "blue" and green_sample.arm == "green"
        assert blue_sample.request_id == green_sample.request_id == pair.request_id


def test_warmup_is_fired_and_discarded() -> None:
    fired: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fired.append(str(request.url))
        return httpx.Response(200, json={})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        run = run_interleaved(
            "http://blue:1", "http://green:2", corpus_of(spec("a", name="a")),
            warmup=4, iterations=2, client=client,
        )

    assert len(fired) == (4 + 2) * 2          # really sent
    assert run.discarded_warmup == 8
    assert len(run.pairs) == 2                # but not measured
    assert len(run.samples) == 4


def test_the_held_back_set_is_never_measured() -> None:
    """Measuring it would spend the exact property that makes it useful."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["name"])
        return httpx.Response(200, json={})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        run = run_interleaved(
            "http://blue:1", "http://green:2",
            corpus_of(spec("bench", name="bench"),
                      held_back=(spec("held", name="held"),)),
            warmup=0, iterations=2, client=client,
        )

    assert "held" not in seen
    assert {p.request_id for p in run.pairs} == {"bench"}


def test_a_transport_failure_on_one_arm_aborts_rather_than_half_measuring() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 2:
            raise httpx.ConnectError("green is down")
        return httpx.Response(200, json={})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ToolError, match="green arm failed"):
            run_interleaved(
                "http://blue:1", "http://green:2", corpus_of(spec("a", name="a")),
                warmup=0, iterations=1, client=client,
            )


def test_a_status_mix_difference_between_arms_is_refused() -> None:
    """A fast error path is the easiest way to accidentally win a benchmark."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.url.port == 2 and calls["n"] > 2:
            return httpx.Response(500, json={})
        return httpx.Response(200, json={})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ContractViolation, match="different status mixes"):
            run_interleaved(
                "http://blue:1", "http://green:2", corpus_of(spec("a", name="a")),
                warmup=0, iterations=3, client=client,
            )


def test_the_order_listing_shows_arm_and_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        run = run_interleaved(
            "http://blue:1", "http://green:2", corpus_of(spec("alpha", name="a")),
            warmup=0, iterations=2, client=client,
        )

    lines = run.order()
    assert len(lines) == 4
    assert "blue" in lines[0] and "alpha" in lines[0]
    assert "green" in lines[1]


# ────────────────────────────  identical flags  ──────────────────────────


def test_flags_differing_only_in_port_and_path_are_accepted() -> None:
    blue = Arm("blue", Path("/b"), 18080,
               ["java", *JVM_FLAGS, "-cp", "/b/target/classes",
                "-Dserver.port=18080", "com.ex.Application"])
    green = Arm("green", Path("/g"), 8081,
                ["java", *JVM_FLAGS, "-cp", "/g/target/classes",
                 "-Dserver.port=8081", "com.ex.Application"])
    assert_flags_identical(blue, green)


@pytest.mark.parametrize(
    "mutation",
    [
        ("-Xmx2g", 2),                       # heap differs
        ("-XX:TieredStopAtLevel=1", 1),      # JIT differs
    ],
)
def test_any_other_flag_difference_is_refused(mutation) -> None:
    token, index = mutation
    base = ["java", *JVM_FLAGS, "-cp", "/x/target/classes",
            "-Dserver.port=1", "com.ex.Application"]
    blue = Arm("blue", Path("/b"), 1, list(base))
    green = Arm("green", Path("/g"), 2, list(base))
    green.argv[index] = token
    with pytest.raises(ContractViolation, match="DIFFERENT flags"):
        assert_flags_identical(blue, green)


def _profile() -> "RepoProfile":
    """A minimal detected profile. arm_argv needs one: the entrypoint and the
    classes directory are properties of the repo, not constants."""
    from perfagent.contracts.repo import RepoProfile
    return RepoProfile.model_validate({
        "repo": "/tmp/x",
        "build": {"tool": "maven", "invocation": ["mvn"], "build_args": [],
                  "test_args": [], "classes_dir": "target/classes",
                  "build_inputs": ["src/"]},
        "layout": {"source_root": "src/main/java", "test_root": "src/test/java",
                   "resource_roots": ["src/main/resources"]},
        "entrypoint": "com.example.App",
        "run_profile": "bench",
        "database": {"kind": "none"},
        "write_allowlist": ["src/main/java"],
    })


def test_a_missing_classpath_file_is_reported_not_guessed(tmp_path: Path) -> None:
    with pytest.raises(ContractViolation, match="build_arm"):
        arm_argv(tmp_path, 8081, profile=_profile())


def test_arm_argv_refuses_without_a_profile(tmp_path: Path) -> None:
    """The entrypoint used to be the module constant `com.ex.Application`.
    A launch command naming the wrong main class fails in a way that reads
    like a classpath problem, so this refuses instead of guessing."""
    with pytest.raises(ContractViolation, match="needs a RepoProfile"):
        arm_argv(tmp_path, 8081)


def test_arm_argv_takes_the_entrypoint_and_profile_from_the_repo_profile(
    tmp_path: Path
) -> None:
    (tmp_path / "x-classpath.txt").write_text("/deps/a.jar")
    argv = arm_argv(tmp_path / "x", 9999, classpath_dir=tmp_path,
                    profile=_profile())
    assert argv[-1] == "com.example.App"
    assert "-Dspring.profiles.active=bench" in argv
    assert "-Dserver.port=9999" in argv
    assert any("target/classes" in a for a in argv)


def test_free_port_pair_returns_two_distinct_usable_ports() -> None:
    """Replaces the 18080/8081 literals: ports are chosen at runtime."""
    from perfagent.measure.arms import free_port_pair, port_is_free
    first, second = free_port_pair()
    assert first != second
    assert port_is_free(first) and port_is_free(second)


def test_xms_equals_xmx_in_the_shipped_flags() -> None:
    """Unequal heap bounds let one arm pay for growth the other avoided."""
    xms = [f for f in JVM_FLAGS if f.startswith("-Xms")]
    xmx = [f for f in JVM_FLAGS if f.startswith("-Xmx")]
    assert len(xms) == 1 and len(xmx) == 1
    assert xms[0][4:] == xmx[0][4:]


# ─────────────────────────────  statistics  ──────────────────────────────


def test_delta_pct_is_signed_and_negative_means_faster() -> None:
    assert delta_pct(100.0, 60.0) == -40.0
    assert delta_pct(100.0, 130.0) == 30.0
    assert delta_pct(0.0, 5.0) == 0.0


def test_a_real_difference_is_significant_and_noise_is_not() -> None:
    slow = [100.0 + i * 0.1 for i in range(30)]
    fast = [60.0 + i * 0.1 for i in range(30)]
    assert mann_whitney_p(slow, fast) < 0.01

    same_a = [100.0 + (i % 5) for i in range(30)]
    same_b = [100.0 + ((i + 2) % 5) for i in range(30)]
    assert mann_whitney_p(same_a, same_b) > 0.05


def test_identical_samples_are_not_reported_as_significant() -> None:
    assert mann_whitney_p([50.0] * 10, [50.0] * 10) == 1.0


def test_too_few_samples_return_one_rather_than_a_confident_number() -> None:
    assert mann_whitney_p([1.0, 2.0], [3.0, 4.0]) == 1.0


def test_wilcoxon_uses_the_pairing() -> None:
    pairs = [Pair(0, f"r{i}", 100.0 + i, 60.0 + i) for i in range(12)]
    assert wilcoxon_p(pairs) < 0.01
    assert wilcoxon_p([Pair(0, "r", 5.0, 5.0) for _ in range(12)]) is None


def test_interleaving_cancels_a_drift_that_sequential_would_not() -> None:
    """The whole reason for alternating.

    A machine that gets linearly slower during the run: interleaved pairs see
    the drift on both arms, so the measured delta stays ~0. Split the same
    samples into a blue-half and a green-half — what sequential running would
    produce — and the drift alone reads as a large regression.
    """
    n = 40
    drift = [i * 5.0 for i in range(n)]          # +0 .. +195 ms over the run
    blue_interleaved = [100.0 + drift[i] for i in range(n)]
    green_interleaved = [100.0 + drift[i] for i in range(n)]
    assert abs(delta_pct(
        sum(blue_interleaved) / n, sum(green_interleaved) / n)) < 1.0

    first_half, second_half = drift[: n // 2], drift[n // 2:]
    blue_sequential = [100.0 + d for d in first_half]
    green_sequential = [100.0 + d for d in second_half]
    sequential_delta = delta_pct(
        sum(blue_sequential) / len(blue_sequential),
        sum(green_sequential) / len(green_sequential),
    )
    assert sequential_delta > 50.0, "sequential must show the drift as a change"


# ──────────────────────────  mechanism check  ────────────────────────────


def digest(sql: str, calls: int, rows: int = 0) -> SqlDigestStat:
    return SqlDigestStat(
        digest=sql[:16], sample_sql=sql, calls=calls, total_ms=1.0, mean_ms=0.1,
        rows_examined=rows, rows_sent=0, no_index_used=False,
    )


def test_a_removed_n_plus_one_confirms_the_mechanism() -> None:
    blue = ArmSql(0, 4_000_000, [digest("SELECT amount FROM invoice", 200, 4_000_000)])
    green = ArmSql(0, 40_000, [digest("SELECT amount FROM invoice", 2, 40_000)])
    confirmed, note = mechanism_verdict(blue, green, requests=100)
    assert confirmed
    assert "2.00 -> 0.02" in note
    assert "database work changed" in note


def test_unmoved_sql_reports_right_answer_wrong_reason() -> None:
    """A latency win with unchanged SQL must be visible, not folded in."""
    blue = ArmSql(0, 4_000_000, [digest("SELECT amount FROM invoice", 847, 4_000_000)])
    green = ArmSql(0, 4_000_000, [digest("SELECT amount FROM invoice", 847, 4_000_000)])
    confirmed, note = mechanism_verdict(blue, green, requests=100)
    assert confirmed is False
    assert "did NOT change materially" in note
    assert "mechanism the patch claimed" in note


def test_session_control_statements_are_excluded_from_the_count() -> None:
    blue = ArmSql(0, 0, [digest("COMMIT", 500), digest("SELECT x FROM customer", 10)])
    assert blue.application_statements == 10


def test_the_report_computes_every_number_from_samples() -> None:
    pairs = [Pair(0, f"r{i}", 200.0 + i, 100.0 + i) for i in range(25)]
    run = type("R", (), {})()
    run.pairs = pairs
    run.arm_samples = lambda arm: [
        p.blue_ms if arm == "blue" else p.green_ms for p in pairs
    ]
    blue = ArmSql(0, 1_000_000, [digest("SELECT a FROM invoice", 200, 1_000_000)])
    green = ArmSql(0, 10_000, [digest("SELECT a FROM invoice", 2, 10_000)])

    report = build_report(run, blue, green)

    assert report.interleaved is True          # contract-enforced literal
    assert report.p50_delta_pct < -45.0
    assert report.sql_count_before == 200 and report.sql_count_after == 2
    assert report.rows_examined_before == 1_000_000
    assert report.p_value < 0.01
    assert report.mechanism_confirmed is True


def test_perf_report_cannot_claim_it_was_sequential() -> None:
    from pydantic import ValidationError

    from perfagent.contracts import LatencyStats, PerfReport
    stats = LatencyStats(n=1, p50_ms=1, p95_ms=1, p99_ms=1, mean_ms=1, stdev_ms=0)
    with pytest.raises(ValidationError, match="interleaved"):
        PerfReport(
            before=stats, after=stats, p50_delta_pct=0, p95_delta_pct=0,
            sql_count_before=0, sql_count_after=0, rows_examined_before=0,
            rows_examined_after=0, p_value=1.0, mechanism_confirmed=False,
            mechanism_note="x", interleaved=False,
        )


# ─────────────────────  the refusal in front of measure  ─────────────────


def verification_doc(tmp_path: Path, **overrides) -> Path:
    verdict = EquivalenceVerdict(
        passed=True, compared=12, identical=12, db_state_matched=True,
        external_calls_matched=True, held_back_passed=True, oracle_hash="orc",
    )
    document = {
        "verdict": verdict.model_dump(mode="json"),
        "green_url": "http://localhost:8081",
        "oracle_hash": "orc",
        "oracle_base_commit": "abc",
        "corpus_hash": "cor",
        "benchmark_requests": ["a"],
        "held_back_requests": ["h"],
        "patch_identity": patch_identity("the diff"),
        "verified_at": "2026-09-04T00:00:00Z",
    }
    document.update(overrides)
    (tmp_path / "verification.json").write_text(json.dumps(document))
    return tmp_path


def test_measure_is_refused_when_no_verification_exists(tmp_path: Path) -> None:
    with pytest.raises(ContractViolation, match="no verification"):
        require_verified(tmp_path, green_diff="the diff")


def test_measure_is_refused_when_verification_failed(tmp_path: Path) -> None:
    failed = EquivalenceVerdict(
        passed=False, compared=12, identical=11, db_state_matched=True,
        external_calls_matched=True, held_back_passed=True, oracle_hash="orc",
    )
    directory = verification_doc(
        tmp_path,
        verdict=failed.model_dump(mode="json"),
        diffs_total=111407,
        diff_shapes=[{"shape": "$.content[*].tags[*]", "occurrences": 111407,
                      "requests": ["a"], "example_path": "$.content[0].tags[0]",
                      "example_before": "overdue", "example_after": "OVERDUE"}],
    )
    with pytest.raises(ContractViolation) as caught:
        require_verified(directory, green_diff="the diff")

    message = str(caught.value)
    assert "verification FAILED" in message
    # The TRUE total, not the capped list that was persisted.
    assert "111407" in message
    assert "$.content[*].tags[*]" in message


def test_measure_is_refused_when_the_patch_changed_after_verification(
    tmp_path: Path, monkeypatch
) -> None:
    """Verify, then edit green, then measure would attribute new numbers to an
    old verdict."""
    directory = verification_doc(tmp_path)

    import perfagent.measure.verify as verify_module

    monkeypatch.setattr(
        verify_module, "load_oracle",
        lambda path=None: type("O", (), {"hash": "orc"})(),
    )
    monkeypatch.setattr(
        verify_module, "load_corpus",
        lambda path=None: type("C", (), {"hash": "cor"})(),
    )

    # Unchanged patch: accepted.
    assert require_verified(directory, green_diff="the diff")["oracle_hash"] == "orc"

    with pytest.raises(ContractViolation, match="changed since it was verified"):
        require_verified(directory, green_diff="a DIFFERENT diff")


def test_measure_is_refused_when_the_oracle_changed(tmp_path: Path, monkeypatch) -> None:
    directory = verification_doc(tmp_path)

    import perfagent.measure.verify as verify_module

    monkeypatch.setattr(
        verify_module, "load_oracle",
        lambda path=None: type("O", (), {"hash": "a-different-oracle"})(),
    )
    with pytest.raises(ContractViolation, match="oracle"):
        require_verified(directory, green_diff="the diff")


def test_a_failing_verdict_renders_every_differing_path() -> None:
    from perfagent.contracts import FieldDiff

    verdict = EquivalenceVerdict(
        passed=False, compared=12, identical=10,
        diffs=[
            FieldDiff(request_id="search-a-p0-s1", path="$.content[0].outstanding",
                      before=177949687.02, after=177949687.0),
            FieldDiff(request_id="held-0", path="$.totalElements",
                      before=788, after=787),
        ],
        db_state_matched=True, external_calls_matched=True,
        held_back_passed=False, oracle_hash="orc",
    )
    output = render_verdict(verdict, {
        "green_url": "http://localhost:8081", "oracle_base_commit": "abc",
        "patch_identity": "pid", "benchmark_requests": ["a"],
        "held_back_requests": ["h"],
    })

    assert "FAILED" in output
    # The exact path and both values, which is what a retry is built from.
    assert "$.content[0].outstanding" in output
    assert "177949687.02" in output
    assert "$.totalElements" in output
    assert "held-0" in output
    assert "held-back set  : 1 request(s) — FAILED" in output


def test_leaf_diffs_collapse_to_path_shapes() -> None:
    """One changed string inside a per-row loop produced 111,407 leaf diffs of a
    single shape. The shape is the actionable fact; the leaves are noise, and
    they do not fit in a model's context."""
    from perfagent.contracts import FieldDiff

    diffs = [
        FieldDiff(request_id=f"req-{r}", path=f"$.content[{r}].tags[{i}]",
                  before="overdue", after="OVERDUE")
        for r in range(3) for i in range(500)
    ]
    shapes = summarise_diffs(diffs)

    assert len(shapes) == 1
    assert shapes[0]["shape"] == "$.content[*].tags[*]"
    assert shapes[0]["occurrences"] == 1500
    assert len(shapes[0]["requests"]) == 3
    assert shapes[0]["example_before"] == "overdue"


def test_path_shape_normalises_indices() -> None:
    assert path_shape("$.content[3].tags[7]") == "$.content[*].tags[*]"
    assert path_shape("$.totalElements") == "$.totalElements"


def test_shapes_are_ordered_commonest_first() -> None:
    from perfagent.contracts import FieldDiff

    diffs = [FieldDiff(request_id="r", path="$.rare", before=1, after=2)]
    diffs += [
        FieldDiff(request_id="r", path=f"$.items[{i}].x", before=1, after=2)
        for i in range(9)
    ]
    shapes = summarise_diffs(diffs)
    assert [e["shape"] for e in shapes] == ["$.items[*].x", "$.rare"]


def test_a_huge_diff_set_does_not_print_every_leaf() -> None:
    from perfagent.contracts import EquivalenceVerdict, FieldDiff

    diffs = [
        FieldDiff(request_id="r", path=f"$.content[{i}].tags[0]",
                  before="overdue", after="OVERDUE")
        for i in range(5000)
    ]
    verdict = EquivalenceVerdict(
        passed=False, compared=12, identical=2, diffs=diffs,
        db_state_matched=True, external_calls_matched=True,
        held_back_passed=False, oracle_hash="orc",
    )
    document = {
        "green_url": "u", "oracle_base_commit": "c", "patch_identity": "p",
        "benchmark_requests": ["a"], "held_back_requests": ["h"],
        "diff_shapes": summarise_diffs(diffs), "diffs_total": len(diffs),
        "diffs_persisted": 200,
    }
    output = render_verdict(verdict, document)

    assert "$.content[*].tags[*]" in output
    assert "5000 occurrence(s)" in output
    assert "further leaf difference(s) not printed" in output
    # Readable, not a five-thousand-line dump.
    assert len(output.splitlines()) < 60

    everything = render_verdict(verdict, document, all_diffs=True)
    assert len(everything.splitlines()) > 5000


def test_stale_bytecode_is_refused(tmp_path: Path) -> None:
    """A real bite: --no-build after editing source meant the arm served the
    previous bytecode, so the verdict described code that was no longer there
    while the source-derived patch identity said otherwise."""
    import os
    import time

    arm = tmp_path / "green"
    (arm / "target/classes/com/ex").mkdir(parents=True)
    (arm / "src/main/java/com/ex").mkdir(parents=True)

    compiled = arm / "target/classes/com/ex/Application.class"
    compiled.write_bytes(b"\xca\xfe\xba\xbe")
    source = arm / "src/main/java/com/ex/Application.java"
    source.write_text("class Application {}\n")

    # Source older than classes: fine.
    old = time.time() - 100
    os.utime(source, (old, old))
    assert_build_current(arm)

    # Source newer than classes: refused, and it names the file.
    newer = time.time() + 100
    os.utime(source, (newer, newer))
    with pytest.raises(ContractViolation, match="newer than the"):
        assert_build_current(arm)


def test_a_never_built_arm_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ContractViolation, match="never been built"):
        assert_build_current(tmp_path / "nope")
