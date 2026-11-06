"""P15 tests: the six gates, the evidence ledger, and the narrative filter.

No network. The narrative tests use a stub client, so the one LLM call in the
phase is exercised without making it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from perfagent.contracts import (ArtifactRef, CandidatePlan, EquivalenceVerdict,
                                 FieldDiff, GateName, Hypothesis, Layer,
                                 LatencyStats, Patch, PerfReport, RiskTier,
                                 TerminalState)
from perfagent.errors import ContractViolation
from perfagent.gate.rules import (MAX_CHANGED_LINES, MAX_P_VALUE,
                                  P95_IMPROVEMENT_PCT, classify_mechanism,
                                  evaluate_gates, mapping_paths,
                                  record_signatures, terminal_state_for,
                                  touched_files)
from perfagent.ids import content_hash
from perfagent.report.narrative import (find_violations, strip_measurements,
                                        write_narrative)
from perfagent.report.render import Backed, EvidenceLedger


# ──────────────────────────────  fixtures  ───────────────────────────────


def stats(p50: float, p95: float) -> LatencyStats:
    return LatencyStats(n=72, p50_ms=p50, p95_ms=p95, p99_ms=p95 * 1.1,
                        mean_ms=p50 * 0.7, stdev_ms=p50 * 0.5)


def verdict(passed: bool = True, held_back: bool = True) -> EquivalenceVerdict:
    return EquivalenceVerdict(
        passed=passed, compared=12, identical=12 if passed else 2,
        diffs=[] if passed else [
            FieldDiff(request_id="r", path="$.x", before=1, after=2)
        ],
        db_state_matched=True, external_calls_matched=True,
        held_back_passed=held_back, oracle_hash="orc",
    )


def perf(
    p95_before: float = 1367.1, p95_after: float = 1417.2,
    sql_before: int = 53, sql_after: int = 53,
    rows_before: int = 1_120_858, rows_after: int = 1_120_858,
) -> PerfReport:
    delta = 100.0 * (p95_after - p95_before) / p95_before
    return PerfReport(
        before=stats(1022.0, p95_before), after=stats(1068.2, p95_after),
        p50_delta_pct=4.5, p95_delta_pct=round(delta, 3),
        sql_count_before=sql_before, sql_count_after=sql_after,
        rows_examined_before=rows_before, rows_examined_after=rows_after,
        p_value=0.359, mechanism_confirmed=sql_after < sql_before,
        mechanism_note="n",
    )


DIFF = """diff --git a/src/main/java/com/ex/dto/CustomerDto.java b/src/main/java/com/ex/dto/CustomerDto.java
--- a/src/main/java/com/ex/dto/CustomerDto.java
+++ b/src/main/java/com/ex/dto/CustomerDto.java
@@ -1,7 +1,9 @@
+import com.fasterxml.jackson.databind.annotation.JsonSerialize;
+@JsonSerialize(using = CustomerDto.CustomerDtoSerializer.class)
 public record CustomerDto(Long id, String name, BigDecimal outstanding, List<String> tags) {
"""


def patch(diff: str = DIFF, added: int = 22, removed: int = 0) -> Patch:
    return Patch(
        plan_id="h1", branch="perfagent/p12-004", diff=diff,
        files_changed=["src/main/java/com/ex/dto/CustomerDto.java"],
        lines_added=added, lines_removed=removed, build_iterations=4,
        tests_passed=0, tests_before=0,
    )


def plan(mechanism: str = "hand-coded JsonSerializer replaces reflection",
         files: list[str] | None = None) -> CandidatePlan:
    return CandidatePlan(
        id="h1", targets_hypothesis="h1", description="custom serializer",
        mechanism_changed=mechanism, predicted_effect="less CPU per row",
        files_to_touch=files if files is not None
        else ["src/main/java/com/ex/dto/CustomerDto.java"],
        risk_tier=RiskTier.MEDIUM, what_could_break=["drift"],
    )


def result_for(decision, name: GateName):
    return next(r for r in decision.results if r.name == name)


# ────────────────────────────  the six gates  ────────────────────────────


def test_the_current_run_fails_on_improvement_and_mechanism() -> None:
    """The real artifacts: behaviour preserved, but the patch is slower."""
    decision = evaluate_gates(
        verdict(), perf(), patch(), plan(), paired_p_value=0.0014920851506249918
    )
    outcome = {r.name: r.passed for r in decision.results}

    assert outcome[GateName.BEHAVIOUR_PRESERVED] is True
    assert outcome[GateName.IMPROVEMENT_THRESHOLD] is False
    assert outcome[GateName.STATISTICAL_SIGNIFICANCE] is True
    assert outcome[GateName.DIFF_MINIMALITY] is True
    assert outcome[GateName.MECHANISM_MATCH] is False
    assert outcome[GateName.NO_FORBIDDEN_CHANGE] is True

    assert decision.passed is False
    assert decision.decided_by == "code"
    assert terminal_state_for(decision) == TerminalState.NO_SIGNIFICANT_IMPROVEMENT


def test_every_gate_reports_a_threshold_and_an_actual_value() -> None:
    """The report prints the comparison so a reader need not trust it."""
    decision = evaluate_gates(verdict(), perf(), patch(), plan(), paired_p_value=0.5)
    for result in decision.results:
        assert result.threshold, f"{result.name} has no threshold"
        assert result.actual, f"{result.name} has no actual value"
        assert result.detail


def test_a_behaviour_change_fails_the_first_gate_and_picks_its_state() -> None:
    decision = evaluate_gates(
        verdict(passed=False), perf(p95_after=900.0), patch(), plan(),
        paired_p_value=0.0001,
    )
    assert result_for(decision, GateName.BEHAVIOUR_PRESERVED).passed is False
    assert decision.passed is False
    # A behaviour change makes every other verdict irrelevant.
    assert terminal_state_for(decision) == TerminalState.BEHAVIOUR_NOT_PRESERVED


def test_a_held_back_failure_alone_fails_behaviour_preserved() -> None:
    """This is the anti-gaming arm: the benchmark set can look perfect."""
    decision = evaluate_gates(
        verdict(passed=True, held_back=False), perf(), patch(), plan(),
        paired_p_value=0.001,
    )
    assert result_for(decision, GateName.BEHAVIOUR_PRESERVED).passed is False


@pytest.mark.parametrize(
    "p95_after,expected",
    [(1367.1 * 0.84, True), (1367.1 * 0.86, False), (1367.1, False)],
)
def test_improvement_threshold_is_fifteen_percent(p95_after, expected) -> None:
    decision = evaluate_gates(
        verdict(), perf(p95_after=p95_after), patch(), plan(), paired_p_value=0.001
    )
    assert result_for(decision, GateName.IMPROVEMENT_THRESHOLD).passed is expected


def test_significance_uses_the_paired_p_value_not_the_unpaired_one() -> None:
    """PerfReport.p_value is Mann-Whitney = 0.359 on the real run; the paired
    value is 0.0015. Gating on the unpaired one would pass regressions."""
    report = perf()
    assert report.p_value > MAX_P_VALUE          # unpaired says "noise"

    decision = evaluate_gates(
        verdict(), report, patch(), plan(), paired_p_value=0.0015
    )
    significance = result_for(decision, GateName.STATISTICAL_SIGNIFICANCE)
    assert significance.passed is True
    assert "Wilcoxon" in significance.threshold
    assert "0.001500" in significance.actual


def test_a_missing_paired_p_value_is_not_a_pass() -> None:
    decision = evaluate_gates(
        verdict(), perf(), patch(), plan(), paired_p_value=None
    )
    result = result_for(decision, GateName.STATISTICAL_SIGNIFICANCE)
    assert result.passed is False
    assert "not computed" in result.actual


def test_significance_is_direction_blind_and_says_so() -> None:
    """A significant REGRESSION passes this gate; IMPROVEMENT_THRESHOLD is what
    carries direction. The detail text has to make that explicit or a passing
    row reads as good news."""
    decision = evaluate_gates(
        verdict(), perf(p95_after=2000.0), patch(), plan(), paired_p_value=0.0001
    )
    assert result_for(decision, GateName.STATISTICAL_SIGNIFICANCE).passed is True
    assert result_for(decision, GateName.IMPROVEMENT_THRESHOLD).passed is False
    detail = result_for(decision, GateName.STATISTICAL_SIGNIFICANCE).detail
    assert "not direction" in detail


def test_a_file_outside_the_plan_fails_minimality() -> None:
    decision = evaluate_gates(
        verdict(), perf(), patch(), plan(files=["src/main/java/com/ex/Other.java"]),
        paired_p_value=0.001,
    )
    result = result_for(decision, GateName.DIFF_MINIMALITY)
    assert result.passed is False
    assert "outside the plan" in result.detail


def test_a_touched_test_file_fails_minimality_and_forbidden_change() -> None:
    test_diff = DIFF + (
        "diff --git a/src/test/java/com/ex/CustomerSearchIT.java "
        "b/src/test/java/com/ex/CustomerSearchIT.java\n"
        "+++ b/src/test/java/com/ex/CustomerSearchIT.java\n"
        "-    void searchReturnsOk() {\n"
    )
    decision = evaluate_gates(
        verdict(), perf(), patch(diff=test_diff), plan(), paired_p_value=0.001
    )
    assert result_for(decision, GateName.DIFF_MINIMALITY).passed is False
    assert result_for(decision, GateName.NO_FORBIDDEN_CHANGE).passed is False
    assert "test file" in result_for(decision, GateName.NO_FORBIDDEN_CHANGE).detail


def test_a_large_diff_fails_minimality() -> None:
    decision = evaluate_gates(
        verdict(), perf(), patch(added=MAX_CHANGED_LINES, removed=5), plan(),
        paired_p_value=0.001,
    )
    result = result_for(decision, GateName.DIFF_MINIMALITY)
    assert result.passed is False
    assert "not a minimal fix" in result.detail


def test_a_predicted_query_reduction_must_actually_reduce_queries() -> None:
    """Right answer, wrong reason: latency fell but the query count did not."""
    sql_plan = plan(mechanism="two queries per row replaced by one fetch join")
    assert classify_mechanism(sql_plan) == "sql_count"

    unmoved = evaluate_gates(
        verdict(), perf(p95_after=900.0), patch(), sql_plan, paired_p_value=0.0001
    )
    mechanism = result_for(unmoved, GateName.MECHANISM_MATCH)
    assert mechanism.passed is False
    assert "did not drop" in mechanism.detail
    # Everything else passed, so this gate alone is what stops the ship.
    assert unmoved.passed is False
    assert terminal_state_for(unmoved) == TerminalState.BOTTLENECK_NOT_FIXABLE

    moved = evaluate_gates(
        verdict(), perf(p95_after=900.0, sql_after=2), patch(), sql_plan,
        paired_p_value=0.0001,
    )
    assert result_for(moved, GateName.MECHANISM_MATCH).passed is True
    assert moved.passed is True


def test_an_uninstrumented_mechanism_fails_closed() -> None:
    """A serialization win cannot be confirmed because nothing measures it. The
    honest answer is a failed gate that says so, not a pass by default."""
    result = result_for(
        evaluate_gates(verdict(), perf(p95_after=900.0), patch(), plan(),
                       paired_p_value=0.0001),
        GateName.MECHANISM_MATCH,
    )
    assert result.passed is False
    assert "nothing in this harness measures that" in result.detail


def test_schema_ddl_is_forbidden_but_an_added_index_is_not() -> None:
    for ddl, allowed in (
        ("+ALTER TABLE customer ADD COLUMN foo VARCHAR(10);", False),
        ("+DROP TABLE invoice;", False),
        ("+CREATE INDEX idx_customer_name ON customer(name);", True),
        ("+ALTER TABLE invoice ADD INDEX idx_cust (customer_id);", True),
    ):
        diff = DIFF + f"+++ b/src/main/resources/schema.sql\n{ddl}\n"
        decision = evaluate_gates(
            verdict(), perf(), patch(diff=diff),
            plan(files=["src/main/java/com/ex/dto/CustomerDto.java",
                        "src/main/resources/schema.sql"]),
            paired_p_value=0.001,
        )
        result = result_for(decision, GateName.NO_FORBIDDEN_CHANGE)
        assert result.passed is allowed, f"{ddl!r} -> {result.detail}"


def test_a_changed_request_mapping_is_a_forbidden_api_change() -> None:
    diff = DIFF + (
        '-    @PostMapping("/search")\n'
        '+    @PostMapping("/search/v2")\n'
    )
    result = result_for(
        evaluate_gates(verdict(), perf(), patch(diff=diff), plan(),
                       paired_p_value=0.001),
        GateName.NO_FORBIDDEN_CHANGE,
    )
    assert result.passed is False
    assert "request mapping changed" in result.detail


def test_a_changed_response_record_is_a_forbidden_api_change() -> None:
    diff = (
        "+++ b/src/main/java/com/ex/dto/CustomerDto.java\n"
        "-public record CustomerDto(Long id, String name, BigDecimal outstanding, List<String> tags) {\n"
        "+public record CustomerDto(Long id, String name) {\n"
    )
    result = result_for(
        evaluate_gates(verdict(), perf(), patch(diff=diff), plan(),
                       paired_p_value=0.001),
        GateName.NO_FORBIDDEN_CHANGE,
    )
    assert result.passed is False
    assert "response shape changed" in result.detail


def test_the_real_patch_keeps_the_record_signature_intact() -> None:
    """It annotates the record without changing its components, which is why
    the oracle passed."""
    added = [l[1:] for l in DIFF.splitlines() if l.startswith("+") and not l.startswith("+++")]
    assert record_signatures(added) == {}


def test_every_gate_runs_even_after_one_fails() -> None:
    """A report that stops at the first failure hides the other five."""
    decision = evaluate_gates(
        verdict(passed=False), perf(), patch(diff="+++ b/src/test/x.java\n"),
        plan(files=[]), paired_p_value=None,
    )
    assert len(decision.results) == 6
    assert {r.name for r in decision.results} == set(GateName)


def test_no_argument_can_make_a_failing_gate_pass() -> None:
    """Thresholds are module constants, not parameters. The only inputs are
    evidence."""
    import inspect

    signature = inspect.signature(evaluate_gates)
    assert list(signature.parameters) == [
        "verdict", "perf_report", "patch", "plan", "paired_p_value"
    ]
    for forbidden in ("threshold", "force", "override", "skip", "allow"):
        assert not any(forbidden in name for name in signature.parameters)


def test_gate_decision_cannot_claim_a_model_decided() -> None:
    from pydantic import ValidationError

    from perfagent.contracts import GateDecision
    with pytest.raises(ValidationError, match="decided_by"):
        GateDecision(results=[], passed=True, decided_by="model")


def test_all_passing_gates_yield_pr_opened_only_when_actually_shipped() -> None:
    decision = evaluate_gates(
        verdict(), perf(p95_after=900.0, sql_after=2), patch(),
        plan(mechanism="one fetch join replaces the per-row query"),
        paired_p_value=0.0001,
    )
    assert decision.passed is True
    assert terminal_state_for(decision, shipped=True) == TerminalState.PR_OPENED
    # Nothing pushed means no PR_OPENED, however good the numbers are -- and
    # specifically READY_TO_SHIP, not the pre-fix fallthrough of
    # BOTTLENECK_NOT_FIXABLE, which is a FAILURE state and would make a
    # passing, unshipped decision indistinguishable from a real refusal.
    assert terminal_state_for(decision, shipped=False) == TerminalState.READY_TO_SHIP
    assert terminal_state_for(decision, shipped=False) != TerminalState.BOTTLENECK_NOT_FIXABLE


def test_diff_helpers_read_the_diff_not_the_claim() -> None:
    assert touched_files(DIFF) == ["src/main/java/com/ex/dto/CustomerDto.java"]
    assert mapping_paths(['@PostMapping("/search")']) == {"/search"}


# ────────────────────  the evidence ledger — THE test  ───────────────────


@pytest.fixture
def backed_ref(tmp_path: Path) -> ArtifactRef:
    payload = {"p95_ms": 1417.2}
    path = tmp_path / "perf.json"
    path.write_text(json.dumps(payload))
    return ArtifactRef(kind="perf", hash=content_hash(payload), path=str(path))


def test_rendering_an_unbacked_number_raises(backed_ref: ArtifactRef) -> None:
    """THIS TEST IS THE EVIDENCE LEDGER.

    Invariant 4 says no number appears in a report without an artifact behind
    it. That is a property of this renderer, not a habit of its callers, and
    these are the three ways a caller would otherwise smuggle one in.
    """
    ledger = EvidenceLedger()

    # 1. A slot passed as a bare value, with no artifact at all.
    with pytest.raises(ContractViolation, match="bare float"):
        ledger.render("p95 was {p95} ms", p95=1417.2)

    # 2. A slot with no entry on the ledger.
    with pytest.raises(ContractViolation, match="no artifact behind them"):
        ledger.render("p95 was {p95} ms")

    # 3. A number hardcoded in the template's own prose, which would sail past
    #    every check on the slots.
    ledger.record("p95", 1417.2, backed_ref)
    with pytest.raises(ContractViolation, match="hardcoded numeral"):
        ledger.render("the fix made it 40% faster, p95 now {p95} ms")

    # And the backed path works.
    assert ledger.render("p95 was {p95} ms") == "p95 was 1417.2 ms"


def test_a_citation_that_does_not_resolve_is_refused(tmp_path: Path) -> None:
    ghost = ArtifactRef(kind="perf", hash="abc", path=str(tmp_path / "gone.json"))
    with pytest.raises(ContractViolation, match="does not exist"):
        EvidenceLedger().record("p95", 1417.2, ghost)


def test_an_artifact_edited_after_citation_is_refused(
    tmp_path: Path, backed_ref: ArtifactRef
) -> None:
    """A number quoted from an artifact that has since changed is a number that
    no longer exists."""
    ledger = EvidenceLedger()
    ledger.record("p95", 1417.2, backed_ref)

    Path(backed_ref.path).write_text(json.dumps({"p95_ms": 999.9}))
    with pytest.raises(ContractViolation, match="changed since it was cited"):
        ledger.record("p95_again", 1417.2, backed_ref)


def test_terminal_state_must_be_backed_by_the_gate_decision_ref(
    tmp_path: Path, backed_ref: ArtifactRef
) -> None:
    """The exact bug this guard exists to prevent: `terminal_state` backed by
    the perf report (or any artifact other than the GateDecision that
    actually determined it) once let a rendered report state a terminal
    state its cited artifact never produced."""
    ledger = EvidenceLedger()
    with pytest.raises(ContractViolation, match="gate_decision"):
        ledger.record("terminal_state", "READY_TO_SHIP", backed_ref)

    decision_payload = {"passed": True, "results": []}
    decision_path = tmp_path / "gate_decision.json"
    decision_path.write_text(json.dumps(decision_payload))
    decision_ref = ArtifactRef(
        kind="gate_decision", hash=content_hash(decision_payload),
        path=str(decision_path),
    )
    # The correct ref is accepted without complaint.
    ledger.record("terminal_state", "READY_TO_SHIP", decision_ref)
    assert ledger.get("terminal_state").value == "READY_TO_SHIP"


def test_digits_inside_identifiers_are_allowed(backed_ref: ArtifactRef) -> None:
    """`p95` and `h1` are labels, not claims."""
    ledger = EvidenceLedger()
    ledger.record("value", 1.0, backed_ref)
    assert "p95" in ledger.render("gate p95 for h1 in CustomerDto2: {value}")


def test_record_from_pulls_values_out_of_the_cited_artifact(
    backed_ref: ArtifactRef,
) -> None:
    ledger = EvidenceLedger()
    ledger.record_from(backed_ref, {"p95_ms": 1417.2}, {"p95": "p95_ms"})
    assert ledger.get("p95").value == 1417.2
    assert ledger.get("p95").ref is backed_ref

    with pytest.raises(ContractViolation, match="no value at"):
        ledger.record_from(backed_ref, {"p95_ms": 1.0}, {"nope": "not.there"})


def test_provenance_lists_every_claim_with_its_artifact(
    backed_ref: ArtifactRef,
) -> None:
    ledger = EvidenceLedger()
    ledger.record("p95", 1417.2, backed_ref)
    ledger.record("p50", 1068.2, backed_ref)
    provenance = ledger.provenance()
    assert [e["claim"] for e in provenance] == ["p50", "p95"]
    assert all(e["hash"] == backed_ref.hash for e in provenance)


def test_the_shipped_report_template_has_no_literal_numbers() -> None:
    """The template is where a hardcoded figure would hide most easily."""
    from perfagent.report.render import _HEADER, _SLOT, _BARE_NUMERAL

    literal = _SLOT.sub("", _HEADER)
    offenders = [l for l in literal.splitlines() if _BARE_NUMERAL.search(l)]
    assert offenders == []


# ────────────────────────────  the narrative  ────────────────────────────


class StubClient:
    """Stands in for the one LLM call."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def complete(self, *, messages, tools=None, max_tokens=4096):
        self.calls.append(messages)

        class Response:
            def __init__(self, text: str) -> None:
                self._text = text

            def text(self) -> str:
                return self._text

        return Response(self.reply)


def a_hypothesis() -> Hypothesis:
    return Hypothesis(
        id="h1", claim="JSON serialization dominates the non-SQL latency",
        mechanism="each row is mapped to a DTO and written to JSON by Jackson",
        layer=Layer.SERIALIZATION,
        evidence=["body size correlates with latency"],
        attributed_ms=1400.0, prediction="a slimmer DTO would be faster",
        confidence=0.85,
    )


def test_a_clean_paragraph_is_used_as_written() -> None:
    clean = (
        "Each result row is enriched individually, so the endpoint pays the cost "
        "of building an object graph and then serialising it once per row. The "
        "work therefore scales with the size of the page rather than with the "
        "cost of the underlying query, which is why a request returning no rows "
        "is fast while one returning a handful is slow."
    )
    client = StubClient(clean)
    text, trace = write_narrative(a_hypothesis(), plan(), client=client)

    assert trace["source"] == "model"
    assert trace["violations"] == []
    assert find_violations(text) == []
    assert len(client.calls) == 1, "the phase makes exactly ONE LLM call"


def test_one_stray_figure_is_stripped_and_the_prose_kept() -> None:
    client = StubClient(
        "Each result row triggers its own enrichment pass, so the endpoint pays "
        "for building and serialising an object graph once per row, which is "
        "roughly 40% of the request. The cost therefore scales with page size "
        "rather than with the underlying query, and a request returning no rows "
        "stays fast."
    )
    text, trace = write_narrative(a_hypothesis(), plan(), client=client)

    assert trace["source"] == "model-stripped"
    assert find_violations(text) == []
    assert "%" not in text
    assert "scales with page size" in text


def test_a_paragraph_full_of_numbers_is_rejected_for_the_fallback() -> None:
    """Stripping five figures leaves damaged prose; the deterministic paragraph
    is better than that."""
    client = StubClient(
        "The endpoint takes 1417.2 ms because CustomerService.java issues 2 "
        "queries per row across 50000 customers and 1200000 invoices, a 40% "
        "overhead."
    )
    text, trace = write_narrative(a_hypothesis(), plan(), client=client)

    assert trace["source"] == "fallback"
    assert find_violations(text) == []
    assert "CustomerService.java" not in text


def test_the_narrative_never_contains_a_number_whatever_the_model_says() -> None:
    """The property that matters, over every failure mode."""
    for reply in (
        "It is 40% slower.",
        "Roughly 1400 ms is spent in serialization per request.",
        "See src/main/java/com/ex/service/CustomerService.java line 38.",
        "",
        "2",
    ):
        text, trace = write_narrative(a_hypothesis(), plan(), client=StubClient(reply))
        assert find_violations(text) == [], f"reply {reply!r} produced {text!r}"


def test_an_empty_reply_is_not_accepted_as_a_narrative() -> None:
    """This happened on the first real run: the provider returned its text in
    `reasoning` and left `content` blank, so a clean trace produced a report
    with an empty Root cause section. Compliance is necessary, not sufficient."""
    for reply in ("", "   ", "The endpoint is slow.", "Serialization dominates."):
        text, trace = write_narrative(a_hypothesis(), plan(), client=StubClient(reply))
        assert trace["source"] == "fallback", f"{reply!r} was accepted"
        assert len(text.split()) >= 25
        assert find_violations(text) == []


def test_a_failing_llm_still_produces_a_report_paragraph() -> None:
    class Broken:
        def complete(self, **kwargs):
            raise RuntimeError("rate limited")

    text, trace = write_narrative(a_hypothesis(), plan(), client=Broken())
    assert trace["source"] == "fallback"
    assert "rate limited" in trace["reason"]
    assert len(text.split()) > 20


def test_the_prompt_forbids_numbers_explicitly() -> None:
    client = StubClient("fine")
    write_narrative(a_hypothesis(), plan(), client=client)
    prompt = client.calls[0][0]["content"]
    assert "NO numbers" in prompt
    assert "NO file paths" in prompt


def test_strip_removes_percent_signs_cleanly() -> None:
    """A trailing \\b after % never matches before a space, so an earlier
    version left a stray '%' behind."""
    assert "%" not in strip_measurements("it was 40% faster")
    assert "%" not in strip_measurements("it was 40 % faster")


# ───────────────  PR body: pre-existing risks and excluded tables  ───────


def test_pre_existing_rows_renders_every_uncertainty() -> None:
    from perfagent.ship import _pre_existing_rows

    rendered = _pre_existing_rows([
        "AUTOMATED CHECK: pagination is offset-based and no ORDER BY was found.",
        "unsure whether page size defaults to 50 or 20",
    ])
    assert rendered.count("\n- ") == 1          # two bullets, one join
    assert rendered.startswith("- AUTOMATED CHECK")
    assert "unsure whether page size" in rendered


def test_pre_existing_rows_says_so_when_there_are_none() -> None:
    from perfagent.ship import _pre_existing_rows

    assert _pre_existing_rows([]) == "- (none recorded)"
