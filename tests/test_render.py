"""Rendering. Everything here is a pure function of an artifact or an event."""

from __future__ import annotations

from pathlib import Path

from senapathy.contracts import EndpointModel, IndexInfo, TableInfo
from senapathy.render.endpoint import render_endpoint_model
from senapathy.render.transcript import TranscriptRenderer, attach_transcript
from senapathy.trace import EventType, Trace

MODEL = EndpointModel.model_validate({
    "route": "POST /api/customers/search",
    "call_chain": [
        {"method": "CustomerController.search", "file": "C.java", "line": 18,
         "does": "delegates to the service"},
        {"method": "CustomerService.enrich", "file": "S.java", "line": 36,
         "does": "runs once per row"},
    ],
    "data_access": [
        {"trigger": "repo.findByFilter", "kind": "jpql_query",
         "file": "R.java", "line": 13, "executes_per_row": False},
        {"trigger": "lazy load of Customer.invoices", "kind": "lazy_collection",
         "file": "D.java", "line": 23, "executes_per_row": True},
    ],
    "tables": [TableInfo(
        table="invoice", entity="Invoice", columns_read=["amount", "status"],
        all_columns=[f"c{n} int" for n in range(5)],
        existing_indexes=[IndexInfo(name="idx_invoice_customer_id",
                                    columns=["customer_id"], unique=False,
                                    cardinality=28691)],
        approx_rows=1_196_424).model_dump()],
    "uncertainties": ["the default page size"],
    "prompt_hash": "deadbeefdeadbeef",
})


def test_the_per_row_count_is_called_out():
    out = render_endpoint_model(MODEL)
    assert "2 total, 1 per result row" in out
    assert "per row " in out and "1×/request" in out


def test_existing_indexes_are_shown_with_their_cardinality():
    out = render_endpoint_model(MODEL)
    assert "idx_invoice_customer_id(customer_id)" in out
    assert "cardinality 28,691" in out


def test_a_table_with_no_indexes_says_none_rather_than_nothing():
    model = MODEL.model_copy(deep=True)
    model.tables[0].existing_indexes = []
    assert "NO INDEXES" in render_endpoint_model(model)


def test_over_fetching_is_visible():
    assert "reads 2 of 5 columns" in render_endpoint_model(MODEL)


def test_no_declared_uncertainty_is_itself_reported():
    model = MODEL.model_copy(deep=True)
    model.uncertainties = []
    assert "the agent declared none" in render_endpoint_model(model)


def test_the_prompt_version_is_printed():
    assert "prompt version    deadbeefdeadbeef" in render_endpoint_model(MODEL)


# ────────────────────────────  transcript  ───────────────────────────────


def test_the_transcript_is_built_from_trace_events(tmp_path: Path) -> None:
    lines: list[str] = []
    trace = Trace(tmp_path, "r1")
    attach_transcript(trace, lines.append)

    trace.emit(EventType.LLM_REQUEST, phase="understand", step=0,
               tools=["read_file"])
    trace.emit(EventType.LLM_RESPONSE, phase="understand", step=1,
               stop_reason="tool_use", input_tokens=100, output_tokens=20,
               usd=0.002, duration_ms=440, from_cassette=False,
               text="", reasoning="I should read the controller")
    trace.emit(EventType.TOOL_CALL, phase="understand", step=1,
               tool="read_file", args="path='C.java'")
    trace.emit(EventType.TOOL_RESULT, phase="understand", step=1,
               tool="read_file", is_error=False, guardrail=False, bytes=936,
               preview="1| package com.ex;")

    body = "\n".join(lines)
    assert "CONVERSATION START" in body
    assert "tokens in=100 out=20" in body
    assert "MODEL (thinking to itself)" in body
    assert "read_file(path='C.java')" in body
    assert "tool result · 936 chars" in body


def test_a_guardrail_reads_differently_from_an_error(tmp_path: Path) -> None:
    lines: list[str] = []
    renderer = TranscriptRenderer(lines.append)
    renderer.on_event({"type": "tool_result", "tool": "read_file",
                       "is_error": True, "guardrail": True, "preview": "no"})
    renderer.on_event({"type": "tool_result", "tool": "read_file",
                       "is_error": True, "guardrail": False, "preview": "no"})
    body = "\n".join(lines)
    assert "REFUSED by a guardrail" in body
    assert "(tool error)" in body


def test_a_cassette_hit_is_labelled(tmp_path: Path) -> None:
    lines: list[str] = []
    TranscriptRenderer(lines.append).on_event(
        {"type": "llm_response", "step": 3, "stop_reason": "tool_use",
         "input_tokens": 1, "output_tokens": 1, "usd": 0.0, "duration_ms": 0,
         "from_cassette": True})
    assert "FROM CASSETTE" in "\n".join(lines)


def test_a_long_result_is_clipped_for_display_only(tmp_path: Path) -> None:
    lines: list[str] = []
    renderer = TranscriptRenderer(lines.append, result_lines=3)
    renderer.on_event({"type": "tool_result", "tool": "read_file",
                       "is_error": False, "guardrail": False, "bytes": 999,
                       "preview": "\n".join(str(n) for n in range(10))})
    body = "\n".join(lines)
    assert "7 more lines sent to the model, hidden here" in body


def test_a_broken_renderer_cannot_kill_a_run(tmp_path: Path) -> None:
    trace = Trace(tmp_path, "r1")

    def explode(event: dict) -> None:
        raise RuntimeError("renderer bug")

    trace.add_listener(explode)
    trace.emit(EventType.NOTE, phase="understand", note="still fine")
    assert trace.read_all()[-1]["note"] == "still fine"


def test_there_is_no_print_left_in_the_agent_loop() -> None:
    """The transcript moved to render/; library code emits events."""
    source = (Path(__file__).resolve().parent.parent
              / "senapathy" / "agent.py").read_text()
    assert "print(" not in source
