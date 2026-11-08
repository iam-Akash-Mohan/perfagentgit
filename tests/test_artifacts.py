from __future__ import annotations
import pytest
from senapathy.artifacts import ArtifactStore
from senapathy.trace import Trace, EventType
from senapathy.contracts import Citation, EndpointModel
from senapathy.errors import ContractViolation


def _model():
    return EndpointModel(route="POST /x", entry=Citation(file="C.java", line=1),
                         path=[Citation(file="C.java", line=1)])


def test_write_read_roundtrip(tmp_path):
    store = ArtifactStore(tmp_path)
    ref = store.write("endpoint_model", _model())
    assert store.exists("endpoint_model")
    back = store.read(ref, EndpointModel)
    assert back.route == "POST /x"


def test_write_is_content_addressed_and_idempotent(tmp_path):
    store = ArtifactStore(tmp_path)
    a = store.write("endpoint_model", _model())
    b = store.write("endpoint_model", _model())
    assert a.hash == b.hash


def test_tampered_artifact_is_rejected(tmp_path):
    """Artifacts are immutable. If the bytes change, reading must fail —
    this is what makes the evidence ledger trustworthy."""
    store = ArtifactStore(tmp_path)
    ref = store.write("endpoint_model", _model())
    p = tmp_path / "artifacts" / "endpoint_model.json"
    p.write_text(p.read_text().replace("POST /x", "POST /tampered"))
    with pytest.raises(ContractViolation):
        store.read(ref, EndpointModel)


def test_trace_appends_and_never_raises(tmp_path):
    t = Trace(tmp_path, "run1")
    t.emit(EventType.PHASE_START, "P1")
    t.emit(EventType.LLM_RESPONSE, "P1", usd=0.02, tokens_in=100)
    t.emit(EventType.PHASE_END, "P1")
    events = t.read_all()
    assert len(events) == 3
    assert [e["seq"] for e in events] == [0, 1, 2]
    assert all(e["run"] == "run1" for e in events)


def test_trace_cost_by_phase(tmp_path):
    t = Trace(tmp_path, "run1")
    t.emit(EventType.LLM_RESPONSE, "P1", usd=0.02)
    t.emit(EventType.LLM_RESPONSE, "P1", usd=0.03)
    t.emit(EventType.LLM_RESPONSE, "P5", usd=0.10)
    costs = t.cost_by_phase()
    assert round(costs["P1"], 4) == 0.05
    assert round(costs["P5"], 4) == 0.10


def test_trace_survives_unserialisable_field(tmp_path):
    """A broken trace must never kill a run."""
    t = Trace(tmp_path, "run1")
    t.emit(EventType.NOTE, "P1", weird=object())
    assert len(t.read_all()) == 1
