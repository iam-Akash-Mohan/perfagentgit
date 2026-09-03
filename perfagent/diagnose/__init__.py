"""P09 — diagnosis.

Takes the measured evidence produced by the baseline harness and the call
chain produced by comprehension, and turns them into ranked, falsifiable
hypotheses about where the time actually goes.

The model reasons and writes prose. It does not count, sum, compare, or decide
whether its own diagnosis is acceptable — evidence.py computes the facts and
validate.py returns the verdict.
"""
from __future__ import annotations

from .agent import EMIT_HYPOTHESES_TOOL, build_task, render_diagnosis, run_diagnosis
from .evidence import latest_run_dir, load_baseline, load_endpoint_model, render_evidence
from .probe import RUN_PROBE_TOOL, ProbeLog, make_run_probe
from .validate import validate_hypothesis_set

__all__ = [
    "EMIT_HYPOTHESES_TOOL",
    "ProbeLog",
    "RUN_PROBE_TOOL",
    "build_task",
    "latest_run_dir",
    "load_baseline",
    "load_endpoint_model",
    "make_run_probe",
    "render_diagnosis",
    "render_evidence",
    "run_diagnosis",
    "validate_hypothesis_set",
]
