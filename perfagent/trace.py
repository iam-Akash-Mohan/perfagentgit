from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(str, Enum):
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ARTIFACT_WRITTEN = "artifact_written"
    GUARDRAIL_TRIGGERED = "guardrail_triggered"
    CHECKPOINT = "checkpoint"
    GATE_EVALUATED = "gate_evaluated"
    TERMINAL_STATE = "terminal_state"
    NOTE = "note"


class Trace:
    """Append-only JSONL event log for one run."""

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "trace.jsonl"
        self._seq = 0

    def emit(self, type: EventType, phase: str, **fields: Any) -> None:
        """Append one JSON line. Auto-adds ts, run, seq. Never raises —
        a broken trace must not kill a run."""
        try:
            record: dict[str, Any] = {
                "ts": time.time(),
                "run": self.run_id,
                "seq": self._seq,
                "type": type.value if isinstance(type, EventType) else str(type),
                "phase": phase,
            }
            record.update(fields)
            line = json.dumps(record, default=repr, ensure_ascii=False)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._seq += 1
        except Exception:  # noqa: BLE001 — a broken trace must not kill a run
            pass

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        events: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def cost_by_phase(self) -> dict[str, float]:
        costs: dict[str, float] = {}
        for event in self.read_all():
            usd = event.get("usd")
            if not isinstance(usd, (int, float)):
                continue
            phase = str(event.get("phase", ""))
            costs[phase] = costs.get(phase, 0.0) + float(usd)
        return costs
