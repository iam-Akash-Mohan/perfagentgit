from __future__ import annotations


class PerfAgentError(Exception):
    """Base class for every error this package raises."""


class ToolError(PerfAgentError):
    """Expected tool failure. The agent loop turns this into an is_error
    tool_result and continues. Message must be actionable for the model."""

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.message}\nSuggestion: {self.suggestion}"
        return self.message


class GuardrailError(ToolError):
    """A CONTAINMENT check refused the call: a path escape, a denied write.

    A subclass rather than a flag because the agent loop has to tell the two
    apart in the trace. An ordinary ToolError is the model making a mistake it
    can fix; this one is the model reaching for something it is not allowed to
    have, and those need different colours in a post-mortem.
    """


class BudgetExhausted(PerfAgentError):
    """The run exceeded its token / dollar budget."""


class ContractViolation(PerfAgentError):
    """An artifact failed validation, or a guardrail was breached."""
