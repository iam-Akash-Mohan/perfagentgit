"""One LLM call, for prose only. Numbers come from the ledger, never from here.

The division is the point of invariant 4: the model is good at explaining a
mechanism in a paragraph and bad at being trusted with a figure. So it gets the
evidence, is asked for exactly one paragraph, and its output is then checked by
code for anything that looks like a measurement. A response that quotes a
latency is not edited into shape and used — the offending figures are stripped,
and if the result still fails the check it is rejected in favour of a
deterministic sentence assembled from the artifacts.

That fallback matters: a reporting phase that can be blocked by a chatty model
is a reporting phase that will block on the day it is needed.
"""

from __future__ import annotations

import os
import re
from typing import Any

from ..contracts import CandidatePlan, Hypothesis
from ..errors import ContractViolation

MAX_WORDS = 170

# An empty or near-empty reply satisfies "contains no numbers" trivially, and
# that is exactly what happened on the first real run: the provider returned its
# text in `reasoning` and left `content` blank, so a clean-looking trace
# ("source=model, violations=0") produced a report with an empty Root cause
# section. Compliance is necessary, not sufficient.
MIN_WORDS = 25

# Generous on purpose. This provider's model emits a reasoning trace before its
# answer — measured at ~2000 characters for this prompt — and at a 500-token cap
# the budget ran out mid-reasoning, so `content` came back empty and the report
# rendered with a blank Root cause section.
DEFAULT_MAX_TOKENS = 2000

# What the model may not say, because the ledger says it instead.
# `%` is a non-word character, so a trailing \b after it never matches before a
# space — "40% overhead" left a stray "%" behind. The percent forms therefore
# get their own alternative with no word boundary.
_MEASUREMENT = re.compile(
    r"\d+(?:[.,]\d+)?\s*%"
    r"|\d+(?:[.,]\d+)?\s*(?:ms|s|percent|milliseconds?|seconds?)\b",
    re.I,
)

# How much stripping a paragraph may survive. Removing one stray figure leaves
# readable prose; removing five leaves "The endpoint takes because issues
# queries per row", which is worse than the deterministic fallback.
MAX_STRIPPED_VIOLATIONS = 2
_PATH = re.compile(r"[\w./-]*/[\w./-]+\.\w+|\b\w+\.(?:java|json|xml|yml|yaml|py)\b")
_ANY_DIGIT_RUN = re.compile(r"(?<![A-Za-z_])\d[\d.,]*")


def build_prompt(hypothesis: Hypothesis, plan: CandidatePlan) -> str:
    """The evidence, and a hard prohibition on quoting any of it back."""
    return f"""You are writing ONE paragraph for an engineering report about a performance
investigation that has already been measured and decided. Your job is the
explanation, and only the explanation.

WHAT WAS DIAGNOSED
  claim      {hypothesis.claim}
  layer      {hypothesis.layer.value}
  mechanism  {hypothesis.mechanism}

WHAT WAS CHANGED
  description {plan.description}
  mechanism   {plan.mechanism_changed}
  predicted   {plan.predicted_effect}

WRITE: one paragraph, at most {MAX_WORDS} words, explaining WHY the diagnosed
cause costs time — the chain of cause and effect a reviewer needs in order to
understand the change.

ABSOLUTE PROHIBITIONS. Your paragraph must contain:
  - NO numbers of any kind. Not latencies, not counts, not percentages, not
    row counts, not query counts. None.
  - NO units. Never write ms, seconds, or percent.
  - NO file paths, class names with extensions, or line numbers.

Every figure in this report is templated from a recorded measurement and
inserted by code. If you write a number it will be stripped, and if numbers
remain your paragraph will be discarded entirely and replaced. Describe the
mechanism in words: "one query per result row" rather than any count.

Reply with the paragraph and nothing else. No heading, no bullet list, no
preamble."""


def find_violations(text: str) -> list[str]:
    """Everything in the response that only the ledger is allowed to say."""
    found: list[str] = []
    found.extend(f"measurement: {m.group(0)}" for m in _MEASUREMENT.finditer(text))
    found.extend(f"path: {m.group(0)}" for m in _PATH.finditer(text))
    found.extend(f"numeral: {m.group(0)}" for m in _ANY_DIGIT_RUN.finditer(text))
    return found


def strip_measurements(text: str) -> str:
    """Remove figures, then tidy the punctuation the removal leaves behind."""
    cleaned = _MEASUREMENT.sub("", text)
    cleaned = _PATH.sub("", cleaned)
    cleaned = _ANY_DIGIT_RUN.sub("", cleaned)
    cleaned = re.sub(r"\s*[%]\s*", " ", cleaned)      # any percent sign left over
    cleaned = re.sub(r"\(\s*[,;]?\s*\)", "", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])\s*([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def fallback_narrative(hypothesis: Hypothesis, plan: CandidatePlan) -> str:
    """A deterministic paragraph, assembled from the artifacts.

    Used when the model will not stop quoting figures. Contains no numbers
    because it contains nothing the model wrote.
    """
    return (
        f"The diagnosis identifies the {hypothesis.layer.value.replace('_', ' ')} "
        f"layer as the dominant cost: {_desentence(hypothesis.claim)}. The "
        f"mechanism is that {_desentence(hypothesis.mechanism)}. The change "
        f"under review addresses it by {_desentence(plan.mechanism_changed)}. "
        f"Whether that expectation held is answered by the measurement table "
        f"and the gates below, not by this paragraph."
    )


def _desentence(text: str) -> str:
    """Lower-case the first word so a sentence can be embedded mid-clause —
    unless it is an acronym, because "jSON" is worse than the capital."""
    stripped = strip_measurements(text).strip().rstrip(".")
    if not stripped:
        return "the recorded cause"
    if len(stripped) > 1 and stripped[0].isupper() and stripped[1].isupper():
        return stripped
    return stripped[0].lower() + stripped[1:]


def write_narrative(
    hypothesis: Hypothesis,
    plan: CandidatePlan,
    *,
    client: Any = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[str, dict[str, Any]]:
    """One call. Returns (paragraph, provenance about how it was obtained).

    The provenance is recorded because "the model wrote this" and "code wrote
    this because the model would not comply" are different facts and the reader
    is entitled to know which.
    """
    prompt = build_prompt(hypothesis, plan)
    trace: dict[str, Any] = {"source": "model", "attempts": 0, "violations": []}

    try:
        llm = client or _default_client()
    except ContractViolation as exc:
        trace.update(source="fallback", reason=f"no LLM available: {exc}")
        return fallback_narrative(hypothesis, plan), trace

    raw = ""
    try:
        response = llm.complete(
            messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens
        )
        raw = response.text().strip()
        trace["attempts"] = 1
    except Exception as exc:  # noqa: BLE001 - a report must still be produced
        trace.update(source="fallback", reason=f"{type(exc).__name__}: {exc}")
        return fallback_narrative(hypothesis, plan), trace

    violations = find_violations(raw)
    trace["violations"] = violations
    trace["raw_words"] = len(raw.split())

    if len(raw.split()) < MIN_WORDS:
        trace.update(
            source="fallback",
            reason=f"the model returned {len(raw.split())} word(s), fewer than "
                   f"the {MIN_WORDS} a root-cause paragraph needs",
        )
        return fallback_narrative(hypothesis, plan), trace

    if not violations:
        return _trim(raw), trace

    # Strip first, as specified — a paragraph whose only sin was one figure is
    # worth keeping. But a paragraph that needed heavy surgery is not: what
    # survives reads as damaged prose, and the fallback is better than that.
    distinct = len({v.split(": ", 1)[-1] for v in violations})
    stripped = strip_measurements(raw)
    if (
        not find_violations(stripped)
        and len(stripped.split()) >= MIN_WORDS
        and distinct <= MAX_STRIPPED_VIOLATIONS
    ):
        trace.update(source="model-stripped", stripped_count=distinct)
        return _trim(stripped), trace

    trace.update(
        source="fallback",
        reason=(
            "numbers remained after stripping"
            if find_violations(stripped)
            else f"{distinct} distinct figure(s) removed left the paragraph "
                 "too damaged to publish"
        ),
    )
    return fallback_narrative(hypothesis, plan), trace


def _trim(text: str) -> str:
    words = text.split()
    if len(words) <= MAX_WORDS:
        return " ".join(words)
    return " ".join(words[:MAX_WORDS]).rstrip(",;:") + "."


def _default_client():
    from ..llm import GroqClient

    from dotenv import load_dotenv

    load_dotenv()
    for name in ("CEREBRAS_API_KEY", "GROQ_API_KEY"):
        key = os.environ.get(name)
        if key:
            return GroqClient(api_key=key)
    raise ContractViolation("neither CEREBRAS_API_KEY nor GROQ_API_KEY is set")
