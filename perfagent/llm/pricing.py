"""Per-model token prices, in US dollars per million tokens.

A price is configuration, not a measurement (invariant 4 cuts both ways): the
numbers here are list prices at the time of writing and they go stale. Tokens
are always reported alongside dollars so a wrong rate here can never hide the
underlying fact.

An unknown model prices at zero and says so, rather than crashing a run or —
worse — silently inventing a plausible rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..contracts import Usage

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    output: float
    # Anthropic bills cache reads at a discount and cache writes at a premium.
    # Providers without a cache leave these None and they fall back to `input`.
    cache_read: float | None = None
    cache_write: float | None = None


# Keys are matched exactly first, then by longest prefix, so a dated model id
# like "claude-opus-4-5-20260101" resolves to its family entry.
PRICES: dict[str, Price] = {
    # ── Anthropic ────────────────────────────────────────────────────────
    "claude-opus-4-5": Price(5.00, 25.00, cache_read=0.50, cache_write=6.25),
    "claude-sonnet-4-5": Price(3.00, 15.00, cache_read=0.30, cache_write=3.75),
    "claude-haiku-4-5": Price(1.00, 5.00, cache_read=0.10, cache_write=1.25),
    # ── Cerebras ─────────────────────────────────────────────────────────
    "qwen-3-coder-480b": Price(2.00, 2.00),
    "qwen-3-235b-a22b-instruct-2507": Price(0.60, 1.20),
    "llama-4-scout-17b-16e-instruct": Price(0.65, 0.85),
    "gpt-oss-120b": Price(0.25, 0.69),
    # ── Gemini (Google list prices, paid tier, text tokens) ──────────────
    "gemini-2.5-flash": Price(0.30, 2.50),
    "gemini-2.5-pro": Price(1.25, 10.00),
    "gemini-2.0-flash": Price(0.10, 0.40),
    # ── Groq (its ids are vendor-prefixed, and prefix matching is
    #    longest-first, so these need their own entries) ────────────────────
    "openai/gpt-oss-120b": Price(0.15, 0.75),
    "openai/gpt-oss-20b": Price(0.10, 0.50),
    "qwen/qwen3.8-27b": Price(0.30, 0.60),
    "llama-3.3-70b-versatile": Price(0.59, 0.79),
    "moonshotai/kimi-k2-instruct": Price(1.00, 3.00),
    # ── Ollama (local; the electricity is not billed to the run) ─────────
    "ollama": Price(0.0, 0.0),
}

_FREE = Price(0.0, 0.0)

# Models we have already warned about. Warning once per model per process is
# informative; warning on every turn of a thirty-step loop is noise.
_warned: set[str] = set()


def price_for(model: str) -> Price | None:
    """The price entry for a model, or None if we have never heard of it."""
    if model in PRICES:
        return PRICES[model]
    candidates = [key for key in PRICES if model.startswith(key)]
    if candidates:
        return PRICES[max(candidates, key=len)]
    return None


def usd_for(model: str, usage: Usage) -> float:
    """Cost of one call. Unknown model → 0.0, with a warning that says so."""
    price = price_for(model)
    if price is None:
        if model not in _warned:
            _warned.add(model)
            log.warning(
                "no price is configured for model %r — this run's dollar "
                "figures will read 0.00 for it. Token counts are still exact. "
                "Add it to perfagent/llm/pricing.py to fix the cost report.",
                model,
            )
        price = _FREE

    cache_read = price.cache_read if price.cache_read is not None else price.input
    cache_write = price.cache_write if price.cache_write is not None else price.input

    return (
        usage.input_tokens / 1_000_000 * price.input
        + usage.output_tokens / 1_000_000 * price.output
        + usage.cache_read_tokens / 1_000_000 * cache_read
        + usage.cache_write_tokens / 1_000_000 * cache_write
    )


def is_known(model: str) -> bool:
    return price_for(model) is not None
