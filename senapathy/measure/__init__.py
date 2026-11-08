"""Deterministic measurement harness.

Nothing in this package calls an LLM. Every number it produces comes from a
real HTTP response or a real performance_schema row, so that a report built on
top of it can point at an artifact for each claim.
"""
from __future__ import annotations

from .bench import run_benchmark
from .corpus import DEFAULT_CORPUS_PATH, corpus_hash, load_corpus, save_corpus
from .sql import MySQLTarget, collect_digests, reset_digests

__all__ = [
    "DEFAULT_CORPUS_PATH",
    "MySQLTarget",
    "collect_digests",
    "corpus_hash",
    "load_corpus",
    "reset_digests",
    "run_benchmark",
    "save_corpus",
]
