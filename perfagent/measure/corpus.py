from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from ..contracts import RequestCorpus, RequestSpec
from ..errors import ContractViolation
from ..ids import content_hash

DEFAULT_CORPUS_PATH = Path(__file__).with_name("corpus.json")


def corpus_hash(
    benchmark: list[RequestSpec], held_back: list[RequestSpec] | None = None
) -> str:
    """Content hash of the request set.

    Every measurement is stamped with this. Two runs whose corpus hashes differ
    are not comparable, and a corpus edited between a before and an after run is
    detectable rather than silently accepted.
    """
    return content_hash(
        {
            "benchmark": [s.model_dump(mode="json") for s in benchmark],
            "held_back": [s.model_dump(mode="json") for s in (held_back or [])],
        }
    )


def load_corpus(path: str | Path | None = None) -> RequestCorpus:
    """Load a RequestCorpus from JSON.

    The file carries `benchmark` (and optionally `held_back`); the hash is
    computed here rather than trusted. If the file states a hash that disagrees
    with its own content, that is a ContractViolation — the corpus was edited
    without being re-stamped.
    """
    corpus_path = Path(path) if path is not None else DEFAULT_CORPUS_PATH
    if not corpus_path.exists():
        raise ContractViolation(f"corpus file not found: {corpus_path}")

    try:
        payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractViolation(f"{corpus_path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractViolation(f"{corpus_path} must contain a JSON object")

    try:
        benchmark = [RequestSpec.model_validate(r) for r in payload.get("benchmark", [])]
        held_back = [RequestSpec.model_validate(r) for r in payload.get("held_back", [])]
    except ValidationError as exc:
        raise ContractViolation(f"{corpus_path} has a malformed request: {exc}") from exc

    if not benchmark:
        raise ContractViolation(f"{corpus_path} has an empty `benchmark` list")

    ids = [r.id for r in benchmark] + [r.id for r in held_back]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ContractViolation(
            f"{corpus_path} has duplicate request ids: {', '.join(duplicates)}"
        )

    digest = corpus_hash(benchmark, held_back)
    stated = payload.get("hash")
    if stated is not None and stated != digest:
        raise ContractViolation(
            f"{corpus_path} states hash {stated} but its content hashes to {digest}"
        )

    return RequestCorpus(benchmark=benchmark, held_back=held_back, hash=digest)


def save_corpus(corpus: RequestCorpus, path: str | Path) -> Path:
    """Write a corpus back out with a freshly computed hash."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = corpus.model_dump(mode="json")
    payload["hash"] = corpus_hash(corpus.benchmark, corpus.held_back)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out
