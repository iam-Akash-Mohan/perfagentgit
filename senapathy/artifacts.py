from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .contracts import ArtifactRef
from .errors import ContractViolation
from .ids import content_hash

T = TypeVar("T", bound=BaseModel)


class ArtifactStore:
    """Content-addressed, immutable store for run artifacts."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str) -> Path:
        return self.artifacts_dir / f"{kind}.json"

    def write(self, kind: str, model: BaseModel) -> ArtifactRef:
        """Serialise, content-hash, write to run_dir/artifacts/{kind}.json,
        return ref. Writing the same content twice is idempotent."""
        payload: Any = model.model_dump(mode="json")
        digest = content_hash(payload)
        path = self._path(kind)
        canonical = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
        path.write_text(canonical + "\n", encoding="utf-8")
        return ArtifactRef(kind=kind, hash=digest, path=str(path))

    def read(self, ref: ArtifactRef, model_cls: type[T]) -> T:
        """Read and validate. Raise ContractViolation if the hash doesn't match
        the content — artifacts are immutable once written."""
        path = Path(ref.path)
        if not path.exists():
            raise ContractViolation(f"artifact {ref.kind} missing at {ref.path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ContractViolation(
                f"artifact {ref.kind} at {ref.path} is not valid JSON: {exc}"
            ) from exc

        digest = content_hash(payload)
        if digest != ref.hash:
            raise ContractViolation(
                f"artifact {ref.kind} at {ref.path} was modified: "
                f"expected hash {ref.hash}, found {digest}"
            )

        try:
            return model_cls.model_validate(payload)
        except ValidationError as exc:
            raise ContractViolation(
                f"artifact {ref.kind} does not satisfy {model_cls.__name__}: {exc}"
            ) from exc

    def exists(self, kind: str) -> bool:
        return self._path(kind).exists()
