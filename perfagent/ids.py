from __future__ import annotations

import hashlib
import json
from typing import Any

from ulid import ULID


def new_run_id() -> str:
    """A fresh, lexicographically sortable run identifier."""
    return str(ULID())


def content_hash(obj: Any) -> str:
    """sha256 of canonical JSON (sorted keys, no whitespace), first 16 hex chars."""
    canonical = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
