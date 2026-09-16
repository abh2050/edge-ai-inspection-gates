"""These helpers preserve local evidence and refuse changed records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from cycletime.contracts import EvidenceRef
from cycletime.dataio.archive import sha256
from cycletime.dataio.verify import write_json


def digest(value: object) -> str:
    """Hash canonical JSON; refuse values without a JSON representation."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def record(path: Path, value: dict) -> EvidenceRef:
    """Write a timestamped record atomically; refuse nonfinite values."""
    digest(value)
    value = dict(value, created_at_utc=datetime.now(UTC).isoformat(), run_id=str(uuid4()))
    write_json(path, value)
    return reference(path)


def reference(path: Path) -> EvidenceRef:
    """Reference an existing JSON record; refuse missing provenance fields."""
    value = json.loads(path.read_text())
    return EvidenceRef(path, sha256(path), value["created_at_utc"], value["run_id"])


def read(ref: EvidenceRef) -> dict:
    """Read unchanged evidence; refuse a mismatched digest."""
    if sha256(ref.path) != ref.sha256:
        raise ValueError(f"Evidence changed: {ref.path}")
    return json.loads(ref.path.read_text())


def serialize(ref: EvidenceRef) -> dict:
    """Encode an evidence reference; refuse implicit path coercion outside the path field."""
    return dict(asdict(ref), path=str(ref.path))


def restore(value: dict) -> EvidenceRef:
    """Restore and verify a reference; refuse modified evidence."""
    ref = EvidenceRef(**dict(value, path=Path(value["path"])))
    read(ref)
    return ref
