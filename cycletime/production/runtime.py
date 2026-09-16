"""cycletime/production/runtime.py defines the offline inspection boundary for a future station adapter."""
from __future__ import annotations

import io
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from cycletime.contracts import JSON, EvidenceRef
from cycletime.dataio.archive import sha256
from cycletime.evidence import read, record


class StationFault(RuntimeError):
    """Signal the configured controlled fault; the station never defaults to accepting a part."""

    def __init__(self, reason: str, detail: JSON | None = None):
        self.reason = reason
        self.detail = detail or {}
        super().__init__(reason)


def verified_release(release: EvidenceRef) -> JSON:
    """Read a release verification record; refuse anything the manifest module did not verify."""
    record_data = read(release)
    if record_data.get("status") != "verified":
        raise StationFault("the station refuses an unverified release", {"release": str(release.path)})
    return record_data


def inspect_frame(frame: bytes, frame_id: str, captured_at_ns: int, release: EvidenceRef) -> EvidenceRef:
    """Validate a fresh frame and execute its signed deterministic preprocessing, inference, threshold, and result contract locally; include frame identity, score, timing, and artifact hash; refuse LLM calls, stale frames, unverified releases, silent default accepts, and direct machine actuation."""
    import onnxruntime as ort
    from PIL import Image

    from cycletime.config import load_config
    from cycletime.dataio.images import preprocess_decoded

    verified = verified_release(release)
    root = Path(release.path).parent
    config = load_config(Path(verified["config_dir"]) / "production.yaml")
    dataset = load_config(Path(verified["config_dir"]) / "dataset.yaml")
    runtime = config["runtime"]
    if not runtime["offline_inspection_required"]:
        raise StationFault("the station contract requires offline inspection")
    maximum_age = runtime["maximum_frame_age_ms"]
    if maximum_age is None:
        raise StationFault("the station requires a configured maximum frame age")
    age_ms = (perf_counter_ns() - int(captured_at_ns)) / 1_000_000
    if age_ms < 0 or age_ms > float(maximum_age):
        raise StationFault(
            runtime["unknown_or_late_frame_action"],
            {"frame_id": frame_id, "frame_age_ms": age_ms, "maximum_frame_age_ms": maximum_age},
        )
    if not frame_id:
        raise StationFault(runtime["unknown_or_late_frame_action"], {"frame_id": frame_id})
    artifact = root / str(verified["artifact_path"])
    if not artifact.is_file() or sha256(artifact) != verified["artifact_sha256"]:
        raise StationFault("the released artifact failed its integrity check")
    started = perf_counter_ns()
    try:
        with Image.open(io.BytesIO(frame)) as image:
            decoded = image.convert("RGB")
            tensor = preprocess_decoded(decoded, dataset)[None]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(artifact), sess_options=options, providers=["CPUExecutionProvider"])
        score, _ = session.run(["image_score", "anomaly_map"], {"image": tensor})
        value = float(score[0])
        if not np.isfinite(value):
            raise ValueError("inference produced a nonfinite score")
    except StationFault:
        raise
    except Exception as exc:
        raise StationFault("inspection failed before producing a verified result", {"error": str(exc)}) from exc
    completed = perf_counter_ns()
    threshold = float(verified["threshold"])
    return record(
        root / "inspections" / f"{frame_id}.json",
        {
            "frame_id": frame_id,
            "captured_at_ns": int(captured_at_ns),
            "frame_sha256": sha256(frame) if isinstance(frame, Path) else __import__("hashlib").sha256(frame).hexdigest(),
            "artifact_sha256": verified["artifact_sha256"],
            "release_id": verified["release_id"],
            "score": value,
            "threshold": threshold,
            "decision": "reject" if value >= threshold else "accept",
            "service_ns": completed - started,
            "frame_age_ms": age_ms,
            "llm_consulted": False,
            "actuation": "the station records a decision and never drives the line directly",
        },
    )
