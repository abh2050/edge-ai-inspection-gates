"""cycletime/production/telemetry.py defines measurements that can reveal drift and missed deadlines."""
from __future__ import annotations

from pathlib import Path

from cycletime.contracts import JSON, EvidenceRef
from cycletime.evidence import read, record

RAW_IMAGE_FIELDS = ("frame_bytes", "image", "raw_image", "thumbnail", "crop")


def record_station_health(observation: EvidenceRef, config: JSON) -> EvidenceRef:
    """Record scores, latency, errors, capture conditions, and permitted thermal signals under retention and tenant controls; refuse raw-image retention without authorization and claims that score drift alone proves reduced recall."""
    data = read(observation)
    telemetry = config["telemetry"]
    present = [field for field in RAW_IMAGE_FIELDS if data.get(field) is not None]
    if present and int(telemetry["raw_image_retention_days"]) <= 0:
        raise ValueError(f"Raw image retention is unauthorized; the observation carries {present}.")
    if telemetry["require_tenant_isolation"] and not data.get("tenant_id"):
        raise ValueError("Station telemetry requires a tenant identifier.")
    if data.get("claims_recall_change") and not data.get("labeled_defect_evidence"):
        raise ValueError(
            "Score drift alone cannot establish reduced recall; labeled defect evidence is required."
        )
    thermal = data.get("thermal") if telemetry["record_temperature_when_available"] else None
    return record(
        Path(observation.path).with_name(f"health-{data.get('window_id', 'window')}.json"),
        {
            "tenant_id": data.get("tenant_id"),
            "station_id": data.get("station_id"),
            "window_id": data.get("window_id"),
            "frame_ids": data.get("frame_ids") if telemetry["record_frame_ids"] else None,
            "scores": data.get("scores") if telemetry["record_scores"] else None,
            "latency_ms": data.get("latency_ms"),
            "deadline_misses": data.get("deadline_misses") if telemetry["record_deadline_misses"] else None,
            "errors": data.get("errors", []),
            "capture_conditions": data.get("capture_conditions"),
            "thermal": thermal,
            "thermal_status": "recorded when available" if thermal else "unavailable or not recorded",
            "raw_images_retained": False,
            "raw_image_retention_days": telemetry["raw_image_retention_days"],
            "log_retention_days": telemetry["log_retention_days"],
            "drift_interpretation": "score drift is a monitoring signal and does not measure recall",
        },
    )
