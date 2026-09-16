"""These tests specify required behavior and refuse silent scaffold success."""
import hashlib
import hmac
import json
from pathlib import Path
from time import perf_counter_ns

import pytest

from cycletime.dataio.archive import sha256
from cycletime.evidence import read, record
from cycletime.production.manifest import ReleaseRejected, signed_payload, verify_release
from cycletime.production.rollback import RollbackRefused, rollback
from cycletime.production.runtime import StationFault, inspect_frame
from cycletime.production.telemetry import record_station_health

pytestmark = [pytest.mark.contract]


def production_config(tmp_path: Path, key_path: Path) -> dict:
    """Return a fully configured station whose blockers the operator has resolved."""
    return {
        "release_enabled": True,
        "customer_dataset_manifest": "customer/dataset.json",
        "commercial_rights_record": "customer/rights.md",
        "camera_contract": "customer/camera.md",
        "station_contract": "customer/station.md",
        "acceptance": {
            "line_rate_parts_per_minute": 45,
            "minimum_recall": 0.95,
            "maximum_false_reject_rate": 0.02,
            "maximum_deadline_miss_rate": 0.0,
            "confidence_level": 0.95,
            "minimum_defective_test_parts": 50,
            "shift_soak_hours": 8,
            "minimum_soak_shifts": 3,
            "maximum_restart_seconds": 120,
            "maximum_rollback_seconds": 300,
        },
        "runtime": {
            "offline_inspection_required": True,
            "unknown_or_late_frame_action": "controlled_station_fault",
            "maximum_frame_age_ms": 100,
            "trusted_release_keys": [
                {"key_id": "station-key-1", "algorithm": "hmac-sha256", "key_path": str(key_path)}
            ],
            "current_release_manifest": str(tmp_path / "current.json"),
            "rollback_release_manifest": str(tmp_path / "previous.json"),
        },
        "telemetry": {
            "record_frame_ids": True,
            "record_scores": True,
            "record_deadline_misses": True,
            "record_temperature_when_available": True,
            "raw_image_retention_days": 0,
            "log_retention_days": 30,
            "require_tenant_isolation": True,
        },
        "release": {
            "require_signed_manifest": True,
            "require_sbom": True,
            "require_locked_dependencies": True,
            "require_customer_acceptance": True,
        },
    }


def signed_manifest(tmp_path: Path, key: bytes, **overrides) -> Path:
    """Write a signed release manifest with real artifact and SBOM digests."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"onnx-artifact")
    sbom = tmp_path / "sbom.json"
    sbom.write_text(json.dumps({"packages": []}))
    manifest = {
        "release_id": "release-1",
        "station_id": "station-1",
        "artifact_path": artifact.name,
        "artifact_sha256": sha256(artifact),
        "preprocessing_sha256": "b" * 64,
        "dataset_manifest_sha256": "c" * 64,
        "dataset_name": "customer_line_a",
        "dataset_license": "customer-proprietary",
        "threshold": 2.0,
        "threshold_source": "customer_validation_holdout",
        "lock_sha256": "d" * 64,
        "sbom_path": sbom.name,
        "sbom_sha256": sha256(sbom),
        "commercial_rights_record": "customer/rights.md",
        "customer_acceptance": {
            "approved_by_role": "quality_manager",
            "signature_key_id": "customer-key-9",
            "independent_holdout_evidence": "customer/holdout.json",
            "defective_parts_tested": 60,
        },
    }
    manifest.update(overrides)
    manifest["signature"] = {
        "key_id": "station-key-1",
        "algorithm": "hmac-sha256",
        "value": hmac.new(key, signed_payload(manifest), hashlib.sha256).hexdigest(),
    }
    path = tmp_path / "release.json"
    path.write_text(json.dumps(manifest, sort_keys=True))
    return path


def test_reject_unqualified_release(tmp_path: Path) -> None:
    """A release must reject missing commercial rights, missing customer acceptance, and a modified signature or artifact digest."""
    key_path = tmp_path / "station.key"
    key = b"station-secret"
    key_path.write_bytes(key)
    config = production_config(tmp_path, key_path)

    verified = read(verify_release(signed_manifest(tmp_path, key), config))
    assert verified["status"] == "verified" and verified["signer"]["verified"] is True

    # An agent may never stand in as the customer approver, and acceptance needs holdout evidence.
    for acceptance, expected in (
        ({"approved_by_role": "agent", "signature_key_id": "customer-key-9", "independent_holdout_evidence": "x", "defective_parts_tested": 60}, "named human approver"),
        ({"approved_by_role": "quality_manager", "signature_key_id": "station-key-1", "independent_holdout_evidence": "x", "defective_parts_tested": 60}, "cannot also stand in"),
        ({"approved_by_role": "quality_manager", "signature_key_id": "customer-key-9", "independent_holdout_evidence": None, "defective_parts_tested": 60}, "independent holdout evidence"),
        ({"approved_by_role": "quality_manager", "signature_key_id": "customer-key-9", "independent_holdout_evidence": "x", "defective_parts_tested": 3}, "defective-part sample"),
    ):
        with pytest.raises(ReleaseRejected, match=expected):
            verify_release(signed_manifest(tmp_path / str(id(expected)), key, customer_acceptance=acceptance), config)

    # Research data can never qualify a commercial release.
    with pytest.raises(ReleaseRejected, match="MVTec evidence cannot qualify"):
        verify_release(signed_manifest(tmp_path / "mvtec", key, dataset_name="MVTec AD"), config)
    with pytest.raises(ReleaseRejected, match="customer validation data"):
        verify_release(signed_manifest(tmp_path / "frozen", key, threshold_source="held_out_normal_training"), config)

    # A tampered artifact or signature invalidates the release.
    tampered = signed_manifest(tmp_path / "tampered", key)
    (tampered.parent / "model.onnx").write_bytes(b"replaced-artifact")
    with pytest.raises(ReleaseRejected, match="does not match its file"):
        verify_release(tampered, config)

    forged = signed_manifest(tmp_path / "forged", key)
    content = json.loads(forged.read_text())
    content["threshold"] = 0.1
    forged.write_text(json.dumps(content, sort_keys=True))
    with pytest.raises(ReleaseRejected, match="signature does not cover"):
        verify_release(forged, config)

    with pytest.raises(ReleaseRejected, match="not a trusted release key"):
        verify_release(
            signed_manifest(tmp_path / "untrusted", key),
            dict(config, runtime=dict(config["runtime"], trusted_release_keys=[{"key_id": "other", "algorithm": "hmac-sha256", "key_path": str(key_path)}])),
        )


def test_fault_on_stale_input(tmp_path: Path) -> None:
    """A stale frame or unavailable verified model must produce the configured station fault instead of an accept result."""
    unverified = record(tmp_path / "unverified.json", {"status": "rejected", "reasons": ["unsigned"]})
    with pytest.raises(StationFault, match="refuses an unverified release"):
        inspect_frame(b"\x89PNG", "frame-1", perf_counter_ns(), unverified)

    release = record(
        tmp_path / "release-verification.json",
        {
            "status": "verified",
            "release_id": "release-1",
            "artifact_path": "model.onnx",
            "artifact_sha256": "e" * 64,
            "threshold": 2.0,
            "config_dir": str(tmp_path / "config"),
        },
    )
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config/production.yaml").write_text(
        json.dumps({"runtime": {"offline_inspection_required": True, "maximum_frame_age_ms": 100, "unknown_or_late_frame_action": "controlled_station_fault"}})
    )
    (tmp_path / "config/dataset.yaml").write_text(json.dumps({"input": {"color": "RGB", "layout": "NCHW", "image_interpolation": "bilinear"}}))

    stale = perf_counter_ns() - 5_000_000_000
    with pytest.raises(StationFault, match="controlled_station_fault") as fault:
        inspect_frame(b"\x89PNG", "frame-2", stale, release)
    assert fault.value.detail["frame_age_ms"] > 100

    with pytest.raises(StationFault, match="controlled_station_fault"):
        inspect_frame(b"\x89PNG", "", perf_counter_ns(), release)

    # The released artifact is missing, so the station faults rather than accepting the part.
    with pytest.raises(StationFault, match="integrity check"):
        inspect_frame(b"\x89PNG", "frame-3", perf_counter_ns(), release)


def test_rollback_authorization(tmp_path: Path) -> None:
    """A rollback must require external authorization and a valid prior release and must preserve the current release when validation fails."""
    key_path = tmp_path / "station.key"
    key_path.write_bytes(b"station-secret")
    config = production_config(tmp_path, key_path)
    pointer = Path(config["runtime"]["current_release_manifest"])
    pointer.write_text(json.dumps({"release_id": "release-2", "manifest": "current"}))

    current = record(tmp_path / "current-verification.json", {"status": "verified", "station_id": "station-1", "release_id": "release-2"})
    target = record(
        tmp_path / "previous-verification.json",
        {"status": "verified", "station_id": "station-1", "release_id": "release-1", "manifest": "previous", "artifact_sha256": "a" * 64, "signer": {"verified": True}},
    )
    approval = {
        "approver_role": "plant_manager",
        "external_authorization_reference": "change-ticket-42",
        "release_id": "release-1",
        "station_id": "station-1",
    }

    agent_signed = record(tmp_path / "agent-auth.json", dict(approval, approver_role="llm_agent"))
    with pytest.raises(RollbackRefused, match="agent may not authorize"):
        rollback(current, target, agent_signed, config)
    assert json.loads(pointer.read_text())["release_id"] == "release-2"

    no_reference = record(tmp_path / "no-ref.json", dict(approval, external_authorization_reference=None))
    with pytest.raises(RollbackRefused, match="external authorization reference"):
        rollback(current, target, no_reference, config)

    unsigned = record(tmp_path / "unsigned-target.json", {"status": "verified", "station_id": "station-1", "release_id": "release-1", "signer": {}})
    with pytest.raises(RollbackRefused, match="verified signed release"):
        rollback(current, unsigned, record(tmp_path / "ok-auth.json", approval), config)
    assert json.loads(pointer.read_text())["release_id"] == "release-2"

    completed = read(rollback(current, target, record(tmp_path / "auth.json", approval), config))
    assert completed["status"] == "completed"
    assert completed["health_check"]["pointer_matches_target"] is True
    assert json.loads(pointer.read_text())["release_id"] == "release-1"
    assert "never actuates the line" in completed["line_actuation"]


def test_station_telemetry_limits(tmp_path: Path) -> None:
    """Telemetry must refuse raw-image retention, missing tenant scope, and recall claims from drift alone."""
    key_path = tmp_path / "station.key"
    key_path.write_bytes(b"k")
    config = production_config(tmp_path, key_path)

    with pytest.raises(ValueError, match="Raw image retention is unauthorized"):
        record_station_health(record(tmp_path / "raw.json", {"tenant_id": "t", "frame_bytes": "iVBORw0"}), config)
    with pytest.raises(ValueError, match="requires a tenant identifier"):
        record_station_health(record(tmp_path / "no-tenant.json", {"scores": [1.0]}), config)
    with pytest.raises(ValueError, match="Score drift alone cannot establish reduced recall"):
        record_station_health(
            record(tmp_path / "drift.json", {"tenant_id": "t", "scores": [1.0], "claims_recall_change": True}), config
        )

    healthy = read(
        record_station_health(
            record(tmp_path / "ok.json", {"tenant_id": "t", "station_id": "s", "window_id": "w1", "scores": [1.0], "latency_ms": 15.0, "deadline_misses": 0, "frame_ids": ["f1"]}),
            config,
        )
    )
    assert healthy["raw_images_retained"] is False
    assert "does not measure recall" in healthy["drift_interpretation"]
