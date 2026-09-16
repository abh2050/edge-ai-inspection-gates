"""cycletime/production/rollback.py defines an authorized transition to a verified prior release."""
from __future__ import annotations

import json
from pathlib import Path

from cycletime.contracts import JSON, EvidenceRef
from cycletime.evidence import read, record, serialize

AGENT_ROLES = {"agent", "llm", "llm_agent", "cycletime.agent", "automation"}


class RollbackRefused(ValueError):
    """Carry the unmet rollback conditions while the current release stays in place."""

    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("; ".join(reasons))


def rollback(
    current: EvidenceRef, previous: EvidenceRef, authorization: EvidenceRef, config: JSON
) -> EvidenceRef:
    """Verify the prior signed release, station compatibility, and external authorization before an atomic switch and health check; preserve audit evidence and the configured fault action; refuse agent approval, unsigned targets, and automatic line actuation."""
    reasons: list[str] = []
    current_record = read(current)
    target = read(previous)
    approval = read(authorization)
    if target.get("status") != "verified" or not target.get("signer", {}).get("verified"):
        reasons.append("the rollback target is not a verified signed release")
    if current_record.get("station_id") != target.get("station_id"):
        reasons.append("the rollback target belongs to another station")
    if approval.get("approver_role", "").lower() in AGENT_ROLES or approval.get("approved_by_agent"):
        reasons.append("an agent may not authorize a rollback")
    if not approval.get("external_authorization_reference"):
        reasons.append("rollback requires an external authorization reference")
    if approval.get("release_id") != target.get("release_id"):
        reasons.append("the authorization does not name the rollback target")
    if approval.get("station_id") != target.get("station_id"):
        reasons.append("the authorization does not name this station")
    maximum_seconds = config["acceptance"]["maximum_rollback_seconds"]
    if maximum_seconds is None:
        reasons.append("the acceptance criteria leave maximum_rollback_seconds unset")
    pointer = config["runtime"]["current_release_manifest"]
    if not pointer:
        reasons.append("the station has no configured current release pointer")
    destination = Path(current.path).parent / "rollback.json"
    if reasons:
        record(
            destination,
            {
                "status": "refused",
                "reasons": reasons,
                "current_release": serialize(current),
                "target_release": serialize(previous),
                "authorization": serialize(authorization),
                "current_release_unchanged": True,
                "fault_action": config["runtime"]["unknown_or_late_frame_action"],
            },
        )
        raise RollbackRefused(reasons)
    # The switch replaces one pointer file atomically; the station reloads the verified target.
    pointer_path = Path(str(pointer))
    staged = pointer_path.with_suffix(".staged")
    staged.write_text(json.dumps({"release_id": target["release_id"], "manifest": target["manifest"]}, sort_keys=True))
    staged.replace(pointer_path)
    health = {
        "pointer": str(pointer_path),
        "release_id": target["release_id"],
        "artifact_sha256": target["artifact_sha256"],
        "pointer_matches_target": json.loads(pointer_path.read_text())["release_id"] == target["release_id"],
    }
    return record(
        destination,
        {
            "status": "completed" if health["pointer_matches_target"] else "failed",
            "current_release": serialize(current),
            "target_release": serialize(previous),
            "authorization": serialize(authorization),
            "approver_role": approval.get("approver_role"),
            "external_authorization_reference": approval.get("external_authorization_reference"),
            "health_check": health,
            "maximum_rollback_seconds": maximum_seconds,
            "line_actuation": "the station switches releases and never actuates the line",
        },
    )
