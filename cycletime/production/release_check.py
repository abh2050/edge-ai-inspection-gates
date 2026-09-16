"""cycletime/production/release_check.py reports whether a commercial release is qualified."""

from __future__ import annotations

import json
from pathlib import Path

from cycletime.config import load_config
from cycletime.contracts import EvidenceRef
from cycletime.evidence import record
from cycletime.production.manifest import ReleaseRejected, verify_release

RESEARCH_GATES = ("gate0", "gate1", "gate2", "gate3", "gate4", "gate5")


def run(config_dir: Path) -> EvidenceRef:
    """Require commercial data rights, independent customer validation, signed artifacts, shift soak tests, fault recovery, and acceptance evidence; refuse null production requirements, passing research gates as release qualification, and activation."""
    config_dir = config_dir.resolve()
    project = config_dir.parent
    production = load_config(config_dir / "production.yaml")
    costs = load_config(config_dir / "costs.yaml")
    unmet: list[str] = []

    research = {}
    for name in RESEARCH_GATES:
        path = project / f"artifacts/{name}.json"
        research[name] = json.loads(path.read_text())["status"] if path.is_file() else "missing"
    if all(status == "passed" for status in research.values()):
        unmet.append("research gates passed, which qualifies research feasibility and never a commercial release")

    if not production.get("release_enabled"):
        unmet.append("release_enabled is false")
    for key in ("customer_dataset_manifest", "commercial_rights_record", "camera_contract", "station_contract"):
        if production.get(key) is None:
            unmet.append(f"{key} is unset")
    for key, value in production["acceptance"].items():
        if value is None:
            unmet.append(f"acceptance criterion {key} is unset")
    for key in ("maximum_frame_age_ms", "current_release_manifest", "rollback_release_manifest"):
        if production["runtime"][key] is None:
            unmet.append(f"runtime {key} is unset")
    if not production["runtime"]["trusted_release_keys"]:
        unmet.append("no trusted release key is configured")
    if not costs["commercial"]["customer_validated"]:
        unmet.append("no independent customer validation has been recorded")
    for key, value in costs["commercial"].items():
        if value is None:
            unmet.append(f"commercial input {key} is unset")

    manifest_path = production["runtime"]["current_release_manifest"]
    verification = None
    if manifest_path:
        try:
            verification = str(verify_release(project / str(manifest_path), production).path)
        except ReleaseRejected as exc:
            unmet.extend(f"release manifest: {reason}" for reason in exc.reasons)
    else:
        unmet.append("no candidate release manifest is recorded")

    soak = project / "artifacts/production/soak.json"
    if not soak.is_file():
        unmet.append("no shift soak evidence is recorded")
    recovery = project / "artifacts/production/fault-recovery.json"
    if not recovery.is_file():
        unmet.append("no camera loss, disk, process, or power recovery rehearsal is recorded")

    report = record(
        project / "artifacts/release-check.json",
        {
            "status": "not_qualified" if unmet else "qualified",
            "research_gate_status": research,
            "research_is_not_release_acceptance": True,
            "unmet_conditions": unmet,
            "verification": verification,
            "dataset_license": "MVTec AD is CC BY-NC-SA 4.0 and cannot support a commercial claim",
            "activation": "this check never activates a station",
        },
    )
    if unmet:
        raise ValueError(
            f"The release is not qualified. Evidence: {report.path}. Unmet conditions: {'; '.join(unmet)}"
        )
    return report
