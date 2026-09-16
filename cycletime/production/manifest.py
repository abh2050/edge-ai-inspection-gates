"""cycletime/production/manifest.py verifies the provenance of a candidate release."""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

from cycletime.contracts import JSON, EvidenceRef
from cycletime.dataio.archive import sha256
from cycletime.evidence import record

REQUIRED_MANIFEST_FIELDS = (
    "release_id",
    "station_id",
    "artifact_path",
    "artifact_sha256",
    "preprocessing_sha256",
    "dataset_manifest_sha256",
    "threshold",
    "threshold_source",
    "lock_sha256",
    "sbom_path",
    "sbom_sha256",
    "signature",
    "customer_acceptance",
    "commercial_rights_record",
)
NONCOMMERCIAL_DATASETS = ("mvtec", "mvtec_ad", "mvtec anomaly detection")


class ReleaseRejected(ValueError):
    """Carry every unmet release condition instead of a single first failure."""

    def __init__(self, reasons: list[str], report: EvidenceRef | None = None):
        self.reasons = reasons
        self.report = report
        super().__init__("; ".join(reasons))


def signed_payload(manifest: JSON) -> bytes:
    """Return the exact bytes a signer covers: every field except the signature itself."""
    covered = {key: value for key, value in manifest.items() if key != "signature"}
    return json.dumps(covered, sort_keys=True, separators=(",", ":")).encode("utf-8")


def verify_signature(manifest: JSON, config: JSON, reasons: list[str]) -> JSON | None:
    """Verify a detached signature from a configured trusted signer; refuse unknown keys and algorithms."""
    signature = manifest.get("signature") or {}
    trusted = {key["key_id"]: key for key in config["runtime"]["trusted_release_keys"]}
    key = trusted.get(signature.get("key_id"))
    if not trusted:
        reasons.append("no trusted release key is configured")
        return None
    if key is None:
        reasons.append(f"signer {signature.get('key_id')!r} is not a trusted release key")
        return None
    if signature.get("algorithm") != key.get("algorithm") or key.get("algorithm") != "hmac-sha256":
        # A commercial deployment replaces this with asymmetric signing; the algorithm must match.
        reasons.append("the signature algorithm is unsupported or does not match the trusted key")
        return None
    material = Path(key["key_path"])
    if not material.is_file():
        reasons.append("the trusted key material is unavailable")
        return None
    expected = hmac.new(material.read_bytes(), signed_payload(manifest), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, str(signature.get("value", ""))):
        reasons.append("the signature does not cover this manifest content")
        return None
    return {"key_id": key["key_id"], "algorithm": key["algorithm"], "verified": True}


def verify_release(manifest_path: Path, production_config: JSON) -> EvidenceRef:
    """Verify signature, trusted signer, artifact and data hashes, preprocessing, frozen threshold, lockfile, SBOM, customer acceptance, and rights records; refuse MVTec-based commercial claims, unknown licenses, missing holdout evidence, and an agent signature used as customer authorization."""
    manifest_path = Path(manifest_path)
    reasons: list[str] = []
    if not production_config.get("release_enabled"):
        reasons.append("release_enabled is false in config/production.yaml")
    for key in ("customer_dataset_manifest", "commercial_rights_record", "camera_contract", "station_contract"):
        if production_config.get(key) is None:
            reasons.append(f"production configuration leaves {key} unset")
    acceptance = production_config["acceptance"]
    for key, value in acceptance.items():
        if value is None:
            reasons.append(f"acceptance criterion {key} is unset")
    if not manifest_path.is_file():
        reasons.append(f"the release manifest {manifest_path.name} does not exist")
        raise ReleaseRejected(reasons)
    manifest = json.loads(manifest_path.read_text())
    for field in REQUIRED_MANIFEST_FIELDS:
        if manifest.get(field) in (None, "", [], {}):
            reasons.append(f"the release manifest omits {field}")
    root = manifest_path.parent
    for field, digest_field in (("artifact_path", "artifact_sha256"), ("sbom_path", "sbom_sha256")):
        target = root / str(manifest.get(field, ""))
        if not target.is_file():
            reasons.append(f"the release names a missing {field}")
        elif sha256(target) != manifest.get(digest_field):
            reasons.append(f"the recorded {digest_field} does not match its file")
    release_config = production_config["release"]
    if release_config["require_signed_manifest"]:
        signer = verify_signature(manifest, production_config, reasons)
    else:
        signer = None
        reasons.append("a release requires a signed manifest")
    if release_config["require_locked_dependencies"] and manifest.get("lock_sha256") != manifest.get(
        "verified_lock_sha256", manifest.get("lock_sha256")
    ):
        reasons.append("the locked dependency digest is inconsistent")
    acceptance_record = manifest.get("customer_acceptance") or {}
    if release_config["require_customer_acceptance"]:
        if acceptance_record.get("approved_by_role") in (None, "", "agent", "llm_agent"):
            reasons.append("customer acceptance requires a named human approver, never an agent")
        if acceptance_record.get("signature_key_id") == (manifest.get("signature") or {}).get("key_id"):
            reasons.append("the release signer cannot also stand in as the customer approver")
        if not acceptance_record.get("independent_holdout_evidence"):
            reasons.append("customer acceptance requires independent holdout evidence")
        if acceptance_record.get("defective_parts_tested", 0) < (
            acceptance["minimum_defective_test_parts"] or 1
        ):
            reasons.append("customer acceptance lacks the configured defective-part sample")
    dataset_name = str(manifest.get("dataset_name", "")).lower()
    if any(name in dataset_name for name in NONCOMMERCIAL_DATASETS):
        reasons.append("MVTec evidence cannot qualify a commercial release")
    if str(manifest.get("dataset_license", "")).lower() in ("", "unknown"):
        reasons.append("the release does not identify its dataset license")
    if manifest.get("threshold_source") != "customer_validation_holdout":
        reasons.append("the frozen threshold must come from customer validation data")
    report = record(
        root / "release-verification.json",
        {
            "status": "rejected" if reasons else "verified",
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "release_id": manifest.get("release_id"),
            "station_id": manifest.get("station_id"),
            "artifact_sha256": manifest.get("artifact_sha256"),
            "threshold": manifest.get("threshold"),
            "signer": signer,
            "reasons": reasons,
            "activation": "verification never activates a station",
        },
    )
    if reasons:
        raise ReleaseRejected(reasons, report)
    return report
