"""Gate 2 collects conversion, parity, and exactly-once accuracy evidence."""

import fcntl
from pathlib import Path

from cycletime.contracts import EvidenceRef, PredictionCache
from cycletime.dataio.archive import sha256
from cycletime.evidence import read, record, reference, restore, serialize
from cycletime.export.metadata import load_artifact
from cycletime.export.quantize_int8 import export_int8
from cycletime.export.to_fp16 import export_fp16
from cycletime.export.verify_parity import ParityError, verify_export
from cycletime.model.evaluate import evaluate_once, require_quality, verify_cached_result
from cycletime.model.pipeline import prepare


def run(config_dir: Path) -> EvidenceRef:
    """Reuse FP32 and assess both reduced exports; record every failure and never repeat FP32 evaluation."""
    model, data, config = prepare(config_dir)
    project = Path(data["_project"])
    gate1_ref = reference(project / "artifacts/gate1.json")
    gate1 = read(gate1_ref)
    if gate1["status"] != "passed" or gate1["lock_sha256"] != sha256(project / "uv.lock"):
        raise ValueError("Gate 2 requires passed Gate 1 evidence and unchanged dependencies.")
    fp32, metadata = load_artifact(project / config["fp32"]["path"])
    if fp32.sha256 != gate1["artifact_sha256"]:
        raise ValueError("Gate 1 and the canonical FP32 artifact disagree.")
    checkpoint = restore(gate1["checkpoint"])
    checkpoint_data = read(checkpoint)
    if checkpoint_data["model_config"] != model or checkpoint_data["dataset_config"] != data:
        raise ValueError("Training configuration changed after Gate 1.")
    if metadata["checkpoint_sha256"] != checkpoint.sha256:
        raise ValueError("FP32 checkpoint provenance changed.")
    partitions = restore(gate1["partitions"])
    threshold = restore(gate1["threshold"])
    metrics_ref = restore(gate1["metrics"])
    verify_cached_result({"samples": gate1["predictions"], "metrics": gate1["metrics"]})
    require_quality(
        PredictionCache(
            fp32, restore(gate1["predictions"]), metrics_ref, read(metrics_ref)["evaluator_sha256"]
        ),
        model["quality_floors"],
    )
    with (project / "artifacts/gate2.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another Gate 2 process is running.") from exc
        rows = [
            {
                "precision": "fp32",
                "artifact_sha256": fp32.sha256,
                "status": "passed",
                "parity_status": "passed_in_gate1",
                "metrics": serialize(metrics_ref),
                "metrics_values": read(metrics_ref),
                "evaluation": "reused_gate1_cache",
            }
        ]
        for precision, converter in [
            ("fp16", lambda: export_fp16(fp32, config)),
            ("int8", lambda: export_int8(fp32, partitions, config)),
        ]:
            row = {"precision": precision, "status": "failed", "errors": []}
            print(f"Gate 2 checks {precision}.", flush=True)
            try:
                artifact = converter()
                row.update(artifact_sha256=artifact.sha256, artifact_path=str(artifact.path))
                try:
                    parity = verify_export(fp32, artifact, partitions, config["parity"])
                    row.update(parity=serialize(parity), parity_status="passed")
                except ParityError as exc:
                    row.update(parity=serialize(exc.report), parity_status="failed")
                    row["errors"].append(str(exc))
                # Accuracy is independent evidence for each exported artifact. A failed
                # numerical comparison remains a gate failure even if AUROC is acceptable.
                cache = evaluate_once(
                    artifact,
                    dict(data, _threshold=serialize(threshold)),
                    project / data["test"]["ledger"],
                )
                row.update(
                    metrics=serialize(cache.metrics),
                    predictions=serialize(cache.samples),
                    metrics_values=read(cache.metrics),
                    evaluation="exactly_once_cache",
                )
                try:
                    require_quality(cache, model["quality_floors"])
                except ValueError as exc:
                    row["errors"].append(str(exc))
                row["status"] = "failed" if row["errors"] else "passed"
            except Exception as exc:  # noqa: BLE001 - Each required export records failures independently.
                row["errors"].append(f"{type(exc).__name__}: {exc}")
            rows.append(row)
        passed = all(row["status"] == "passed" for row in rows)
        report = record(
            project / "artifacts/gate2.json",
            {
                "gate": 2,
                "status": "passed" if passed else "failed",
                "gate1": serialize(gate1_ref),
                "exports": rows,
                "export_config": config,
                "quality_floors": model["quality_floors"],
            },
        )
        lines = [
            "# Gate 2 records precision conversion results.",
            "",
            f"Gate 2 {'passed' if passed else 'failed'} its configured checks.",
            "",
            "| Precision | Parity | Image AUROC | Pixel AUROC | Recall | Status |",
            "|---|---|---:|---:|---:|---|",
        ]
        for row in rows:
            m = row.get("metrics_values", {})
            values = [
                f"{m[k]:.6f}" if k in m else "Unavailable"
                for k in ("image_auroc", "pixel_auroc", "recall")
            ]
            lines.append(
                f"| {row['precision']} | {row.get('parity_status', 'failed_before_comparison')} | {' | '.join(values)} | {row['status']} |"
            )
        lines.extend(
            [
                "",
                "The fp16 row represents mixed FP16/FP32 execution in the serialized graph.",
                "The int8 row represents mixed INT8/FP32 execution in the serialized graph.",
                "Each revised graph converts 36 of 104 convolution nodes selected as paired squeeze-and-excitation operations.",
                "All configurations use the threshold frozen before the FP32 test evaluation.",
                "Gate 2 reuses FP32 accuracy and records one complete evaluation for each supported new artifact.",
                "The INT8 calibration uses 64 distinct images from the normal training fit partition.",
                "Parity compares both outputs on 16 held-out normal validation images.",
                "The conversion metadata records graph precision coverage; it does not establish CPU or Neural Engine kernel placement.",
                "ADR 0005 records the validation-only diagnosis and the mixed-precision decision.",
                "This gate does not measure latency.",
                "",
            ]
        )
        for row in rows:
            if row.get("errors"):
                lines.append(f"The {row['precision']} checks reported: {'; '.join(row['errors'])}")
        (project / "docs/gate2.md").write_text("\n".join(lines) + "\n")
        if not passed:
            raise ValueError(f"Gate 2 failed. Evidence: {report.path}")
        return report
