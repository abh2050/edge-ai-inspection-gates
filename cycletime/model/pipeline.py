"""Gate 1 consumes verified dataset evidence and refuses skipped quality checks."""

from pathlib import Path

from cycletime.config import load_config
from cycletime.dataio.archive import sha256
from cycletime.evidence import read, reference


def prepare(config_dir: Path) -> tuple[dict, dict, dict]:
    """Load matching Gate 0 evidence and configs; refuse stale dataset or lock provenance."""
    config_dir = config_dir.resolve()
    project = config_dir.parent
    data = load_config(config_dir / "dataset.yaml")
    gate0 = read(reference(project / "artifacts/gate0.json"))
    if gate0["status"] != "passed" or gate0["dataset_config"] != data:
        raise ValueError("Gate 0 has not passed for this dataset configuration.")
    if sha256(project / str(data["manifest"])) != gate0["manifest_sha256"]:
        raise ValueError("The dataset manifest changed after Gate 0.")
    if not (project / "uv.lock").is_file():
        raise ValueError("Gate 1 requires locked dependencies.")
    data.update(_project=str(project), _dataset_sha256=gate0["dataset_sha256"])
    return load_config(config_dir / "model.yaml"), data, load_config(config_dir / "export.yaml")


def run(config_dir: Path):
    """Run Gate 1 under a process lock; preserve failed quality results and refuse a second evaluation."""
    import fcntl

    from cycletime.dataio.calibration import build_partitions
    from cycletime.evidence import record, serialize
    from cycletime.export.to_onnx import export_fp32
    from cycletime.model.evaluate import evaluate_once, require_quality
    from cycletime.model.threshold import initial_threshold
    from cycletime.model.train import train

    model_config, dataset_config, export_config = prepare(config_dir)
    project = Path(dataset_config["_project"])
    with (project / "artifacts/gate1.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another Gate 1 process is running.") from exc
        partitions = build_partitions(dataset_config)
        checkpoint = train(model_config, dataset_config, partitions)
        threshold = initial_threshold(checkpoint, partitions, model_config["threshold"])
        artifact = export_fp32(checkpoint, export_config)
        evaluation_config = dict(dataset_config, _threshold=serialize(threshold))
        cache = evaluate_once(
            artifact, evaluation_config, project / dataset_config["test"]["ledger"]
        )
        metrics = read(cache.metrics)
        error = None
        try:
            require_quality(cache, model_config["quality_floors"])
        except ValueError as exc:
            error = str(exc)
        report = record(
            project / "artifacts/gate1.json",
            {
                "gate": 1,
                "status": "failed" if error else "passed",
                "failure_reason": error,
                "artifact_sha256": artifact.sha256,
                "checkpoint": serialize(checkpoint),
                "threshold": serialize(threshold),
                "metrics": serialize(cache.metrics),
                "predictions": serialize(cache.samples),
                "partitions": serialize(partitions),
                "quality_floors": model_config["quality_floors"],
                "lock_sha256": sha256(project / "uv.lock"),
            },
        )
        rows = read(cache.samples)["images"]
        markdown = [
            "# Gate 1 records the FP32 model evaluation.",
            "",
            f"Gate 1 {'failed' if error else 'passed'} its configured AUROC floors.",
            "",
            "| Metric | Measured value | Required floor |",
            "|---|---:|---:|",
            f"| Image AUROC | {metrics['image_auroc']:.6f} | {model_config['quality_floors']['image_auroc']:.2f} |",
            f"| Pixel AUROC | {metrics['pixel_auroc']:.6f} | {model_config['quality_floors']['pixel_auroc']:.2f} |",
            f"| Recall at frozen threshold | {metrics['recall']:.6f} | Gate 5 enforces the recall constraint. |",
            "",
            f"The held-out normal threshold is {metrics['threshold']:.8f}.",
            f"The evaluation used {metrics['image_count']} test images and {metrics['pixel_count']} original-resolution pixels.",
            "The ledger records one completed evaluation for this artifact.",
            "The pixel metric resizes score maps bilinearly to the original geometry and does not resize ground truth.",
            "The 256-pixel model input can lose small defects; the following counts describe misses at the frozen threshold.",
            "",
            "| Test subtype | Images | Rejected | Defect pixels, min–max |",
            "|---|---:|---:|---:|",
        ]
        for label in sorted({r["defect_type"] for r in rows}):
            group = [r for r in rows if r["defect_type"] == label]
            areas = [r["defect_pixels"] for r in group]
            markdown.append(
                f"| {label} | {len(group)} | {sum(r['score'] >= metrics['threshold'] for r in group)} | {min(areas)}–{max(areas)} |"
            )
        markdown.extend(
            [
                "",
                f"The artifact SHA256 is `{artifact.sha256}`.",
                "The evidence references are recorded in artifacts/gate1.json.",
                "This gate records accuracy and does not establish latency or commercial readiness.",
            ]
        )
        (project / "docs/gate1.md").write_text("\n".join(markdown) + "\n")
        if error:
            raise ValueError(error)
        return report
