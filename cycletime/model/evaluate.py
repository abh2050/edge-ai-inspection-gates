"""This module durably reserves one evaluation and preserves exact cached predictions."""

from __future__ import annotations

import fcntl
import json
import math
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from cycletime.contracts import JSON, Artifact, EvidenceRef, PredictionCache
from cycletime.dataio.archive import sha256
from cycletime.dataio.images import load_image, preprocessing_digest
from cycletime.dataio.verify import write_json
from cycletime.evidence import digest, read, record, reference, restore, serialize


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Calculate anomaly-positive counts and metrics; refuse missing classes or nonfinite scores."""
    if (
        labels.shape != scores.shape
        or set(np.unique(labels)) != {0, 1}
        or not np.isfinite(scores).all()
        or not math.isfinite(threshold)
    ):
        raise ValueError("Binary labels, both classes, and finite scores are required.")
    actual = labels.astype(bool)
    predicted = scores >= threshold
    tp = int(np.sum(actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    tn = int(np.sum(~actual & ~predicted))
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": tp / (tp + fn),
        "false_positive_rate": fp / (fp + tn),
        "f1": 2 * tp / (2 * tp + fp + fn),
    }


def reserve_once(ledger_path: Path, artifact_hash: str, identity: dict, work) -> dict:
    """Reserve a durable run before work and retain failures; refuse duplicate starts or changed cache identity."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_suffix(".json.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = (
            json.loads(ledger_path.read_text())
            if ledger_path.exists()
            else {"schema_version": 1, "runs": {}}
        )
        existing = ledger["runs"].get(artifact_hash)
        if existing is not None:
            if (
                existing["status"] != "completed"
                or existing["started_count"] != 1
                or existing["completed_count"] != 1
                or existing["identity"] != identity
            ):
                raise ValueError(
                    "This artifact already has a started evaluation; automatic re-evaluation is forbidden."
                )
            return existing["result"]
        ledger["runs"][artifact_hash] = dict(
            identity,
            identity=identity,
            artifact_sha256=artifact_hash,
            started_count=1,
            completed_count=0,
            status="started",
            started_at_utc=datetime.now(UTC).isoformat(),
            completed_at_utc=None,
            prediction_cache_sha256=None,
            failure_reason=None,
        )
        write_json(ledger_path, ledger)
    try:
        result = work()
    except BaseException as exc:
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = json.loads(ledger_path.read_text())
            ledger["runs"][artifact_hash].update(
                status="failed", failure_reason=f"{type(exc).__name__}: {exc}"
            )
            write_json(ledger_path, ledger)
        raise
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(ledger_path.read_text())
        ledger["runs"][artifact_hash].update(
            status="completed",
            completed_count=1,
            completed_at_utc=datetime.now(UTC).isoformat(),
            result=result,
            prediction_cache_sha256=result["samples"]["sha256"],
        )
        write_json(ledger_path, ledger)
    return result


def verify_cached_result(result: dict) -> tuple[EvidenceRef, EvidenceRef]:
    """Validate cached metadata and prediction bytes; refuse missing or corrupt cached files."""
    samples, metrics = restore(result["samples"]), restore(result["metrics"])
    for item in read(samples)["images"]:
        path = samples.path.parent / item["prediction_file"]
        if sha256(path) != item["prediction_sha256"]:
            raise ValueError("Cached prediction bytes changed; re-evaluation remains forbidden.")
    return samples, metrics


def evaluate_once(artifact: Artifact, dataset_config: JSON, ledger_path: Path) -> PredictionCache:
    """Evaluate every test image once on CPU with original mask geometry; refuse duplicate runs, stale evidence, test tuning, and undefined metrics."""
    project = Path(str(dataset_config["_project"]))
    root = project / str(dataset_config["root"])
    if sha256(artifact.path) != artifact.sha256:
        raise ValueError("Exported artifact hash mismatch.")
    if (
        artifact.preprocessing_sha256 != preprocessing_digest(dataset_config)
        or artifact.dataset_sha256 != dataset_config["_dataset_sha256"]
    ):
        raise ValueError("Artifact data or preprocessing provenance mismatch.")
    settings = dataset_config["test"]
    if (
        settings["canonical_provider"] != "CPUExecutionProvider"
        or settings["maximum_full_evaluations_per_artifact"] != 1
    ):
        raise ValueError("Evaluation requires exactly one complete CPU pass.")
    threshold_ref = restore(dataset_config["_threshold"])
    threshold_data = read(threshold_ref)
    export_metadata = read(reference(artifact.path.with_suffix(".json")))
    if (
        export_metadata["artifact"]["sha256"] != artifact.sha256
        or export_metadata["checkpoint_sha256"] != threshold_data["checkpoint_sha256"]
    ):
        raise ValueError("The export and frozen threshold must share a checkpoint.")
    threshold = threshold_data["threshold"]
    if not math.isfinite(threshold) or threshold_data["source"] != "held_out_normal_training":
        raise ValueError("Evaluation requires a frozen normal-validation threshold.")
    source_files = [Path(__file__), Path(__file__).parent.parent / "dataio/images.py"]
    identity = {
        "dataset_sha256": artifact.dataset_sha256,
        "preprocessing_sha256": artifact.preprocessing_sha256,
        "evaluator_sha256": digest(
            {
                "files": {p.name: sha256(p) for p in source_files},
                "ort": ort.__version__,
                "torch": str(torch.__version__),
                "protocol": "original_geometry_v1",
            }
        ),
        "threshold_sha256": threshold_ref.sha256,
        "canonical_provider": "CPUExecutionProvider",
    }
    manifest = json.loads((project / str(dataset_config["manifest"])).read_text())
    prefix = f"{dataset_config['category']}/test/"
    names = sorted(
        n for n in manifest["file_hashes"] if n.startswith(prefix) and n.endswith(".png")
    )
    actual = {
        p.relative_to(root).as_posix()
        for p in (root / str(dataset_config["category"]) / "test").rglob("*.png")
    }
    if actual != set(names) or not names:
        raise ValueError("The complete test inventory must match the verified manifest.")
    total_pixels = 0
    shapes = {}
    for name in names:
        path = root / name
        if sha256(path) != manifest["file_hashes"][name]:
            raise ValueError(f"Test image hash mismatch: {name}")
        with Image.open(path) as image:
            shapes[name] = image.size
            total_pixels += image.width * image.height
        category, _, label, filename = Path(name).parts
        if label != "good":
            mask = f"{category}/ground_truth/{label}/{Path(filename).stem}_mask.png"
            if sha256(root / mask) != manifest["file_hashes"][mask]:
                raise ValueError(f"Mask hash mismatch: {mask}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(artifact.path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if [o.name for o in session.get_outputs()] != ["image_score", "anomaly_map"]:
        raise ValueError("Unexpected model outputs.")
    cache_dir = project / str(settings["predictions_cache"]) / artifact.sha256

    def perform() -> dict:
        cache_dir.mkdir(parents=True, exist_ok=False)
        rows = []
        with tempfile.TemporaryDirectory(prefix="pixel-metrics-", dir=cache_dir) as scratch:
            pixels = np.memmap(
                Path(scratch) / "scores.bin", mode="w+", dtype=np.float32, shape=(total_pixels,)
            )
            targets = np.memmap(
                Path(scratch) / "labels.bin", mode="w+", dtype=np.uint8, shape=(total_pixels,)
            )
            offset = 0
            for index, name in enumerate(names):
                x = load_image(root / name, dataset_config)[None]
                score, small_map = session.run(["image_score", "anomaly_map"], {"image": x})
                if (
                    score.shape != (1,)
                    or small_map.shape != (1, 1, 256, 256)
                    or not np.isfinite(score).all()
                    or not np.isfinite(small_map).all()
                ):
                    raise ValueError(f"Invalid inference outputs: {name}")
                category, _, label, filename = Path(name).parts
                width, height = shapes[name]
                # Scores return to the original image geometry; ground truth is never resampled.
                restored = F.interpolate(
                    torch.from_numpy(small_map),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).numpy()[0, 0]
                mask = np.zeros((height, width), dtype=np.uint8)
                if label != "good":
                    mask_name = f"{category}/ground_truth/{label}/{Path(filename).stem}_mask.png"
                    with Image.open(root / mask_name) as image:
                        mask = (np.asarray(image.convert("L")) > 0).astype(np.uint8)
                if mask.shape != restored.shape:
                    raise ValueError(f"Mask geometry mismatch: {name}")
                count = mask.size
                pixels[offset : offset + count] = restored.ravel()
                targets[offset : offset + count] = mask.ravel()
                offset += count
                prediction = cache_dir / f"{index:04d}.npz"
                np.savez_compressed(prediction, anomaly_map=restored, mask=mask)
                rows.append(
                    {
                        "image_id": name,
                        "label": int(label != "good"),
                        "defect_type": label,
                        "score": float(score[0]),
                        "defect_pixels": int(mask.sum()),
                        "total_pixels": count,
                        "prediction_file": prediction.name,
                        "prediction_sha256": sha256(prediction),
                    }
                )
                if (index + 1) % 10 == 0 or index + 1 == len(names):
                    print(f"Evaluated {index + 1}/{len(names)} test images.", flush=True)
            if offset != total_pixels:
                raise ValueError("Incomplete pixel evaluation.")
            labels = np.array([r["label"] for r in rows])
            scores = np.array([r["score"] for r in rows])
            print("Computing exact pixel AUROC at original image resolution.", flush=True)
            metrics = dict(
                image_auroc=float(roc_auc_score(labels, scores)),
                pixel_auroc=float(roc_auc_score(targets, pixels)),
                threshold=threshold,
                **confusion(labels, scores, threshold),
            )
            del pixels, targets
        metrics.update(
            image_count=len(rows),
            pixel_count=total_pixels,
            artifact_sha256=artifact.sha256,
            threshold_sha256=threshold_ref.sha256,
            evaluator_sha256=identity["evaluator_sha256"],
            pixel_geometry="original_image_resolution",
            threshold_source="held_out_normal_training",
        )
        samples_ref = record(
            cache_dir / "predictions.json",
            {
                "images": rows,
                "artifact_sha256": artifact.sha256,
                "identity": identity,
                "full_test_split_complete": True,
            },
        )
        metrics_ref = record(cache_dir / "metrics.json", metrics)
        return {"samples": serialize(samples_ref), "metrics": serialize(metrics_ref)}

    result = reserve_once(ledger_path, artifact.sha256, identity, perform)
    samples_ref, metrics_ref = verify_cached_result(result)
    return PredictionCache(artifact, samples_ref, metrics_ref, identity["evaluator_sha256"])


def require_quality(cache: PredictionCache, floors: JSON) -> EvidenceRef:
    """Compare cached AUROCs with configured floors; refuse missing, nonfinite, or below-floor metrics without rerunning inference."""
    metrics = read(cache.metrics)
    failures = []
    for name in ("image_auroc", "pixel_auroc"):
        value = metrics.get(name)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
            or value < floors[name]
        ):
            failures.append(f"{name}={value} (required >= {floors[name]})")
    if failures:
        raise ValueError("Gate 1 quality floor failed: " + "; ".join(failures))
    return cache.metrics
