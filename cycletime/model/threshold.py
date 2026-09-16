"""This module freezes the initial threshold before test inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cycletime.contracts import JSON, EvidenceRef, PredictionCache
from cycletime.dataio.calibration import validate_partitions
from cycletime.dataio.images import load_image
from cycletime.evidence import digest, read, record, reference
from cycletime.model.train import load_checkpoint


def initial_threshold(
    checkpoint: EvidenceRef, normal_validation: EvidenceRef, config: JSON
) -> EvidenceRef:
    """Freeze a normal-validation quantile before test inference; refuse test labels and claims of validated defect recall."""
    parts = validate_partitions(normal_validation)
    data = parts["dataset_config"]
    root = Path(data["_project"]) / data["root"]
    if (
        config["initial_method"] != "normal_validation_quantile"
        or not 0 < config["normal_quantile"] < 1
    ):
        raise ValueError("Unsupported initial threshold rule.")
    identity = digest(
        {"checkpoint": checkpoint.sha256, "partitions": normal_validation.sha256, "config": config}
    )
    destination = Path(data["_project"]) / "artifacts/training/threshold.json"
    if destination.exists():
        ref = reference(destination)
        if read(ref)["identity"] != identity:
            raise ValueError("The frozen threshold belongs to another experiment.")
        return ref
    model, _ = load_checkpoint(checkpoint)
    scores = []
    with torch.inference_mode():
        for name in parts["validation"]:
            value, _ = model(torch.from_numpy(load_image(root / name, data)[None]))
            scores.append(float(value.item()))
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite validation scores.")
    threshold = float(np.quantile(scores, config["normal_quantile"], method="linear"))
    return record(
        destination,
        {
            "identity": identity,
            "threshold": threshold,
            "scores": scores,
            "validation_ids": parts["validation"],
            "source": "held_out_normal_training",
            "quantile": config["normal_quantile"],
            "checkpoint_sha256": checkpoint.sha256,
            "defect_recall_validated": False,
        },
    )


def threshold_metrics(cache: PredictionCache, thresholds: list[float]) -> EvidenceRef:
    """Compute threshold metrics from cached predictions; refuse new inference and nonfinite thresholds."""
    from cycletime.model.evaluate import confusion

    samples = read(cache.samples)
    labels = np.array([r["label"] for r in samples["images"]])
    scores = np.array([r["score"] for r in samples["images"]])
    if not thresholds or not np.isfinite(thresholds).all():
        raise ValueError("Thresholds must be finite and nonempty.")
    return record(
        cache.samples.path.parent / "threshold_metrics.json",
        {
            "samples_sha256": cache.samples.sha256,
            "retrospective_research_only": True,
            "points": [dict(threshold=t, **confusion(labels, scores, t)) for t in thresholds],
        },
    )
