"""Render one documentation figure from recorded predictions and the local MVTec AD sample images.

ADR 0009 permits this single derived figure. The figure carries dataset attribution and the
CC BY-NC-SA 4.0 notice, and the repository still ships no dataset archive and no trained weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT / "docs/screenshots/00-detection-example.png"
NORMAL = "bottle/test/good/000.png"
DEFECTIVE = "bottle/test/broken_large/000.png"


def cached(gate1: dict, image_id: str) -> tuple[dict, np.ndarray]:
    """Return one recorded prediction row and its cached anomaly map."""
    predictions = Path(gate1["predictions"]["path"])
    rows = json.loads(predictions.read_text())["images"]
    row = next(item for item in rows if item["image_id"] == image_id)
    with np.load(predictions.parent / row["prediction_file"]) as payload:
        return row, payload[payload.files[0]]


def main() -> None:
    """Draw the normal part, the defective part, and the recorded anomaly map for the defect."""
    gate1 = json.loads((PROJECT / "artifacts/gate1.json").read_text())
    metrics = json.loads(Path(gate1["metrics"]["path"]).read_text())
    threshold = metrics["threshold"]
    dataset = json.loads((PROJECT / "artifacts/partitions.json").read_text())["dataset_config"]
    root = PROJECT / dataset["root"]

    normal_row, _ = cached(gate1, NORMAL)
    defect_row, defect_map = cached(gate1, DEFECTIVE)
    normal = Image.open(root / NORMAL).convert("RGB")
    defective = Image.open(root / DEFECTIVE).convert("RGB")
    heat = np.squeeze(defect_map)

    figure, axes = plt.subplots(1, 3, figsize=(11, 4.1))
    figure.patch.set_facecolor("white")
    panels = [
        (normal, None, "Normal part", f"score {normal_row['score']:.3f} · accepted"),
        (defective, None, "Defective part", f"score {defect_row['score']:.3f} · rejected"),
        (defective, heat, "Recorded anomaly map", f"{defect_row['defect_pixels']:,} annotated defect pixels"),
    ]
    for axis, (image, overlay, title, caption) in zip(axes, panels, strict=True):
        axis.imshow(image)
        if overlay is not None:
            resized = np.asarray(
                Image.fromarray(overlay.astype(np.float32)).resize(image.size, Image.BILINEAR)
            )
            axis.imshow(resized, cmap="inferno", alpha=0.55)
        axis.set_title(title, fontsize=11, pad=8)
        axis.set_xlabel(caption, fontsize=9, color="#444444")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"MVTec AD bottle category · frozen decision threshold {threshold:.4f} · FP32 artifact",
        fontsize=10.5,
        y=0.99,
    )
    figure.text(
        0.5, 0.018,
        "Images: MVTec Anomaly Detection dataset, CC BY-NC-SA 4.0. Anomaly map read from the recorded gate 1 evaluation.",
        ha="center", fontsize=8.5, color="#666666",
    )
    figure.tight_layout(rect=(0, 0.075, 1, 0.96))
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(DESTINATION, dpi=110)
    plt.close(figure)
    print(f"{DESTINATION.relative_to(PROJECT)}: {DESTINATION.stat().st_size} bytes")


if __name__ == "__main__":
    main()
