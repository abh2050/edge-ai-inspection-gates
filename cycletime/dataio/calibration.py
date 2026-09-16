"""This module records deterministic normal training partitions and refuses test leakage."""

from __future__ import annotations

import json
import random
from pathlib import Path

from cycletime.contracts import JSON, EvidenceRef
from cycletime.dataio.archive import sha256
from cycletime.evidence import read, record, reference


def build_partitions(dataset_config: JSON) -> EvidenceRef:
    """Partition verified normal training IDs deterministically; refuse test access and overlapping partitions."""
    project = Path(str(dataset_config["_project"]))
    manifest = json.loads((project / str(dataset_config["manifest"])).read_text())
    category = str(dataset_config["category"])
    settings = dataset_config["train"]
    fraction = settings["normal_validation_fraction"]
    if not 0 < fraction < 1 or settings["allowed_labels"] != ["good"]:
        raise ValueError("Expected a normal-only training split with a nonempty holdout.")
    prefix = f"{category}/train/good/"
    ids = sorted(k for k in manifest["file_hashes"] if k.startswith(prefix) and k.endswith(".png"))
    if len(ids) < 2:
        raise ValueError("Insufficient normal training images.")
    for name in ids:
        if sha256(project / str(dataset_config["root"]) / name) != manifest["file_hashes"][name]:
            raise ValueError(f"Training image hash mismatch: {name}")
    random.Random(settings["seed"]).shuffle(ids)
    count = max(1, round(len(ids) * fraction))
    validation, fit = sorted(ids[:count]), sorted(ids[count:])
    if not fit or set(validation) & set(fit):
        raise ValueError("Empty or overlapping partitions.")
    destination = project / "artifacts/partitions.json"
    if destination.exists():
        cached = reference(destination)
        previous = read(cached)
        if (
            previous["fit"] != fit
            or previous["validation"] != validation
            or previous["dataset_config"] != dataset_config
        ):
            raise ValueError("Existing partitions belong to a different experiment.")
        validate_partitions(cached)
        return cached
    return record(
        destination,
        {
            "fit": fit,
            "validation": validation,
            "dataset_config": dataset_config,
            "file_hashes": {name: manifest["file_hashes"][name] for name in ids},
            "seed": settings["seed"],
            "normal_validation_fraction": fraction,
        },
    )


def validate_partitions(partitions: EvidenceRef) -> dict:
    """Validate normal training IDs and hashes; refuse test or validation leakage and duplicates."""
    data = read(partitions)
    fit, validation = data["fit"], data["validation"]
    prefix = f"{data['dataset_config']['category']}/train/good/"
    if not fit or not validation or len(set(fit + validation)) != len(fit + validation):
        raise ValueError("Invalid or overlapping normal partitions.")
    root = Path(data["dataset_config"]["_project"]) / data["dataset_config"]["root"]
    for name in fit + validation:
        parts = Path(name).parts
        if len(parts) != 4 or ".." in parts or not name.startswith(prefix):
            raise ValueError("Partitions may contain only normal training IDs.")
        if sha256(root / name) != data["file_hashes"][name]:
            raise ValueError(f"Partition image changed: {name}")
    return data


def calibration_reader(partitions: EvidenceRef, sample_count: int) -> object:
    """Read distinct fit images for ONNX calibration; refuse test IDs, overlap, and insufficient samples."""
    from onnxruntime.quantization import CalibrationDataReader

    from cycletime.dataio.images import load_image

    data = validate_partitions(partitions)
    if not 0 < sample_count <= len(data["fit"]):
        raise ValueError("Insufficient distinct fit samples.")
    config = data["dataset_config"]
    root = Path(config["_project"]) / config["root"]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.samples = iter(data["fit"][:sample_count])

        def get_next(self):
            name = next(self.samples, None)
            return None if name is None else {"image": load_image(root / name, config)[None]}

    return Reader()
