"""This module verifies archive identity, image integrity, category inventory, and splits."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image

from cycletime.contracts import JSON, EvidenceRef
from cycletime.dataio.archive import (
    approved_manifest,
    inspect_archive,
    require_extraction_space,
    sha256,
)

OBJECTS = {
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
}
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}


def write_json(path: Path, data: dict) -> None:
    """Replace a JSON file atomically; refuse writes through a symlink."""
    if path.is_symlink():
        raise ValueError(f"Refusing symbolic link: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_tree(root: Path, hashes: dict[str, str], dataset_config: JSON) -> dict:
    """Verify every file and image, all categories, normal training, masks, and split disjointness; refuse missing, extra, corrupt, or overlapping samples."""
    expected = OBJECTS | TEXTURES
    if (
        set(dataset_config["objects"]) != OBJECTS
        or set(dataset_config["textures"]) != TEXTURES
        or dataset_config["expected_category_count"] != 15
    ):
        raise ValueError("The configuration must describe ten objects and five textures.")
    if root.is_symlink():
        raise ValueError("The dataset root cannot be a symbolic link.")
    categories = {p.name for p in root.iterdir() if p.is_dir()}
    if categories != expected:
        raise ValueError(
            f"Category mismatch: missing={expected - categories}, extra={categories - expected}"
        )
    actual = set()
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValueError(f"Symbolic link in dataset: {p}")
        if p.is_file():
            actual.add(p.relative_to(root).as_posix())
    if actual != set(hashes):
        raise ValueError("Extracted file inventory differs from the verified archive.")
    counts = Counter()
    dimensions = {}
    train_digests = set()
    test_digests = set()
    masks = set()
    defects = set()
    for name, expected_digest in sorted(hashes.items()):
        path = root / name
        digest = sha256(path)
        if digest != expected_digest:
            raise ValueError(f"File hash mismatch: {name}")
        parts = Path(name).parts
        if parts[0] not in expected:
            if len(parts) != 1 or path.suffix.lower() not in {".txt", ".pdf", ".md"}:
                raise ValueError(f"Unexpected dataset file: {name}")
            continue
        if len(parts) == 2 and parts[1] in {"readme.txt", "license.txt"}:
            continue
        if path.suffix.lower() != ".png" or len(parts) != 4:
            raise ValueError(f"Unexpected sample path: {name}")
        category, split, label, _ = parts
        if split not in {"train", "test", "ground_truth"}:
            raise ValueError(f"Unexpected split: {name}")
        if split == "train" and label != "good":
            raise ValueError(f"Training defect: {name}")
        if split == "ground_truth" and label == "good":
            raise ValueError(f"Unexpected normal mask: {name}")
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"Expected PNG: {name}")
            image.verify()
        with Image.open(path) as image:
            image.load()
            dimensions[name] = image.size
        counts[(category, split)] += 1
        if split == "train":
            train_digests.add(digest)
        elif split == "test":
            test_digests.add(digest)
            if label != "good":
                defects.add(name)
        else:
            masks.add(name)
    if train_digests & test_digests:
        raise ValueError("Training and test images overlap by content hash.")
    expected_masks = set()
    for name in defects:
        category, _, label, filename = Path(name).parts
        mask = f"{category}/ground_truth/{label}/{Path(filename).stem}_mask.png"
        expected_masks.add(mask)
        if mask not in dimensions or dimensions[mask] != dimensions[name]:
            raise ValueError(f"Missing or mismatched defect mask: {name}")
    if masks != expected_masks:
        raise ValueError("The ground truth contains missing or orphan masks.")
    for category in expected:
        for split in ("train", "test", "ground_truth"):
            if not counts[(category, split)]:
                raise ValueError(f"Empty split: {category}/{split}")
        if not list((root / category / "test" / "good").glob("*.png")):
            raise ValueError(f"Missing normal test images: {category}")
    return {
        c: {s: counts[c, s] for s in ("train", "test", "ground_truth")} for c in sorted(expected)
    }


def verify_dataset(manifest_path: Path, dataset_config: JSON) -> EvidenceRef:
    """Verify the approved archive and extracted samples and emit evidence; refuse incomplete provenance, changed files, corrupt images, and overlapping splits."""
    manifest_path = manifest_path.resolve()
    project = manifest_path.parent.parent
    manifest = approved_manifest(manifest_path)
    if set(manifest["expected_categories"]) != OBJECTS | TEXTURES:
        raise ValueError("The manifest must list all fifteen categories.")
    archive = project / manifest["archive_path"]
    root = project / str(dataset_config["root"])
    if not root.is_relative_to(project) or root.is_symlink():
        raise ValueError("The dataset root must remain inside the project.")
    root.parent.mkdir(parents=True, exist_ok=True)
    extracted = root
    if extracted.exists():
        hashes = inspect_archive(archive, manifest["archive_sha256"])
        counts = validate_tree(extracted, hashes, dataset_config)
    else:
        require_extraction_space(archive, manifest["archive_sha256"], root.parent)
        with tempfile.TemporaryDirectory(prefix=".gate0-", dir=root.parent) as directory:
            staging = Path(directory)
            hashes = inspect_archive(archive, manifest["archive_sha256"], staging)
            counts = validate_tree(staging, hashes, dataset_config)
            staging.rename(extracted)
    if manifest.get("file_hashes") and manifest["file_hashes"] != hashes:
        raise ValueError("Recorded file hashes differ from the verified archive.")
    now = datetime.now(UTC).isoformat()
    manifest.update(file_hashes=hashes, status="verified", verified_at_utc=now)
    write_json(manifest_path, manifest)
    run_id = str(uuid4())
    dataset_digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    report = project / "artifacts" / "gate0.json"
    write_json(
        report,
        {
            "schema_version": 1,
            "gate": 0,
            "status": "passed",
            "created_at_utc": now,
            "run_id": run_id,
            "archive_sha256": manifest["archive_sha256"],
            "manifest_sha256": sha256(manifest_path),
            "dataset_sha256": dataset_digest,
            "dataset_root": str(extracted.relative_to(project)),
            "categories": counts,
            "file_count": len(hashes),
            "image_count": sum(v["train"] + v["test"] for v in counts.values()),
            "mask_count": sum(v["ground_truth"] for v in counts.values()),
            "checks": [
                "archive_sha256",
                "safe_members",
                "file_hashes",
                "15_categories",
                "normal_training_only",
                "png_decode",
                "mask_coverage_and_dimensions",
                "train_test_disjoint",
            ],
            "verifier_sha256": sha256(Path(__file__)),
            "dataset_config": dataset_config,
        },
    )
    return EvidenceRef(report, sha256(report), now, run_id)
