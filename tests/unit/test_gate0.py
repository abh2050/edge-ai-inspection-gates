"""These tests exercise Gate 0 integrity checks with synthetic local data."""

import io
import json
import tarfile

import pytest
from PIL import Image

from cycletime.dataio.archive import approved_manifest, inspect_archive, sha256
from cycletime.dataio.verify import OBJECTS, TEXTURES, validate_tree


def make_archive(path, name="safe.txt", link=False):
    with tarfile.open(path, "w") as archive:
        member = tarfile.TarInfo(name)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/outside"
            archive.addfile(member)
        else:
            member.size = 3
            archive.addfile(member, io.BytesIO(b"abc"))


def test_digest_mismatch_precedes_extraction(tmp_path):
    archive = tmp_path / "input.tar"
    make_archive(archive)
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(ValueError, match="SHA256"):
        inspect_archive(archive, "0" * 64, destination)
    assert not list(destination.iterdir())


@pytest.mark.parametrize("name,link", [("../escape", False), ("/escape", False), ("link", True)])
def test_unsafe_archive(tmp_path, name, link):
    archive = tmp_path / "input.tar"
    make_archive(archive, name, link)
    with pytest.raises(ValueError, match="Unsafe"):
        inspect_archive(archive, sha256(archive))


def test_missing_provenance(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"archive_sha256": None}))
    with pytest.raises(ValueError, match="approved SHA256"):
        approved_manifest(path)


@pytest.fixture
def dataset(tmp_path):
    config = {
        "objects": sorted(OBJECTS),
        "textures": sorted(TEXTURES),
        "expected_category_count": 15,
    }
    hashes = {}
    for i, category in enumerate(sorted(OBJECTS | TEXTURES)):
        for j, suffix in enumerate(
            [
                "train/good/000.png",
                "test/good/000.png",
                "test/crack/000.png",
                "ground_truth/crack/000_mask.png",
            ]
        ):
            path = tmp_path / category / suffix
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (i, j, 0)).save(path)
            hashes[path.relative_to(tmp_path).as_posix()] = sha256(path)
    return tmp_path, hashes, config


def test_complete_dataset(dataset):
    counts = validate_tree(*dataset)
    assert len(counts) == 15
    assert counts["bottle"] == {"train": 1, "test": 2, "ground_truth": 1}


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("extra_category", "Category mismatch"),
        ("missing_file", "inventory"),
        ("modified", "hash mismatch"),
        ("train_defect", "Training defect"),
        ("missing_mask", "Missing or mismatched"),
        ("overlap", "overlap"),
        ("corrupt", None),
        ("mask_size", "mismatched"),
    ],
)
def test_reject_invalid_dataset(dataset, mutation, match):
    root, hashes, config = dataset
    path = root / "bottle/train/good/000.png"
    mask = root / "bottle/ground_truth/crack/000_mask.png"
    if mutation == "extra_category":
        (root / "unexpected").mkdir()
    elif mutation == "missing_file":
        path.unlink()
    elif mutation == "modified":
        path.write_bytes(b"changed")
    elif mutation == "train_defect":
        target = root / "bottle/train/crack/000.png"
        target.parent.mkdir()
        path.rename(target)
        hashes[target.relative_to(root).as_posix()] = hashes.pop(path.relative_to(root).as_posix())
    elif mutation == "missing_mask":
        mask.unlink()
        del hashes[mask.relative_to(root).as_posix()]
    elif mutation == "overlap":
        target = root / "bottle/test/good/000.png"
        target.write_bytes(path.read_bytes())
        hashes[target.relative_to(root).as_posix()] = sha256(target)
    elif mutation == "corrupt":
        path.write_bytes(b"not a png")
        hashes[path.relative_to(root).as_posix()] = sha256(path)
    elif mutation == "mask_size":
        Image.new("L", (4, 4)).save(mask)
        hashes[mask.relative_to(root).as_posix()] = sha256(mask)
    with pytest.raises((ValueError, OSError), match=match):
        validate_tree(root, hashes, config)


def test_insufficient_space_prevents_extraction(tmp_path, monkeypatch):
    from collections import namedtuple

    from cycletime.dataio.archive import require_extraction_space

    archive = tmp_path / "input.tar"
    make_archive(archive)
    usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr("cycletime.dataio.archive.shutil.disk_usage", lambda _: usage(1, 1, 0))
    with pytest.raises(ValueError, match="Insufficient disk space"):
        require_extraction_space(archive, sha256(archive), tmp_path)


def test_category_metadata_is_hashed(dataset):
    root, hashes, config = dataset
    for name in ("readme.txt", "license.txt"):
        path = root / "bottle" / name
        path.write_text("Dataset metadata.")
        hashes[path.relative_to(root).as_posix()] = sha256(path)
    assert len(validate_tree(root, hashes, config)) == 15
    (root / "bottle" / "license.txt").write_text("Changed metadata.")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_tree(root, hashes, config)


def test_unexpected_category_file_is_rejected(dataset):
    root, hashes, config = dataset
    path = root / "bottle" / "extra.txt"
    path.write_text("Unexpected file.")
    hashes[path.relative_to(root).as_posix()] = sha256(path)
    with pytest.raises(ValueError, match="Unexpected sample path"):
        validate_tree(root, hashes, config)
