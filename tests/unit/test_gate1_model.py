"""These tests check the selected network and partition boundaries without test inference."""

import json

import pytest
import torch
from PIL import Image

from cycletime.dataio.archive import sha256
from cycletime.dataio.calibration import build_partitions, calibration_reader, validate_partitions
from cycletime.evidence import read, record
from cycletime.model.network import Detector


def test_teacher_stays_frozen_and_stages_match():
    torch.set_num_threads(2)
    model = Detector()
    x = torch.randn(2, 3, 256, 256)
    shapes = [tuple(v.shape[1:]) for v in model.features(model.teacher.eval(), x)]
    assert shapes == [(24, 32, 32), (48, 16, 16), (576, 8, 8)]
    before = {k: v.clone() for k, v in model.teacher.state_dict().items()}
    model.train()
    loss = torch.stack([d.mean() for d in model.distances(x)]).mean()
    loss.backward()
    assert all(p.grad is None for p in model.teacher.parameters())
    assert any(p.grad is not None for p in model.student.parameters())
    assert all(torch.equal(before[k], v) for k, v in model.teacher.state_dict().items())
    model.eval()
    with torch.inference_mode():
        score, image = model(x[:1])
    assert score.shape == (1,) and image.shape == (1, 1, 256, 256)
    assert torch.equal(score, image.flatten(1).amax(1))


def test_partitions_reuse_stable_ids_and_reject_leakage(tmp_path):
    hashes = {}
    for i in range(10):
        path = tmp_path / f"raw/bottle/train/good/{i}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (i, 0, 0)).save(path)
        hashes[path.relative_to(tmp_path / "raw").as_posix()] = sha256(path)
    (tmp_path / "manifest.json").write_text(json.dumps({"file_hashes": hashes}))
    config = {
        "_project": str(tmp_path),
        "root": "raw",
        "manifest": "manifest.json",
        "category": "bottle",
        "train": {"normal_validation_fraction": 0.2, "allowed_labels": ["good"], "seed": 42},
    }
    ref = build_partitions(config)
    assert build_partitions(config) == ref
    data = validate_partitions(ref)
    assert len(data["fit"]) == 8 and len(data["validation"]) == 2
    with pytest.raises(ValueError, match="Insufficient"):
        calibration_reader(ref, 9)
    data["fit"][0] = data["validation"][0]
    bad = record(tmp_path / "bad.json", data)
    with pytest.raises(ValueError, match="overlapping"):
        validate_partitions(bad)
    data = read(ref)
    data["fit"][0] = "bottle/test/good/0.png"
    bad = record(tmp_path / "bad.json", data)
    with pytest.raises(ValueError, match="only normal training"):
        validate_partitions(bad)
