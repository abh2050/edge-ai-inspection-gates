"""These tests require calibration to consume distinct normal fit images."""

import numpy as np
import pytest
from PIL import Image

from cycletime.dataio.archive import sha256
from cycletime.dataio.calibration import calibration_reader
from cycletime.evidence import record


def test_exclude_test_calibration(tmp_path):
    names = [f"bottle/train/good/{i}.png" for i in range(3)]
    hashes = {}
    for index, name in enumerate(names):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (index, index, index)).save(path)
        hashes[name] = sha256(path)
    data = {
        "fit": names[:2],
        "validation": names[2:],
        "file_hashes": hashes,
        "dataset_config": {
            "_project": str(tmp_path),
            "root": ".",
            "category": "bottle",
            "input": {
                "color": "RGB",
                "layout": "NCHW",
                "image_interpolation": "bilinear",
                "resized_width": 256,
                "resized_height": 256,
                "mean": [0, 0, 0],
                "std": [1, 1, 1],
            },
        },
    }
    ref = record(tmp_path / "partitions.json", data)
    reader = calibration_reader(ref, 2)
    first, second = reader.get_next()["image"], reader.get_next()["image"]
    assert first.shape == second.shape == (1, 3, 256, 256)
    assert np.all(first == 0) and np.allclose(second, 1 / 255)
    assert reader.get_next() is None
    with pytest.raises(ValueError, match="Insufficient"):
        calibration_reader(ref, 3)
    data["fit"] = [names[0], names[2]]
    bad = record(tmp_path / "bad.json", data)
    with pytest.raises(ValueError, match="overlapping"):
        calibration_reader(bad, 2)
