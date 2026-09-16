"""These tests protect exactly-once evaluation and cached evidence integrity."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cycletime.contracts import Artifact, PredictionCache
from cycletime.dataio.archive import sha256
from cycletime.dataio.images import preprocessing_digest
from cycletime.evidence import read, record, serialize
from cycletime.model.evaluate import (
    confusion,
    evaluate_once,
    require_quality,
    reserve_once,
    verify_cached_result,
)


def result():
    return {"samples": {"sha256": "sample-hash"}, "metrics": {"sha256": "metrics-hash"}}


def test_reserve_once_atomically(tmp_path):
    entered, release = Event(), Event()
    ledger = tmp_path / "ledger.json"

    def work():
        entered.set()
        assert release.wait(10)
        return result()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(reserve_once, ledger, "artifact", {"data": "verified"}, work)
        assert entered.wait(10)
        try:
            with pytest.raises(ValueError, match="already has a started"):
                reserve_once(ledger, "artifact", {"data": "verified"}, result)
        finally:
            release.set()
        assert first.result() == result()
    entry = json.loads(ledger.read_text())["runs"]["artifact"]
    assert (entry["started_count"], entry["completed_count"], entry["status"]) == (
        1,
        1,
        "completed",
    )


def test_reuse_matching_cache(tmp_path):
    ledger = tmp_path / "ledger.json"
    identity = {"data": "verified"}
    reserve_once(ledger, "artifact", identity, result)
    assert (
        reserve_once(ledger, "artifact", identity, lambda: pytest.fail("Repeated work")) == result()
    )
    with pytest.raises(ValueError, match="forbidden"):
        reserve_once(ledger, "artifact", {"data": "changed"}, result)


def test_preserve_failed_run(tmp_path):
    ledger = tmp_path / "ledger.json"

    def fail():
        raise RuntimeError("Simulated inference failure")

    with pytest.raises(RuntimeError):
        reserve_once(ledger, "artifact", {}, fail)
    entry = json.loads(ledger.read_text())["runs"]["artifact"]
    assert entry["status"] == "failed" and entry["started_count"] == 1
    assert entry["completed_count"] == 0
    with pytest.raises(ValueError, match="forbidden"):
        reserve_once(ledger, "artifact", {}, result)


def test_compute_metrics():
    metrics = confusion(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.6, 0.2]), 0.6)
    assert metrics == {
        "tp": 1,
        "fn": 1,
        "fp": 1,
        "tn": 1,
        "recall": 0.5,
        "false_positive_rate": 0.5,
        "f1": 0.5,
    }
    with pytest.raises(ValueError):
        confusion(np.array([1, 1]), np.array([0.1, 0.2]), 0.5)
    with pytest.raises(ValueError):
        confusion(np.array([0, 1]), np.array([np.nan, 0.2]), 0.5)


def test_corrupt_prediction_cache_is_refused(tmp_path):
    prediction = tmp_path / "0.npz"
    prediction.write_bytes(b"original")
    samples = record(
        tmp_path / "samples.json",
        {"images": [{"prediction_file": prediction.name, "prediction_sha256": sha256(prediction)}]},
    )
    metrics = record(tmp_path / "metrics.json", {"image_auroc": 1.0})
    refs = {"samples": serialize(samples), "metrics": serialize(metrics)}
    verify_cached_result(refs)
    prediction.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Cached prediction"):
        verify_cached_result(refs)


@pytest.mark.parametrize("value", [0.89, None, "NaN"])
def test_quality_floor_refuses_bad_metrics(tmp_path, value):
    metrics = record(tmp_path / "metrics.json", {"image_auroc": value, "pixel_auroc": 1.0})
    cache = PredictionCache(None, metrics, metrics, "evaluator")
    with pytest.raises(ValueError, match="quality floor failed"):
        require_quality(cache, {"image_auroc": 0.9, "pixel_auroc": 0.9})


def test_full_evaluation_caches_exact_geometry_once(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    hashes = {}
    for relative, color in [
        ("bottle/test/good/0.png", 0),
        ("bottle/test/crack/0.png", 255),
        ("bottle/ground_truth/crack/0_mask.png", 255),
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(
            "L" if "ground_truth" in relative else "RGB",
            (12, 9),
            color if "ground_truth" in relative else (color, color, color),
        ).save(path)
        hashes[relative] = sha256(path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"file_hashes": hashes}))
    threshold = record(
        tmp_path / "threshold.json",
        {"threshold": 0.5, "source": "held_out_normal_training", "checkpoint_sha256": "checkpoint"},
    )
    config = {
        "_project": str(tmp_path),
        "_dataset_sha256": "dataset",
        "root": "raw",
        "manifest": "manifest.json",
        "category": "bottle",
        "_threshold": serialize(threshold),
        "test": {
            "canonical_provider": "CPUExecutionProvider",
            "maximum_full_evaluations_per_artifact": 1,
            "predictions_cache": "cache",
        },
        "input": {
            "color": "RGB",
            "layout": "NCHW",
            "image_interpolation": "bilinear",
            "resized_width": 256,
            "resized_height": 256,
            "mean": [0, 0, 0],
            "std": [1, 1, 1],
        },
    }
    path = tmp_path / "model.onnx"
    path.write_bytes(b"Synthetic artifact")
    artifact = Artifact(path, sha256(path), "fp32", preprocessing_digest(config), "dataset")
    record(
        path.with_suffix(".json"),
        {"artifact": {"sha256": artifact.sha256}, "checkpoint_sha256": "checkpoint"},
    )
    calls = []

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def get_outputs(self):
            return [SimpleNamespace(name="image_score"), SimpleNamespace(name="anomaly_map")]

        def run(self, names, inputs):
            calls.append(1)
            score = float(inputs["image"].mean())
            return [np.array([score], np.float32), np.full((1, 1, 256, 256), score, np.float32)]

    monkeypatch.setattr("cycletime.model.evaluate.ort.InferenceSession", Session)
    ledger = tmp_path / "ledger.json"
    cache = evaluate_once(artifact, config, ledger)
    metrics = read(cache.metrics)
    assert metrics["image_auroc"] == metrics["pixel_auroc"] == metrics["recall"] == 1.0
    assert metrics["pixel_count"] == 216
    assert len(calls) == 2
    for row in read(cache.samples)["images"]:
        with np.load(cache.samples.path.parent / row["prediction_file"]) as arrays:
            assert arrays["mask"].shape == arrays["anomaly_map"].shape == (9, 12)
    cached = evaluate_once(artifact, config, ledger)
    assert cached == cache and len(calls) == 2
