"""These tests verify measured sampling, recomputed summaries, and placement policy."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cycletime.bench.report import validate_measurement
from cycletime.bench.runner import measure_iterations
from cycletime.contracts import Artifact, Measurement, ProviderSpec
from cycletime.dataio.archive import sha256
from cycletime.evidence import record


def test_enforce_sampling() -> None:
    """A fake monotonic clock must prove exclusion of exactly twenty warmups and inclusion of at least five hundred timed completions."""
    calls = []
    ticks = iter(range(0, 20_000, 10))
    samples = measure_iterations(calls.append, 20, 500, lambda: next(ticks))
    assert calls == list(range(20)) + list(range(500))
    assert samples == [10] * 500
    with pytest.raises(ValueError, match="exactly 20"):
        measure_iterations(calls.append, 19, 500)
    with pytest.raises(ValueError, match="at least 500"):
        measure_iterations(calls.append, 20, 499)


def fixture_measurement(tmp_path: Path, *, ane: bool = False) -> tuple[Measurement, dict]:
    """Build internally consistent synthetic evidence for verifier unit tests."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    lock = tmp_path / "uv.lock"
    lock.write_text("locked")
    config = {
        "_project": str(tmp_path),
        "warmup_iterations": 20,
        "timed_iterations": 500,
        "batch_size": 1,
        "measurement_scope": "preprocess_infer_postprocess",
        "clock": "perf_counter_ns",
        "quantile_method": "linear",
    }
    artifact_path = tmp_path / "model.onnx"
    artifact_path.write_bytes(b"model")
    artifact = Artifact(artifact_path, sha256(artifact_path), "fp32", "preprocess", "dataset")
    provider = ProviderSpec(
        "coreml" if ane else "cpu",
        "CoreMLExecutionProvider" if ane else "CPUExecutionProvider",
        {},
        ane,
    )
    placement = record(
        tmp_path / "placement.json", {"placement_passed": not ane, "ane_confirmed": False}
    )
    environment = record(tmp_path / "environment.json", {"dependency_lock_sha256": sha256(lock)})
    values = list(range(1_000_000, 1_000_500))
    samples = record(
        tmp_path / "samples.json",
        {
            "artifact_sha256": artifact.sha256,
            "precision": "fp32",
            "provider_id": provider.id,
            "provider_name": provider.name,
            "started_at_utc": "2026-01-01T00:00:00+00:00",
            "completed_at_utc": "2026-01-01T00:01:00+00:00",
            "warmup_iterations": 20,
            "timed_iterations": 500,
            "batch_size": 1,
            "measurement_scope": "preprocess_infer_postprocess",
            "clock": "perf_counter_ns",
            "latencies_ns": values,
            "decoded_before_timing": True,
            "synchronous_completion": True,
        },
    )
    q = np.quantile(np.asarray(values) / 1_000_000, [0.5, 0.95, 0.99], method="linear")
    summary = record(
        tmp_path / "summary.json",
        {"p50_ms": float(q[0]), "p95_ms": float(q[1]), "p99_ms": float(q[2])},
    )
    return Measurement(artifact, provider, samples, environment, placement, summary), config


def test_reject_missing_timestamp_and_changed_quantile(tmp_path: Path) -> None:
    """The verifier must reject stale timestamps and summaries that differ from raw samples."""
    measurement, config = fixture_measurement(tmp_path)
    data = __import__("json").loads(measurement.samples.path.read_text())
    del data["started_at_utc"]
    measurement.samples.path.write_text(__import__("json").dumps(data))
    assert any("stale or missing" in error for error in validate_measurement(measurement, config))

    measurement, config = fixture_measurement(tmp_path / "other")
    bad = record(tmp_path / "other/bad-summary.json", {"p50_ms": 9, "p95_ms": 9, "p99_ms": 9})
    errors = validate_measurement(replace(measurement, summary=bad), config)
    assert "recorded p99_ms differs from raw samples" in errors


def test_require_placement_evidence(tmp_path: Path) -> None:
    """A configured CoreML provider without placement evidence must not produce a claimed ANE measurement."""
    measurement, config = fixture_measurement(tmp_path, ane=True)
    errors = validate_measurement(measurement, config)
    assert "placement policy failed for coreml" in errors
    assert "ANE activity is unconfirmed for coreml" in errors


def test_reject_forbidden_ort_cpu_fallback(tmp_path: Path) -> None:
    """A CoreML row with forbidden ORT CPU fallback must fail even when ANE is not required."""
    measurement, config = fixture_measurement(tmp_path, ane=True)
    provider = replace(measurement.provider, require_ane_evidence=False)
    placement = record(
        tmp_path / "fallback.json",
        {
            "placement_passed": False,
            "ane_confirmed": False,
            "ort_cpu_fallback_observed": True,
            "ort_cpu_fallback_allowed": False,
        },
    )
    errors = validate_measurement(replace(measurement, provider=provider, placement=placement), config)
    assert "ORT CPU fallback is not allowed for coreml" in errors
    assert not any("ANE activity" in error for error in errors)


def test_accept_coreml_without_ane_requirement(tmp_path: Path) -> None:
    """ADR 0006 accepts a clean CoreML row without an ANE claim when ANE is not required."""
    measurement, config = fixture_measurement(tmp_path, ane=True)
    provider = replace(measurement.provider, require_ane_evidence=False)
    placement = record(
        tmp_path / "clean.json",
        {
            "placement_passed": True,
            "ane_confirmed": False,
            "ort_cpu_fallback_observed": False,
            "ort_cpu_fallback_allowed": False,
        },
    )
    samples = __import__("json").loads(measurement.samples.path.read_text())
    assert samples["provider_id"] == provider.id
    errors = validate_measurement(replace(measurement, provider=provider, placement=placement), config)
    assert errors == []


def matrix_config(config: dict, *, coreml_required: bool | None) -> dict:
    """Describe a CPU-required matrix with a CoreML provider of the given requirement."""
    coreml = {"id": "coreml"}
    if coreml_required is not None:
        coreml["required"] = coreml_required
    return dict(
        config,
        required_matrix="every_precision_by_every_required_provider",
        latency_source="local_measured_samples_only",
        providers=[{"id": "cpu", "required": True}, coreml],
    )


def test_matrix_separates_required_and_diagnostic_failures(tmp_path: Path) -> None:
    """Diagnostic failures must stay recorded, while any required failure or gap fails the matrix."""
    import json

    from cycletime.bench.report import MeasurementMatrixError, verify_measurement_matrix

    coreml, config = fixture_measurement(tmp_path / "coreml", ane=True)
    coreml = replace(coreml, provider=replace(coreml.provider, require_ane_evidence=False))
    diagnostic = matrix_config(config, coreml_required=False)
    with pytest.raises(MeasurementMatrixError) as missing:
        verify_measurement_matrix([coreml], diagnostic)
    assert "missing required pair: fp32/cpu" in missing.value.errors
    assert not any("coreml" in error for error in missing.value.errors)
    saved = json.loads((tmp_path / "coreml/artifacts/bench/matrix.json").read_text())
    assert "fp32/coreml: placement policy failed for coreml" in saved["diagnostic_errors"]
    assert "missing diagnostic pair: int8/coreml" in saved["diagnostic_errors"]

    with pytest.raises(MeasurementMatrixError) as required:
        verify_measurement_matrix([coreml], matrix_config(config, coreml_required=True))
    assert "fp32/coreml: placement policy failed for coreml" in required.value.errors


def test_matrix_refuses_undeclared_or_all_diagnostic_providers(tmp_path: Path) -> None:
    """Every provider must declare its role, and at least one provider must decide the gate."""
    from cycletime.bench.report import verify_measurement_matrix

    _, config = fixture_measurement(tmp_path)
    with pytest.raises(ValueError, match="declare whether"):
        verify_measurement_matrix([], matrix_config(config, coreml_required=None))
    none_required = matrix_config(config, coreml_required=False)
    none_required["providers"][0]["required"] = False
    with pytest.raises(ValueError, match="at least one required"):
        verify_measurement_matrix([], none_required)
