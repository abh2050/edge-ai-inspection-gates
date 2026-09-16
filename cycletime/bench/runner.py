"""This module measures preprocessing, synchronous inference, and postprocessing."""

from __future__ import annotations

import gc
import math
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from cycletime.contracts import JSON, Artifact, EvidenceRef, Measurement, ProviderSpec
from cycletime.dataio.archive import sha256
from cycletime.dataio.calibration import validate_partitions
from cycletime.dataio.images import decode_image, preprocess_decoded
from cycletime.evidence import read, record, reference, serialize


def measure_iterations(run, warmups: int, iterations: int, clock=perf_counter_ns) -> list[int]:
    """Discard exactly the configured warmups and time every synchronous completion; refuse too few samples."""
    if warmups != 20 or iterations < 500:
        raise ValueError(
            "Measurements require exactly 20 warmups and at least 500 timed iterations."
        )
    for index in range(warmups):
        run(index)
    samples = []
    for index in range(iterations):
        start = clock()
        run(index)
        end = clock()
        duration = end - start
        if duration <= 0:
            raise ValueError("The monotonic clock produced a nonpositive duration.")
        samples.append(duration)
    return samples


def environment_record(project: Path, config: JSON) -> EvidenceRef:
    """Record a sanitized local host description; refuse a missing dependency lock."""
    lock = project / "uv.lock"
    if not lock.is_file():
        raise ValueError("Benchmarking requires the dependency lock.")
    destination = project / config["outputs"]["environment"]
    if destination.exists():
        existing = reference(destination)
        data = read(existing)
        expected_threads = {
            "intra_op": config["intra_op_threads"],
            "inter_op": config["inter_op_threads"],
            "concurrency": config["concurrency"],
        }
        if (
            data["dependency_lock_sha256"] != sha256(lock)
            or data["thread_settings"] != expected_threads
        ):
            raise ValueError("Existing environment evidence does not match this benchmark.")
        return existing
    chip = "unavailable"
    try:
        output = subprocess.run(
            ["system_profiler", "-json", "SPHardwareDataType"],
            check=True,
            capture_output=True,
            text=True,
        )
        hardware = __import__("json").loads(output.stdout)["SPHardwareDataType"][0]
        chip = hardware.get("chip_type", "unavailable")
        model = hardware.get("machine_model", "unavailable")
        memory = hardware.get("physical_memory", "unavailable")
    except (OSError, subprocess.SubprocessError, KeyError, ValueError):
        model = memory = "unavailable"
    return record(
        destination,
        {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "macos_version": platform.mac_ver()[0],
            "model": model,
            "chip": chip,
            "memory": memory,
            "python_version": sys.version.split()[0],
            "onnxruntime_version": __import__("onnxruntime").__version__,
            "dependency_lock_sha256": sha256(lock),
            "thread_settings": {
                "intra_op": config["intra_op_threads"],
                "inter_op": config["inter_op_threads"],
                "concurrency": config["concurrency"],
            },
            "power_and_ambient_conditions": "not recorded; research microbenchmark only",
        },
    )


def benchmark(
    artifact: Artifact, provider: ProviderSpec, inputs: EvidenceRef, config: JSON
) -> Measurement:
    """Measure decoded batch-one inputs locally; refuse async timing, estimates, missing samples, or enabled agent work."""
    from cycletime.bench.providers import create_session

    if (
        config["clock"] != "perf_counter_ns"
        or config["measurement_scope"] != "preprocess_infer_postprocess"
    ):
        raise ValueError(
            "The benchmark requires the configured monotonic clock and full model scope."
        )
    parts = validate_partitions(inputs)
    dataset = parts["dataset_config"]
    project = Path(dataset["_project"])
    if __import__("yaml").safe_load((project / "config/agent.yaml").read_text())["enabled"]:
        raise ValueError("The local agent must remain disabled during measurements.")
    names = parts["validation"]
    if not names:
        raise ValueError("Benchmark inputs require held-out normal validation images.")
    root = project / dataset["root"]
    decoded = [decode_image(root / name) for name in names]
    input_hashes = {name: parts["file_hashes"][name] for name in names}
    session, placement = create_session(artifact, provider, config)
    environment = environment_record(project, config)
    observed = 0.0

    def run(index: int) -> None:
        nonlocal observed
        image = preprocess_decoded(decoded[index % len(decoded)], dataset)[None]
        outputs = session.run(["image_score", "anomaly_map"], {"image": image})
        if len(outputs) != 2 or outputs[0].shape != (1,) or outputs[1].shape != (1, 1, 256, 256):
            raise ValueError("Synchronous inference returned unexpected outputs.")
        observed += float(outputs[0][0]) + float(np.max(outputs[1]))
        if not math.isfinite(observed):
            raise ValueError("Postprocessing observed a nonfinite output.")

    started = datetime.now(UTC).isoformat()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        samples_ns = measure_iterations(
            run,
            int(config["warmup_iterations"]),
            int(config["timed_iterations"]),
        )
    finally:
        if was_enabled:
            gc.enable()
    completed = datetime.now(UTC).isoformat()
    samples = record(
        project / config["outputs"]["samples"] / provider.id / f"{artifact.path.stem}.json",
        {
            "artifact_sha256": artifact.sha256,
            "precision": artifact.precision,
            "provider_id": provider.id,
            "provider_name": provider.name,
            "started_at_utc": started,
            "completed_at_utc": completed,
            "warmup_iterations": config["warmup_iterations"],
            "timed_iterations": config["timed_iterations"],
            "batch_size": config["batch_size"],
            "measurement_scope": config["measurement_scope"],
            "clock": config["clock"],
            "latencies_ns": samples_ns,
            "input_ids": names,
            "input_hashes": input_hashes,
            "decoded_before_timing": True,
            "synchronous_completion": True,
            "postprocess_observation": observed,
            "placement": serialize(placement),
            "environment": serialize(environment),
        },
    )
    values_ms = np.asarray(samples_ns, dtype=np.float64) / 1_000_000
    quantiles = np.quantile(values_ms, [0.5, 0.95, 0.99], method=config["quantile_method"])
    summary = record(
        samples.path.with_name(f"{artifact.path.stem}-summary.json"),
        {
            "samples": serialize(samples),
            "sample_count": len(samples_ns),
            "p50_ms": float(quantiles[0]),
            "p95_ms": float(quantiles[1]),
            "p99_ms": float(quantiles[2]),
            "quantile_method": config["quantile_method"],
            "capacity_from_p99_parts_per_minute": float(60_000 / quantiles[2]),
            "cycle_allowance_ms": float(60_000 / config["line_rate_parts_per_minute"]),
            "research_cycle_feasible": bool(
                quantiles[2] <= 60_000 / config["line_rate_parts_per_minute"]
            ),
        },
    )
    return Measurement(artifact, provider, samples, environment, placement, summary)
