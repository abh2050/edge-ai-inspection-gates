"""This module verifies that latency reports contain local measured evidence."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np

from cycletime.contracts import JSON, EvidenceRef, Measurement
from cycletime.dataio.archive import sha256
from cycletime.evidence import read, record, serialize


class MeasurementMatrixError(ValueError):
    """Carry failed matrix evidence without concealing successful measurements."""

    def __init__(self, report: EvidenceRef, errors: list[str]):
        self.report = report
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_measurement(measurement: Measurement, config: JSON) -> list[str]:
    """Recompute one summary and return policy failures; refuse stale or synthetic samples."""
    errors = []
    try:
        samples = read(measurement.samples)
        summary = read(measurement.summary)
        placement = read(measurement.placement) if measurement.placement else None
        environment = read(measurement.environment)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        return [f"stale or missing evidence: {exc}"]
    expected = {
        "artifact_sha256": measurement.artifact.sha256,
        "precision": measurement.artifact.precision,
        "provider_id": measurement.provider.id,
        "provider_name": measurement.provider.name,
        "warmup_iterations": config["warmup_iterations"],
        "timed_iterations": config["timed_iterations"],
        "batch_size": config["batch_size"],
        "measurement_scope": config["measurement_scope"],
        "clock": config["clock"],
    }
    for key, value in expected.items():
        if samples.get(key) != value:
            errors.append(f"{key} mismatch")
    for key in ("started_at_utc", "completed_at_utc"):
        try:
            timestamp = datetime.fromisoformat(samples[key])
            if timestamp.utcoffset() is None:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            errors.append(f"missing UTC timestamp: {key}")
    values = samples.get("latencies_ns")
    if (
        not isinstance(values, list)
        or len(values) < 500
        or len(values) != config["timed_iterations"]
        or any(not isinstance(value, int) or value <= 0 for value in values)
    ):
        errors.append("raw measured sample count or values are invalid")
    else:
        quantiles = np.quantile(
            np.asarray(values, dtype=np.float64) / 1_000_000,
            [0.5, 0.95, 0.99],
            method=config["quantile_method"],
        )
        for key, value in zip(("p50_ms", "p95_ms", "p99_ms"), quantiles, strict=True):
            if not math.isclose(summary.get(key, math.nan), float(value), rel_tol=0, abs_tol=1e-12):
                errors.append(f"recorded {key} differs from raw samples")
    if not samples.get("decoded_before_timing") or not samples.get("synchronous_completion"):
        errors.append("timing scope lacks decoded synchronous completion evidence")
    if environment.get("dependency_lock_sha256") != sha256(Path(config["_project"]) / "uv.lock"):
        errors.append("environment dependency lock differs")
    if placement is None or not placement.get("placement_passed"):
        errors.append(f"placement policy failed for {measurement.provider.id}")
    if placement and placement.get("ort_cpu_fallback_observed") and not placement.get(
        "ort_cpu_fallback_allowed"
    ):
        errors.append(f"ORT CPU fallback is not allowed for {measurement.provider.id}")
    if measurement.provider.require_ane_evidence and not (
        placement and placement.get("ane_confirmed")
    ):
        errors.append(f"ANE activity is unconfirmed for {measurement.provider.id}")
    return errors


def required_provider_ids(config: JSON) -> set[str]:
    """Return providers that decide Gate 3; refuse undeclared requirement or an all-diagnostic matrix."""
    providers = config["providers"]
    if any(not isinstance(provider.get("required"), bool) for provider in providers):
        raise ValueError("Every provider must declare whether Gate 3 requires it.")
    ids = {provider["id"] for provider in providers if provider["required"]}
    if not ids:
        raise ValueError("Gate 3 requires at least one required provider.")
    return ids


def verify_measurement_matrix(measurements: list[Measurement], required: JSON) -> EvidenceRef:
    """Require every pair and recompute quantiles; fail on required-provider errors, record diagnostic-provider errors visibly, and refuse incomplete, stale, synthetic, or unproven placement."""
    if (
        required["required_matrix"] != "every_precision_by_every_required_provider"
        or required["latency_source"] != "local_measured_samples_only"
    ):
        raise ValueError("Gate 3 requires the complete locally measured matrix.")
    required_ids = required_provider_ids(required)
    expected = {
        (precision, provider["id"])
        for precision in ("fp32", "fp16", "int8")
        for provider in required["providers"]
    }
    observed = {(item.artifact.precision, item.provider.id) for item in measurements}
    errors = []
    diagnostic_errors = []
    for precision, provider_id in sorted(expected - observed):
        if provider_id in required_ids:
            errors.append(f"missing required pair: {precision}/{provider_id}")
        else:
            diagnostic_errors.append(f"missing diagnostic pair: {precision}/{provider_id}")
    if len(observed) != len(measurements):
        errors.append("duplicate measurement pair")
    rows = []
    for item in measurements:
        pair_errors = validate_measurement(item, required)
        is_required = item.provider.id in required_ids
        (errors if is_required else diagnostic_errors).extend(
            f"{item.artifact.precision}/{item.provider.id}: {error}" for error in pair_errors
        )
        rows.append(
            {
                "precision": item.artifact.precision,
                "provider_id": item.provider.id,
                "required": is_required,
                "artifact_sha256": item.artifact.sha256,
                "samples": serialize(item.samples),
                "summary": serialize(item.summary),
                "placement": serialize(item.placement) if item.placement else None,
                "errors": pair_errors,
            }
        )
    report = record(
        Path(required["_project"]) / "artifacts/bench/matrix.json",
        {
            "status": "failed" if errors else "passed",
            "required_providers": sorted(required_ids),
            "measurements": rows,
            "errors": errors,
            "diagnostic_errors": diagnostic_errors,
        },
    )
    if errors:
        raise MeasurementMatrixError(report, errors)
    return report


def summarize_thermal(series: list[EvidenceRef], config: JSON) -> EvidenceRef:
    """Validate 30 complete windows per required pair and report minute 1, minute 25, worst minute, achieved rate, and misses; refuse applying the 500-sample microbenchmark requirement to a 45-sample minute and refuse overstating a noisy minute p99."""
    from cycletime.bench.thermal import minute_windows, throttling_attribution

    settings = config["sustained"]
    rate = int(settings["rate_parts_per_minute"])
    minutes = int(settings["duration_minutes"])
    first, later = (int(value) for value in settings["comparison_minutes"])
    allowance_ms = 60_000 / config["line_rate_parts_per_minute"]
    tolerance = float(settings["hold_rate_tolerance_fraction"])
    suspension_tolerance_ms = float(settings["host_suspension_tolerance_ms"])
    required_ids = required_provider_ids(config)
    errors = []
    diagnostic_errors = []
    rows = []
    for ref in series:
        try:
            data = read(ref)
        except (ValueError, FileNotFoundError) as exc:
            errors.append(f"stale or missing sustained evidence: {exc}")
            continue
        pair = f"{data.get('precision')}/{data.get('provider_id')}"
        pair_errors = []
        slots = data.get("slots", [])
        if (
            data.get("rate_parts_per_minute") != rate
            or data.get("duration_minutes") != minutes
            or data.get("scheduled_slot_count") != rate * minutes
            or len(slots) != rate * minutes
            or [slot.get("slot") for slot in slots] != list(range(len(slots)))
        ):
            pair_errors.append("incomplete or reconfigured sustained series")
        try:
            windows = minute_windows(slots, rate, config["quantile_method"])
        except ValueError as exc:
            windows = []
            pair_errors.append(str(exc))
        if windows and windows != data.get("windows"):
            pair_errors.append("recorded windows differ from raw slots")
        if len(windows) != minutes or any(w["scheduled_count"] != rate for w in windows):
            pair_errors.append(f"requires {minutes} complete {rate}-slot windows")
        for key in ("started_at_utc", "completed_at_utc"):
            try:
                if datetime.fromisoformat(data[key]).utcoffset() is None:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                pair_errors.append(f"missing UTC timestamp: {key}")
        completed = sum(w["sample_count"] for w in windows)
        missed = sum(w["missed_slot_count"] for w in windows)
        deadline_misses = sum(w["deadline_miss_count"] for w in windows)
        achieved_rate = completed / minutes if minutes else 0.0
        if missed or deadline_misses:
            pair_errors.append(f"{missed} missed slots and {deadline_misses} deadline misses")
        if any(
            "host_suspended_total_ns" not in slot or "wall_offset_ns" not in slot for slot in slots
        ):
            pair_errors.append("series lacks host suspension evidence")
        suspended_minutes = [
            w["minute"]
            for w in windows
            if w["host_suspended_ms"] > suspension_tolerance_ms
            or w["wall_minus_monotonic_ms"] > suspension_tolerance_ms
        ]
        if suspended_minutes:
            pair_errors.append(f"host slept during minutes {suspended_minutes}")
        if abs(achieved_rate - rate) > tolerance * rate:
            pair_errors.append(f"achieved {achieved_rate:.3f} parts/min outside the held rate")
        p99s = [w["scheduled_to_completed_p99_ms"] for w in windows]
        worst = max(
            (w for w in windows if w["scheduled_to_completed_p99_ms"] is not None),
            key=lambda w: w["scheduled_to_completed_p99_ms"],
            default=None,
        )
        if None in p99s or worst is None:
            pair_errors.append("a reporting window has no completed samples")
        elif worst["scheduled_to_completed_p99_ms"] > allowance_ms:
            pair_errors.append("worst-minute p99 exceeds the research cycle allowance")
        attribution = throttling_attribution(data.get("telemetry", []))
        if attribution != data.get("throttling_attribution"):
            pair_errors.append("recorded throttling attribution conflicts with telemetry")
        comparison = {
            str(minute): {
                "p99_ms": windows[minute - 1]["scheduled_to_completed_p99_ms"],
                "sample_count": windows[minute - 1]["sample_count"],
            }
            for minute in (first, later)
            if len(windows) >= minute
        }
        is_required = data.get("provider_id") in required_ids
        (errors if is_required else diagnostic_errors).extend(
            f"{pair}: {error}" for error in pair_errors
        )
        rows.append(
            {
                "precision": data.get("precision"),
                "provider_id": data.get("provider_id"),
                "required": is_required,
                "series": serialize(ref),
                "window_count": len(windows),
                "completed_count": completed,
                "missed_slot_count": missed,
                "deadline_miss_count": deadline_misses,
                "achieved_rate_parts_per_minute": achieved_rate,
                "comparison_minutes": comparison,
                "worst_minute": worst["minute"] if worst else None,
                "worst_minute_p99_ms": worst["scheduled_to_completed_p99_ms"] if worst else None,
                "worst_minute_sample_count": worst["sample_count"] if worst else None,
                "throttling_attribution": attribution,
                "errors": pair_errors,
            }
        )
    observed = {(row["precision"], row["provider_id"]) for row in rows}
    for provider_id in sorted(required_ids):
        for precision in ("fp32", "fp16", "int8"):
            if (precision, provider_id) not in observed:
                errors.append(f"missing required sustained pair: {precision}/{provider_id}")
    report = record(
        Path(config["_project"]) / config["outputs"]["thermal"] / "summary.json",
        {
            "status": "failed" if errors else "passed",
            "rows": rows,
            "errors": errors,
            "diagnostic_errors": diagnostic_errors,
            "minute_p99_note": (
                f"Each minute holds about {rate} samples, so its empirical p99 lies near the largest "
                "observation and is a noisy tail estimate rather than a stable guarantee."
            ),
            "per_minute_sample_requirement": "complete scheduled windows; the 500-sample microbenchmark rule does not apply",
            "latency_definition": "scheduled slot start to synchronous postprocessing completion",
        },
    )
    if errors:
        raise ThermalSummaryError(report, errors)
    return report


class ThermalSummaryError(ValueError):
    """Carry failed sustained evidence without concealing completed series."""

    def __init__(self, report: EvidenceRef, errors: list[str]):
        self.report = report
        self.errors = errors
        super().__init__("; ".join(errors))
