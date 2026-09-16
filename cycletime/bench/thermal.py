"""cycletime/bench/thermal.py measures latency under a sustained target-rate workload."""
from __future__ import annotations

import math
import os
import re
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Self

import numpy as np

from cycletime.contracts import JSON, Artifact, EvidenceRef, ProviderSpec
from cycletime.dataio.archive import sha256
from cycletime.evidence import record, serialize

NS_PER_MS = 1_000_000
SPEED_LIMIT_RE = re.compile(r"CPU_Speed_Limit\s*=\s*(\d+)")
WARNING_RE = re.compile(r"(thermal|performance) warning level set to (\d+)", re.IGNORECASE)


def host_suspended_ns() -> int:
    """Return cumulative host sleep since boot; macOS CLOCK_MONOTONIC counts sleep and CLOCK_UPTIME_RAW does not.

    Three valid thirty-minute series drifted by up to 24 milliseconds within one minute and by
    35 milliseconds overall, in both directions, so this pair resolves host sleep in seconds only.
    The wall-clock comparison in each window carries the primary sleep evidence.
    """
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC) - time.clock_gettime_ns(time.CLOCK_UPTIME_RAW)


def pace(
    run: Callable[[int], None],
    period_ns: int,
    slot_count: int,
    clock: Callable[[], int] = perf_counter_ns,
    sleep_until: Callable[[int], None] | None = None,
    suspended: Callable[[], int] = host_suspended_ns,
    wall: Callable[[], int] = time.time_ns,
) -> list[JSON]:
    """Run one frame per absolute monotonic slot; record a busy slot as missed, never queue catch-up frames, and record host sleep that the monotonic clock cannot see."""
    if period_ns <= 0 or slot_count <= 0:
        raise ValueError("Pacing requires a positive period and slot count.")
    if sleep_until is None:

        def sleep_until(target: int) -> None:
            # macOS timer coalescing can overshoot sleep by about 10 ms, so spin the final 25 ms.
            coarse = target - clock() - 25 * NS_PER_MS
            if coarse > 0:
                time.sleep(coarse / 1_000_000_000)
            while clock() < target:
                pass

    origin = clock()
    wall_origin = wall()
    suspended_origin = suspended()
    busy_until = origin
    slots = []
    for index in range(slot_count):
        scheduled = origin + index * period_ns
        if busy_until > scheduled:
            slots.append(
                {
                    "slot": index,
                    "scheduled_offset_ns": scheduled - origin,
                    "status": "missed",
                    "host_suspended_total_ns": suspended() - suspended_origin,
                    "wall_offset_ns": wall() - wall_origin,
                }
            )
            continue
        sleep_until(scheduled)
        started = clock()
        run(index)
        completed = clock()
        if completed <= started or started < scheduled:
            raise ValueError("The monotonic clock produced an invalid paced interval.")
        busy_until = completed
        slots.append(
            {
                "slot": index,
                "scheduled_offset_ns": scheduled - origin,
                "status": "completed",
                "start_delay_ns": started - scheduled,
                "service_ns": completed - started,
                "scheduled_to_completed_ns": completed - scheduled,
                "deadline_missed": completed > scheduled + period_ns,
                "host_suspended_total_ns": suspended() - suspended_origin,
                "wall_offset_ns": wall() - wall_origin,
            }
        )
    return slots


def last_offset(slot: JSON) -> int:
    """Return a slot's monotonic offset at completion, or its scheduled offset when it was missed."""
    return slot["scheduled_offset_ns"] + slot.get("scheduled_to_completed_ns", 0)


def minute_windows(slots: list[JSON], slots_per_window: int, quantile_method: str) -> list[JSON]:
    """Summarize each reporting window from raw slots; refuse partial windows instead of padding them."""
    if slots_per_window <= 0 or len(slots) % slots_per_window:
        raise ValueError("The sustained series contains an incomplete reporting window.")
    windows = []
    previous_suspended = 0
    for number, first in enumerate(range(0, len(slots), slots_per_window), start=1):
        group = slots[first : first + slots_per_window]
        done = [slot for slot in group if slot["status"] == "completed"]
        suspended_total = max(slot.get("host_suspended_total_ns", 0) for slot in group)
        window = {
            "minute": number,
            "scheduled_count": len(group),
            "sample_count": len(done),
            "missed_slot_count": len(group) - len(done),
            "deadline_miss_count": sum(bool(slot["deadline_missed"]) for slot in done),
            "host_suspended_ms": (suspended_total - previous_suspended) / NS_PER_MS,
            # Host sleep stops the benchmark clock, so wall time outruns it across a slept window.
            "wall_minus_monotonic_ms": (
                (group[-1]["wall_offset_ns"] - group[0]["wall_offset_ns"])
                - (last_offset(group[-1]) - last_offset(group[0]))
            )
            / NS_PER_MS,
        }
        previous_suspended = suspended_total
        for key in ("service_ns", "scheduled_to_completed_ns"):
            values = np.asarray([slot[key] for slot in done], dtype=np.float64) / NS_PER_MS
            name = key.removesuffix("_ns")
            window[f"{name}_p50_ms"] = (
                float(np.quantile(values, 0.5, method=quantile_method)) if len(values) else None
            )
            window[f"{name}_p99_ms"] = (
                float(np.quantile(values, 0.99, method=quantile_method)) if len(values) else None
            )
            window[f"{name}_max_ms"] = float(values.max()) if len(values) else None
        windows.append(window)
    return windows


def parse_pmset_therm(output: str) -> JSON:
    """Parse pmset thermal output; report unavailable fields instead of assuming nominal behavior."""
    limits = [int(value) for value in SPEED_LIMIT_RE.findall(output)]
    warnings = [int(level) for _, level in WARNING_RE.findall(output)]
    no_thermal = "No thermal warning level has been recorded" in output
    no_performance = "No performance warning level has been recorded" in output
    recognized = bool(limits or warnings or no_thermal or no_performance)
    return {
        "available": recognized,
        "cpu_speed_limit_percent": min(limits) if limits else None,
        "warning_levels": warnings,
        "throttling_signal": bool((limits and min(limits) < 100) or any(warnings)),
    }


TELEMETRY_SEPARATOR = "=== cycletime telemetry sample ==="


def parse_telemetry_log(text: str) -> list[JSON]:
    """Split a sampler log into timestamped pmset samples; keep unrecognized blocks as unavailable."""
    samples = []
    for block in text.split(TELEMETRY_SEPARATOR)[1:]:
        stamp, _, body = block.strip().partition("\n")
        samples.append(dict(parse_pmset_therm(body), utc=stamp.strip()))
    return samples


class TelemetrySampler:
    """Poll pmset thermal state in a separate OS process so sampling never holds the benchmark's GIL."""

    def __init__(self, interval_seconds: float, log_path: Path):
        if interval_seconds <= 0:
            raise ValueError("Telemetry requires a positive interval.")
        self.interval = interval_seconds
        self.log_path = log_path
        self.samples: list[JSON] = []
        self._process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> Self:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        script = (
            f"while :; do echo '{TELEMETRY_SEPARATOR}'; date -u +%Y-%m-%dT%H:%M:%S+00:00; "
            f"/usr/bin/pmset -g therm 2>&1; sleep {self.interval}; done"
        )
        with self.log_path.open("wb") as log:
            self._process = subprocess.Popen(["/bin/sh", "-c", script], stdout=log, stderr=log)
        return self

    def __exit__(self, *_: object) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=10)
        self.samples = parse_telemetry_log(self.log_path.read_text(errors="replace"))


def throttling_attribution(telemetry: list[JSON]) -> str:
    """Attribute throttling only from observed telemetry; unavailable telemetry stays unknown regardless of latency."""
    available = [sample for sample in telemetry if sample.get("available")]
    if not available or len(available) != len(telemetry):
        return "unknown_telemetry_unavailable"
    if any(sample.get("throttling_signal") for sample in available):
        return "throttling_signal_observed"
    return "no_throttling_signal_observed"


def starting_conditions(project: Path) -> JSON:
    """Record observable host conditions before a sustained run; mark unmeasured physical conditions explicitly."""

    def command(args: list[str]) -> str | None:
        try:
            return subprocess.run(args, check=True, capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return None

    agent = __import__("yaml").safe_load((project / "config/agent.yaml").read_text())
    return {
        "utc": datetime.now(UTC).isoformat(),
        "load_average_1_5_15": list(os.getloadavg()),
        "power_source": command(["/usr/bin/pmset", "-g", "batt"]),
        "thermal_state": command(["/usr/bin/pmset", "-g", "therm"]),
        "agent_enabled": bool(agent["enabled"]),
        "ambient_temperature_c": None,
        "enclosure": "not measured; developer workstation",
    }


def sustained(
    artifact: Artifact, provider: ProviderSpec, inputs: EvidenceRef, config: JSON
) -> EvidenceRef:
    """Warm up separately, then pace 30 minutes against absolute monotonic deadlines; record service time, scheduled-to-completed time, missed slots, UTC start, per-minute p99, counts, and available thermal telemetry. Refuse queued catch-up, hidden rate reductions, incomplete passes, and unobserved throttling claims."""
    from cycletime.bench.providers import create_session
    from cycletime.dataio.calibration import validate_partitions
    from cycletime.dataio.images import decode_image, preprocess_decoded

    settings = config["sustained"]
    if (
        settings["pacing"] != "monotonic_absolute_deadlines"
        or settings["backlog_policy"] != "record_miss_and_fail_without_queue"
        or settings["reporting_window_seconds"] != 60
        or settings["rate_parts_per_minute"] != config["line_rate_parts_per_minute"]
        or config["clock"] != "perf_counter_ns"
    ):
        raise ValueError("The sustained run requires the configured rate, pacing, and backlog policy.")
    rate = int(settings["rate_parts_per_minute"])
    minutes = int(settings["duration_minutes"])
    if minutes < 30:
        raise ValueError("The sustained run requires at least thirty minutes.")
    parts = validate_partitions(inputs)
    dataset = parts["dataset_config"]
    project = Path(dataset["_project"])
    conditions = starting_conditions(project)
    if conditions["agent_enabled"]:
        raise ValueError("The local agent must remain disabled during sustained measurements.")
    names = parts["validation"]
    root = project / dataset["root"]
    decoded = [decode_image(root / name) for name in names]
    session, placement = create_session(artifact, provider, config, scope="sustained")
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

    for index in range(int(config["warmup_iterations"])):
        run(index)
    period_ns = 60_000_000_000 // rate
    slot_count = rate * minutes
    started = datetime.now(UTC).isoformat()
    destination = project / config["outputs"]["thermal"] / provider.id / f"{artifact.path.stem}.json"
    telemetry_log = destination.with_suffix(".telemetry.log")
    # caffeinate holds idle, system, and disk sleep assertions only while this process runs the series.
    awake = subprocess.Popen(["/usr/bin/caffeinate", "-ims", "-w", str(os.getpid())])
    try:
        with TelemetrySampler(
            float(settings["telemetry_interval_seconds"]), telemetry_log
        ) as telemetry:
            slots = pace(run, period_ns, slot_count)
    finally:
        awake.terminate()
        awake.wait(timeout=10)
    completed = datetime.now(UTC).isoformat()
    windows = minute_windows(slots, rate, config["quantile_method"])
    return record(
        destination,
        {
            "artifact_sha256": artifact.sha256,
            "precision": artifact.precision,
            "provider_id": provider.id,
            "provider_name": provider.name,
            "started_at_utc": started,
            "completed_at_utc": completed,
            "clock": config["clock"],
            "rate_parts_per_minute": rate,
            "duration_minutes": minutes,
            "period_ns": period_ns,
            "scheduled_slot_count": slot_count,
            "warmup_iterations": config["warmup_iterations"],
            "warmup_excluded": True,
            "pacing": settings["pacing"],
            "backlog_policy": settings["backlog_policy"],
            "measurement_scope": config["measurement_scope"],
            "input_ids": names,
            "starting_conditions": conditions,
            "slots": slots,
            "windows": windows,
            "sleep_prevention": "/usr/bin/caffeinate -ims -w <runner pid>",
            "host_suspension_clock": "CLOCK_MONOTONIC minus CLOCK_UPTIME_RAW",
            "telemetry_source": "/usr/bin/pmset -g therm in a separate process",
            "telemetry_log": str(telemetry_log),
            "telemetry_log_sha256": sha256(telemetry_log),
            "telemetry": telemetry.samples,
            "throttling_attribution": throttling_attribution(telemetry.samples),
            "postprocess_observation": observed,
            "placement": serialize(placement),
        },
    )
