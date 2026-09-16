"""These tests specify required behavior and refuse silent scaffold success."""
import json
from pathlib import Path

import pytest

from cycletime.bench.report import ThermalSummaryError, summarize_thermal
from cycletime.bench.thermal import minute_windows, pace, parse_pmset_therm, throttling_attribution
from cycletime.evidence import record

pytestmark = [pytest.mark.contract]
MS = 1_000_000


class FakeClock:
    """Advance only when the scheduler sleeps or the workload reports service time."""

    def __init__(self) -> None:
        self.now = 1_000 * MS

    def __call__(self) -> int:
        return self.now

    def sleep_until(self, target: int) -> None:
        self.now = max(self.now, target)


def test_pace_without_backlog() -> None:
    """A deterministic scheduler fixture must preserve absolute deadlines, count overruns, and fail instead of queueing catch-up frames."""
    clock = FakeClock()
    durations = {0: 10 * MS, 1: 250 * MS, 2: 10 * MS, 3: 10 * MS}
    ran = []

    def run(index: int) -> None:
        ran.append(index)
        clock.now += durations.get(index, 10 * MS)

    slots = pace(run, 100 * MS, 5, clock, clock.sleep_until, suspended=lambda: 0, wall=clock)
    assert [slot["status"] for slot in slots] == ["completed", "completed", "missed", "missed", "completed"]
    assert ran == [0, 1, 4]
    assert [slot["scheduled_offset_ns"] for slot in slots] == [i * 100 * MS for i in range(5)]
    assert slots[1]["deadline_missed"] is True
    assert slots[4]["start_delay_ns"] == 0
    assert slots[4]["scheduled_to_completed_ns"] == slots[4]["service_ns"]


def fake_series(
    tmp_path: Path,
    config: dict,
    *,
    minutes: int,
    telemetry: list[dict],
    late_ms: float = 10.0,
    precision: str = "fp32",
    suspended_ms: int = 0,
    wall_jump_ms: int = 0,
) -> object:
    """Record a synthetic sustained series whose later minutes can run slower."""
    rate = config["sustained"]["rate_parts_per_minute"]
    slots = []
    for index in range(rate * minutes):
        service = int((late_ms if index >= rate * 20 else 10.0) * MS)
        slots.append(
            {
                "slot": index,
                "scheduled_offset_ns": index * (60_000 * MS // rate),
                "status": "completed",
                "start_delay_ns": 0,
                "service_ns": service,
                "scheduled_to_completed_ns": service,
                "deadline_missed": False,
                "host_suspended_total_ns": suspended_ms * MS if index >= rate * 10 else 0,
                "wall_offset_ns": index * (60_000 * MS // rate)
                + service
                + (wall_jump_ms * MS if index >= rate * 10 + 20 else 0),
            }
        )
    windows = minute_windows(slots, rate, "linear") if len(slots) % rate == 0 else []
    return record(
        tmp_path / f"series-{precision}-{minutes}.json",
        {
            "artifact_sha256": "a",
            "precision": precision,
            "provider_id": "cpu",
            "provider_name": "CPUExecutionProvider",
            "started_at_utc": "2026-01-01T00:00:00+00:00",
            "completed_at_utc": "2026-01-01T00:30:00+00:00",
            "rate_parts_per_minute": rate,
            "duration_minutes": config["sustained"]["duration_minutes"],
            "scheduled_slot_count": rate * config["sustained"]["duration_minutes"],
            "slots": slots,
            "windows": windows,
            "telemetry": telemetry,
            "throttling_attribution": throttling_attribution(telemetry),
        },
    )


def sustained_config(tmp_path: Path) -> dict:
    """Mirror the sustained configuration with one required CPU provider."""
    return {
        "_project": str(tmp_path),
        "line_rate_parts_per_minute": 45,
        "quantile_method": "linear",
        "providers": [{"id": "cpu", "required": True}],
        "outputs": {"thermal": "thermal"},
        "sustained": {
            "rate_parts_per_minute": 45,
            "duration_minutes": 30,
            "comparison_minutes": [1, 25],
            "hold_rate_tolerance_fraction": 0.01,
            "host_suspension_tolerance_ms": 10,
        },
    }


def test_retain_complete_windows(tmp_path: Path) -> None:
    """Thirty complete minutes must retain counts, minute-one and minute-twenty-five p99, and the worst minute; incomplete series must fail."""
    config = sustained_config(tmp_path)
    nominal = [{"available": True, "throttling_signal": False}]
    good = fake_series(tmp_path, config, minutes=30, telemetry=nominal, late_ms=12.0)
    others = [
        fake_series(tmp_path, config, minutes=30, telemetry=nominal, precision=p)
        for p in ("fp16", "int8")
    ]
    report = json.loads(summarize_thermal([good, *others], config).path.read_text())
    row = report["rows"][0]
    assert report["status"] == "passed"
    assert row["window_count"] == 30 and row["completed_count"] == 1350
    assert row["comparison_minutes"]["1"] == {"p99_ms": 10.0, "sample_count": 45}
    assert row["comparison_minutes"]["25"]["sample_count"] == 45
    assert row["worst_minute_p99_ms"] == pytest.approx(12.0) and row["worst_minute"] >= 21
    assert "500-sample" in report["per_minute_sample_requirement"]

    short = fake_series(tmp_path, config, minutes=29, telemetry=nominal)
    with pytest.raises(ThermalSummaryError) as failed:
        summarize_thermal([short, *others], config)
    assert any("incomplete" in error or "complete 45-slot windows" in error for error in failed.value.errors)


def test_avoid_throttling_inference(tmp_path: Path) -> None:
    """Unavailable telemetry must produce unknown attribution, even when late-run p99 exceeds early-run p99."""
    config = sustained_config(tmp_path)
    unavailable = [{"available": False}]
    series = [
        fake_series(tmp_path, config, minutes=30, telemetry=unavailable, late_ms=40.0, precision=p)
        for p in ("fp32", "fp16", "int8")
    ]
    row = json.loads(summarize_thermal(series, config).path.read_text())["rows"][0]
    assert row["comparison_minutes"]["25"]["p99_ms"] > row["comparison_minutes"]["1"]["p99_ms"]
    assert row["throttling_attribution"] == "unknown_telemetry_unavailable"
    assert throttling_attribution([]) == "unknown_telemetry_unavailable"
    nominal = parse_pmset_therm("Note: No thermal warning level has been recorded\n")
    assert nominal["available"] and not nominal["throttling_signal"]
    limited = parse_pmset_therm("CPU_Speed_Limit = 80\n")
    assert throttling_attribution([nominal, limited]) == "throttling_signal_observed"


def test_parse_separate_process_telemetry_log() -> None:
    """Sampler logs must keep one timestamped sample per block and mark unrecognized output unavailable."""
    from cycletime.bench.thermal import TELEMETRY_SEPARATOR, parse_telemetry_log

    text = (
        f"{TELEMETRY_SEPARATOR}\n2026-01-01T00:00:00+00:00\nNote: No thermal warning level has been recorded\n"
        f"{TELEMETRY_SEPARATOR}\n2026-01-01T00:00:01+00:00\npmset: command failed\n"
    )
    samples = parse_telemetry_log(text)
    assert [sample["utc"] for sample in samples] == ["2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00"]
    assert samples[0]["available"] and not samples[1]["available"]
    assert throttling_attribution(samples) == "unknown_telemetry_unavailable"


def test_refuse_series_with_host_sleep(tmp_path: Path) -> None:
    """Host sleep hides elapsed time from the monotonic clock, so any slept minute must fail the series."""
    clock = FakeClock()
    asleep = iter([0, 0, 0, 30_000 * MS, 30_000 * MS])

    def run(index: int) -> None:
        clock.now += 10 * MS

    slots = pace(run, 100 * MS, 4, clock, clock.sleep_until, suspended=lambda: next(asleep), wall=clock)
    assert [slot["host_suspended_total_ns"] for slot in slots] == [0, 0, 30_000 * MS, 30_000 * MS]
    assert all(slot["status"] == "completed" for slot in slots)

    config = sustained_config(tmp_path)
    nominal = [{"available": True, "throttling_signal": False}]
    series = [
        fake_series(tmp_path, config, minutes=30, telemetry=nominal, precision=p, suspended_ms=46_000 if p == "fp32" else 0)
        for p in ("fp32", "fp16", "int8")
    ]
    with pytest.raises(ThermalSummaryError) as failed:
        summarize_thermal(series, config)
    assert failed.value.errors == ["fp32/cpu: host slept during minutes [11]"]


def test_detect_sleep_from_wall_clock_when_suspension_counter_drifts(tmp_path: Path) -> None:
    """Wall time outruns the benchmark clock across a slept window even when the suspension counter drifts."""
    config = sustained_config(tmp_path)
    nominal = [{"available": True, "throttling_signal": False}]
    drifting = [
        fake_series(tmp_path, config, minutes=30, telemetry=nominal, precision=p, suspended_ms=-24)
        for p in ("fp32", "fp16", "int8")
    ]
    report = json.loads(summarize_thermal(drifting, config).path.read_text())
    assert report["status"] == "passed"

    slept = [
        fake_series(
            tmp_path / "slept",
            config,
            minutes=30,
            telemetry=nominal,
            precision=p,
            wall_jump_ms=46_000 if p == "int8" else 0,
        )
        for p in ("fp32", "fp16", "int8")
    ]
    with pytest.raises(ThermalSummaryError) as failed:
        summarize_thermal(slept, config)
    assert failed.value.errors == ["int8/cpu: host slept during minutes [11]"]
