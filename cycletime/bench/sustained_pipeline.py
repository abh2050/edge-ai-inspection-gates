"""Gate 4 runs paced thirty-minute series for every required artifact and provider pair."""

from __future__ import annotations

import fcntl
from pathlib import Path

from cycletime.bench.report import ThermalSummaryError, required_provider_ids, summarize_thermal
from cycletime.bench.thermal import sustained
from cycletime.config import load_config
from cycletime.contracts import EvidenceRef, ProviderSpec
from cycletime.evidence import read, record, reference, restore, serialize
from cycletime.export.metadata import load_artifact
from cycletime.model.pipeline import prepare


def run(config_dir: Path) -> EvidenceRef:
    """Validate Gate 3, run each required pair sequentially for thirty paced minutes, and refuse a pass after any incomplete or missed series."""
    _model, dataset, export = prepare(config_dir)
    project = Path(dataset["_project"])
    gate3_ref = reference(project / "artifacts/gate3.json")
    gate3 = read(gate3_ref)
    if gate3["status"] != "passed":
        raise ValueError("Gate 4 requires passed Gate 3 evidence.")
    read(restore(gate3["matrix"]))
    bench = load_config(config_dir.resolve() / "bench.yaml")
    bench["_project"] = str(project)
    settings = bench["sustained"]
    if not settings["run_pairs_sequentially"] or not settings["require_recorded_starting_conditions"]:
        raise ValueError("Gate 4 requires sequential pairs with recorded starting conditions.")
    required_ids = required_provider_ids(bench)
    if sorted(required_ids) != gate3["required_providers"]:
        raise ValueError("Required providers changed after Gate 3.")
    gate2 = read(restore(gate3["gate2"]))
    artifacts = {}
    for row in gate2["exports"]:
        artifact, _ = load_artifact(project / export[row["precision"]]["path"])
        if artifact.sha256 != row["artifact_sha256"]:
            raise ValueError(f"Gate 2 artifact changed: {row['precision']}")
        artifacts[row["precision"]] = artifact
    include_diagnostic = bool(settings.get("include_diagnostic_providers", False))
    providers = [
        ProviderSpec(item["id"], item["name"], item["options"], item["require_ane_evidence"])
        for item in bench["providers"]
        if item["required"] or include_diagnostic
    ]
    partitions = restore(read(reference(project / "artifacts/gate1.json"))["partitions"])
    series = []
    failures = []
    with (project / "artifacts/gate4.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another Gate 4 process is running.") from exc
        for provider in providers:
            for precision in ("fp32", "fp16", "int8"):
                print(
                    f"Gate 4 paces {precision}/{provider.id} for "
                    f"{settings['duration_minutes']} minutes.",
                    flush=True,
                )
                try:
                    ref = sustained(artifacts[precision], provider, partitions, bench)
                    series.append(ref)
                    windows = read(ref)["windows"]
                    print(
                        f"Completed {precision}/{provider.id}: worst minute p99 "
                        f"{max(w['scheduled_to_completed_p99_ms'] or 0 for w in windows):.3f} ms.",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - every pair retains its failure.
                    failures.append(
                        {
                            "precision": precision,
                            "provider_id": provider.id,
                            "required": provider.id in required_ids,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    print(f"Failed {precision}/{provider.id}: {exc}", flush=True)
        errors = [
            f"{item['precision']}/{item['provider_id']}: {item['error']}"
            for item in failures
            if item["required"]
        ]
        try:
            summary = summarize_thermal(series, bench)
        except ThermalSummaryError as exc:
            summary = exc.report
            errors.extend(exc.errors)
        table = write_sustained_table(read(summary), bench, project / "docs/sustained.md")
        report = record(
            project / "artifacts/gate4.json",
            {
                "gate": 4,
                "status": "failed" if errors else "passed",
                "gate3": serialize(gate3_ref),
                "required_providers": sorted(required_ids),
                "series": [serialize(ref) for ref in series],
                "summary": serialize(summary),
                "report_sha256": table,
                "failures": failures,
                "errors": errors,
                "release_feasibility": "not established; non-model overhead remains unmeasured",
            },
        )
        if errors:
            raise ValueError(f"Gate 4 failed. Evidence: {report.path}")
        return report


def write_sustained_table(summary: dict, config: dict, destination: Path) -> str:
    """Render the sustained summary with sample counts beside every minute p99; refuse unlabeled tail claims."""
    from cycletime.dataio.archive import sha256

    first, later = (str(value) for value in config["sustained"]["comparison_minutes"])
    lines = [
        "# Gate 4 records sustained paced latency.",
        "",
        f"Gate 4 {summary['status']} its sustained checks.",
        (
            f"Each series paced {config['sustained']['rate_parts_per_minute']} parts per minute for "
            f"{config['sustained']['duration_minutes']} minutes against absolute monotonic deadlines."
        ),
        "Latency runs from each scheduled slot start to synchronous postprocessing completion.",
        "A busy slot counts as missed, and the runner never queues catch-up frames.",
        f"The exact research cycle allowance is {60_000 / config['line_rate_parts_per_minute']:.6f} milliseconds.",
        "",
        f"| Provider | Precision | Role | Minute {first} p99 ms (n) | Minute {later} p99 ms (n) | Worst-minute p99 ms (minute, n) | Achieved parts/min | Missed slots | Deadline misses | Throttling attribution |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    def cell(entry: dict | None) -> str:
        if not entry or entry.get("p99_ms") is None:
            return "unavailable"
        return f"{entry['p99_ms']:.3f} ({entry['sample_count']})"

    for row in summary["rows"]:
        worst = (
            f"{row['worst_minute_p99_ms']:.3f} ({row['worst_minute']}, {row['worst_minute_sample_count']})"
            if row["worst_minute_p99_ms"] is not None
            else "unavailable"
        )
        lines.append(
            f"| {row['provider_id']} | {row['precision'].upper()} | {'required' if row['required'] else 'diagnostic'} "
            f"| {cell(row['comparison_minutes'].get(first))} | {cell(row['comparison_minutes'].get(later))} "
            f"| {worst} | {row['achieved_rate_parts_per_minute']:.3f} | {row['missed_slot_count']} "
            f"| {row['deadline_miss_count']} | {row['throttling_attribution']} |"
        )
    lines.extend(
        [
            "",
            summary["minute_p99_note"],
            "A later-minute increase alone does not establish thermal throttling.",
            "Throttling attribution comes only from pmset thermal telemetry, and unavailable telemetry stays unknown.",
            "The workstation did not record ambient temperature or a representative enclosure.",
            "Release feasibility remains unestablished because acquisition, transport, and actuation overhead remain unmeasured.",
        ]
    )
    for error in summary["errors"]:
        lines.append(f"Required failure: {error}")
    for error in summary.get("diagnostic_errors", []):
        lines.append(f"Diagnostic failure: {error}")
    destination.write_text("\n".join(lines) + "\n")
    return sha256(destination)
