"""Extract recorded gate evidence into one dashboard data file without recomputing any measurement."""

from __future__ import annotations

import base64
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT / "dashboard/data.json"


def load(path: Path) -> dict:
    """Read one recorded evidence file."""
    return json.loads(path.read_text())


def build() -> dict:
    """Copy recorded values; every figure keeps the gate that produced it."""
    gates = {}
    for number in range(6):
        path = PROJECT / f"artifacts/gate{number}.json"
        record = load(path)
        gates[f"gate{number}"] = {
            "status": record["status"],
            "created_at_utc": record["created_at_utc"],
            "source": f"artifacts/gate{number}.json",
        }

    release = load(PROJECT / "artifacts/release-check.json")
    environment = load(PROJECT / "artifacts/environment.json")
    matrix = load(PROJECT / "artifacts/bench/matrix.json")
    thermal = load(PROJECT / "artifacts/thermal/summary.json")
    gate2 = load(PROJECT / "artifacts/gate2.json")
    gate5 = load(PROJECT / "artifacts/gate5.json")
    costs = yaml.safe_load((PROJECT / "config/costs.yaml").read_text())
    bench = yaml.safe_load((PROJECT / "config/bench.yaml").read_text())
    model = yaml.safe_load((PROJECT / "config/model.yaml").read_text())

    micro = []
    for row in matrix["measurements"]:
        summary = load(Path(row["summary"]["path"]))
        placement = load(Path(row["placement"]["path"]))
        micro.append(
            {
                "precision": row["precision"],
                "provider": row["provider_id"],
                "required": row["required"],
                "p50_ms": summary["p50_ms"],
                "p95_ms": summary["p95_ms"],
                "p99_ms": summary["p99_ms"],
                "capacity_parts_per_minute": summary["capacity_from_p99_parts_per_minute"],
                "placement_label": placement["placement_label"],
                "placement_passed": placement["placement_passed"],
                "source": "artifacts/bench/matrix.json (gate 3)",
            }
        )

    sustained = []
    for row in thermal["rows"]:
        series = load(Path(row["series"]["path"]))
        sustained.append(
            {
                "precision": row["precision"],
                "provider": row["provider_id"],
                "completed_count": row["completed_count"],
                "missed_slot_count": row["missed_slot_count"],
                "deadline_miss_count": row["deadline_miss_count"],
                "achieved_rate": row["achieved_rate_parts_per_minute"],
                "worst_minute": row["worst_minute"],
                "worst_minute_p99_ms": row["worst_minute_p99_ms"],
                "throttling_attribution": row["throttling_attribution"],
                "started_at_utc": series["started_at_utc"],
                "minutes": [
                    {
                        "minute": window["minute"],
                        "p50_ms": window["scheduled_to_completed_p50_ms"],
                        "p99_ms": window["scheduled_to_completed_p99_ms"],
                        "samples": window["sample_count"],
                    }
                    for window in series["windows"]
                ],
                "source": "artifacts/thermal/summary.json (gate 4)",
            }
        )

    accuracy = [
        {
            "precision": row["precision"],
            "image_auroc": row["metrics_values"]["image_auroc"],
            "pixel_auroc": row["metrics_values"]["pixel_auroc"],
            "recall": row["metrics_values"]["recall"],
            "f1": row["metrics_values"]["f1"],
            "frozen_threshold": row["metrics_values"]["threshold"],
            "source": "artifacts/gate2.json (gates 1 and 2)",
        }
        for row in gate2["exports"]
    ]

    curves = []
    for reference in gate5["curves"]:
        curve = load(Path(reference["path"]))
        curves.append(
            {
                "artifact_sha256": curve["artifact_sha256"],
                "threshold_source": curve["threshold_source"],
                "configured_prevalence": curve["configured_prevalence"],
                "observed_test_defect_ratio": curve["observed_test_defect_ratio"],
                "prevalence_range": curve["prevalence_range"],
                "points": [
                    {
                        "threshold": row["threshold"],
                        "cost_usd_per_shift": row["expected_cost_usd_per_shift"],
                        "cost_by_prevalence": row["prevalence_sensitivity"],
                        "recall": row["recall"],
                        "false_reject_rate": row["false_positive_rate"],
                        "f1": row["f1"],
                        "tp": row["tp"],
                        "fp": row["fp"],
                        "fn": row["fn"],
                        "tn": row["tn"],
                    }
                    for row in curve["rows"]
                ],
                "source": "artifacts/evaluation/<artifact>/cost-curve.json (gate 5)",
            }
        )

    precision_of = {row["artifact_sha256"]: row["precision"] for row in gate2["exports"]}
    for curve in curves:
        curve["precision"] = precision_of.get(curve["artifact_sha256"], "unknown")

    # ADR 0009: the detection example is documentation rendered from gate 1 evidence.
    gate1 = load(PROJECT / "artifacts/gate1.json")
    predictions = load(Path(gate1["predictions"]["path"]))["images"]
    metrics = load(Path(gate1["metrics"]["path"]))
    figure_source = PROJECT / "docs/screenshots/00-detection-example.png"
    detection_example = None
    if figure_source.is_file():
        shutil.copyfile(figure_source, DESTINATION.parent / "detection-example.png")
        # The figure also travels inside the data file so no cache or path can lose it.
        encoded = base64.b64encode(figure_source.read_bytes()).decode("ascii")
        def row(image_id: str) -> dict:
            found = next(item for item in predictions if item["image_id"] == image_id)
            return {
                "image_id": image_id,
                "score": found["score"],
                "label": found["label"],
                "defect_pixels": found["defect_pixels"],
                "decision": "reject" if found["score"] >= metrics["threshold"] else "accept",
            }
        detection_example = {
            "image": "detection-example.png",
            "image_data_uri": f"data:image/png;base64,{encoded}",
            "threshold": metrics["threshold"],
            "threshold_source": metrics["threshold_source"],
            "normal": row("bottle/test/good/000.png"),
            "defective": row("bottle/test/broken_large/000.png"),
            "attribution": "MVTec Anomaly Detection dataset, CC BY-NC-SA 4.0",
            "source": "artifacts/gate1.json (gate 1)",
        }

    evidence_index = {
        "gate records": [f"artifacts/gate{n}.json" for n in range(6)],
        "latency matrix": "artifacts/bench/matrix.json",
        "sustained summary": "artifacts/thermal/summary.json",
        "release check": "artifacts/release-check.json",
        "environment": "artifacts/environment.json",
        "reports": ["docs/latency.md", "docs/sustained.md", "docs/scorecard.md"],
    }
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "evidence_index": evidence_index,
        "generator": "scripts/build_dashboard.py",
        "detection_example": detection_example,
        "scope": "renders recorded evidence; recomputes nothing",
        "host": {
            "model": environment["model"],
            "chip": environment["chip"],
            "macos_version": environment["macos_version"],
            "onnxruntime_version": environment["onnxruntime_version"],
            "python_version": environment["python_version"],
            "threads": environment["thread_settings"],
            "source": "artifacts/environment.json",
        },
        "gates": gates,
        "release_check": {
            "status": release["status"],
            "unmet_conditions": release["unmet_conditions"],
            "source": "artifacts/release-check.json",
        },
        "line": {
            "rate_parts_per_minute": bench["line_rate_parts_per_minute"],
            "allowance_ms": 60_000 / bench["line_rate_parts_per_minute"],
            "sustained_minutes": bench["sustained"]["duration_minutes"],
            "warmups": bench["warmup_iterations"],
            "timed_iterations": bench["timed_iterations"],
        },
        "micro": micro,
        "sustained": sustained,
        "accuracy": accuracy,
        "quality_floors": model["quality_floors"],
        "curves": curves,
        "selected_point": gate5["selected_point"],
        "comparator_point": gate5["comparator_point"],
        "threshold_status": gate5["threshold_status"],
        "release_acceptance": gate5["release_acceptance"],
        "costs": {
            key: costs[key]
            for key in (
                "currency",
                "assumption_status",
                "scrap_cost_usd",
                "warranty_return_cost_usd",
                "rework_cost_usd",
                "parts_per_shift",
                "shifts_per_month",
                "production_defect_prevalence",
                "minimum_defect_recall",
            )
        },
    }


def main() -> None:
    """Write the dashboard data file beside the page that renders it."""
    data = build()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"{DESTINATION.relative_to(PROJECT)}: {DESTINATION.stat().st_size} bytes")
    print(f"gates: {', '.join(f'{k}={v['status']}' for k, v in data['gates'].items())}")
    print(f"release check: {data['release_check']['status']}")


if __name__ == "__main__":
    main()
