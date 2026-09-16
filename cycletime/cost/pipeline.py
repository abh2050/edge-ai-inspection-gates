"""Gate 5 verifies prior evidence and emits the retrospective cost decision."""

from __future__ import annotations

import fcntl
from pathlib import Path

from cycletime.bench.report import required_provider_ids
from cycletime.config import load_config
from cycletime.contracts import Artifact, EvidenceRef, Measurement, ProviderSpec
from cycletime.cost.curve import expected_cost_curve
from cycletime.cost.operating_point import (
    NoFeasiblePointError,
    accuracy_comparator,
    candidate_rows,
    select_operating_point,
)
from cycletime.dataio.archive import sha256
from cycletime.evidence import digest, read, record, reference, restore, serialize
from cycletime.export.metadata import load_artifact
from cycletime.model.pipeline import prepare
from cycletime.reporting.figures import draw_figures, write_scorecard


def run(config_dir: Path) -> EvidenceRef:
    """Validate prior gate digests, compute configured costs from cached predictions, select a feasible point, draw curves, and write docs/scorecard.md; refuse stale evidence, no feasible point, and presenting retrospective research as release acceptance."""
    model, dataset, export = prepare(config_dir)
    project = Path(dataset["_project"])
    costs = load_config(config_dir.resolve() / "costs.yaml")
    bench = load_config(config_dir.resolve() / "bench.yaml")
    bench["_project"] = str(project)
    gate4_ref = reference(project / "artifacts/gate4.json")
    gate4 = read(gate4_ref)
    if gate4["status"] != "passed":
        raise ValueError("Gate 5 requires passed Gate 4 evidence.")
    gate3_ref = restore(gate4["gate3"])
    gate3 = read(gate3_ref)
    gate2 = read(restore(gate3["gate2"]))
    gate1 = read(restore(gate2["gate1"]))
    if gate3["status"] != "passed" or gate2["status"] != "passed" or gate1["status"] != "passed":
        raise ValueError("Gate 5 requires passed Gate 1, Gate 2, and Gate 3 evidence.")
    thermal = restore(gate4["summary"])
    matrix = read(restore(gate3["matrix"]))
    required_ids = required_provider_ids(bench)
    if sorted(required_ids) != gate4["required_providers"]:
        raise ValueError("Required providers changed after Gate 4.")

    caches: dict[str, EvidenceRef] = {}
    artifacts: dict[str, Artifact] = {}
    for row in gate2["exports"]:
        precision = row["precision"]
        artifact, _ = load_artifact(project / export[precision]["path"])
        if artifact.sha256 != row["artifact_sha256"]:
            raise ValueError(f"Gate 2 artifact changed: {precision}")
        artifacts[precision] = artifact
        predictions = row.get("predictions") or gate1["predictions"]
        caches[precision] = restore(predictions)

    measurements = []
    for row in matrix["measurements"]:
        if not row["required"]:
            continue
        precision = row["precision"]
        provider_config = next(item for item in bench["providers"] if item["id"] == row["provider_id"])
        measurements.append(
            Measurement(
                artifacts[precision],
                ProviderSpec(
                    provider_config["id"],
                    provider_config["name"],
                    provider_config["options"],
                    provider_config["require_ane_evidence"],
                ),
                restore(row["samples"]),
                restore(read(restore(row["samples"]))["environment"]),
                restore(row["placement"]),
                restore(row["summary"]),
            )
        )

    with (project / "artifacts/gate5.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another Gate 5 process is running.") from exc
        curves = [expected_cost_curve(caches[precision], costs) for precision in ("fp32", "fp16", "int8")]
        candidates = candidate_rows(curves, measurements, thermal, bench)
        comparator_config = dict(costs, **model["threshold"])
        comparator = accuracy_comparator(curves, comparator_config)
        try:
            point = select_operating_point(curves, measurements, thermal, costs, bench)
        except NoFeasiblePointError as exc:
            record(
                project / "artifacts/gate5.json",
                {
                    "gate": 5,
                    "status": "failed",
                    "gate4": serialize(gate4_ref),
                    "error": str(exc),
                    "evaluated_candidate_count": len(exc.candidates),
                    "recall_floor": costs["minimum_defect_recall"],
                    "best_recall_observed": max(row["recall"] for row in exc.candidates),
                },
            )
            raise ValueError(f"Gate 5 failed: {exc}") from exc

        selected = min(
            (
                row
                for row in candidates
                if row["artifact_sha256"] == point.artifact_sha256
                and row["provider_id"] == point.provider_id
                and row["threshold"] == point.threshold
            ),
            key=lambda row: row["expected_cost_usd_per_shift"],
        )
        compared_rows = [
            row for row in candidates if row["threshold"] == comparator.threshold
            and row["artifact_sha256"] == comparator.artifact_sha256
        ]
        comparator_feasible = any(
            row["recall"] >= float(costs["minimum_defect_recall"])
            and row["worst_sustained_p99_ms"] <= row["cycle_allowance_ms"]
            for row in compared_rows
        )
        placement = read(next(
            restore(row["placement"])
            for row in matrix["measurements"]
            if row["provider_id"] == point.provider_id and row["artifact_sha256"] == point.artifact_sha256
        ))
        frozen = read(restore(gate1["metrics"]))
        evidence_hashes = {
            "gate1.json": sha256(project / "artifacts/gate1.json"),
            "gate2.json": sha256(project / "artifacts/gate2.json"),
            "gate3.json": sha256(project / "artifacts/gate3.json"),
            "gate4.json": sha256(project / "artifacts/gate4.json"),
            "thermal summary": thermal.sha256,
            "config/costs.yaml": sha256(config_dir.resolve() / "costs.yaml"),
            "config/bench.yaml": sha256(config_dir.resolve() / "bench.yaml"),
            "artifacts/environment.json": sha256(project / "artifacts/environment.json"),
            "selected artifact": point.artifact_sha256,
        }
        figures = draw_figures(
            measurements,
            curves,
            point,
            {"bench": bench, "costs": costs, "comparator": comparator, "thermal": thermal},
            project / "artifacts/figures",
        )
        inputs = record(
            project / "artifacts/figures/scorecard-inputs.json",
            {
                "costs": costs,
                "selected": dict(selected, placement_label=placement["placement_label"]),
                "comparator": {
                    "precision": next(
                        row["precision"]
                        for row in candidates
                        if row["artifact_sha256"] == comparator.artifact_sha256
                    ),
                    "f1": max(row["f1"] for row in compared_rows),
                    "expected_cost_usd_per_shift": comparator.expected_cost_usd_per_shift,
                    "feasibility": "feasible" if comparator_feasible else "infeasible under the configured recall floor",
                },
                "evaluation_status": "one complete cached evaluation for each artifact",
                "commercial_status": "unqualified; no independent customer validation has run",
                "evaluated_candidate_count": len(candidates),
                "configuration_digest": digest({"costs": costs, "bench": bench, "export": export}),
                "cost_figure": "artifacts/figures/cost-curve.png",
                "latency_figure": "artifacts/figures/latency-budget.png",
                "frozen_recall": frozen["recall"],
                "evidence_hashes": evidence_hashes,
                "unresolved_conditions": "customer data rights, independent validation, signed release, and shift soak acceptance",
            },
        )
        scorecard = write_scorecard(point, comparator, [inputs, *figures], project / "docs/scorecard.md")
        report = record(
            project / "artifacts/gate5.json",
            {
                "gate": 5,
                "status": "passed",
                "gate4": serialize(gate4_ref),
                "curves": [serialize(curve) for curve in curves],
                "selected_point": {
                    "artifact_sha256": point.artifact_sha256,
                    "provider_id": point.provider_id,
                    "precision": selected["precision"],
                    "threshold": point.threshold,
                    "recall": point.recall,
                    "false_reject_rate": point.false_reject_rate,
                    "expected_cost_usd_per_shift": point.expected_cost_usd_per_shift,
                    "worst_sustained_p99_ms": selected["worst_sustained_p99_ms"],
                    "tie_count": selected.get("tie_count"),
                },
                "comparator_point": {
                    "artifact_sha256": comparator.artifact_sha256,
                    "threshold": comparator.threshold,
                    "recall": comparator.recall,
                    "false_reject_rate": comparator.false_reject_rate,
                    "expected_cost_usd_per_shift": comparator.expected_cost_usd_per_shift,
                    "metric": "f1",
                    "feasible": comparator_feasible,
                },
                "threshold_status": "retrospective_research_selected_from_cached_test_scores",
                "release_acceptance": "not established; retrospective research selection is not release acceptance",
                "figures": [serialize(figure) for figure in figures],
                "scorecard": serialize(scorecard),
                "evidence_hashes": evidence_hashes,
            },
        )
        return report
