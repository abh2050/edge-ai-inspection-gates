"""Gate 3 measures every required artifact and provider pair on the local host."""

from __future__ import annotations

import fcntl
from pathlib import Path

from cycletime.bench.report import (
    MeasurementMatrixError,
    required_provider_ids,
    verify_measurement_matrix,
)
from cycletime.bench.runner import benchmark
from cycletime.config import load_config
from cycletime.contracts import Artifact, EvidenceRef, PredictionCache, ProviderSpec
from cycletime.evidence import read, record, reference, restore, serialize
from cycletime.export.metadata import load_artifact
from cycletime.export.verify_parity import ParityError, verify_provider
from cycletime.model.pipeline import prepare
from cycletime.reporting.latency_table import write_latency_tables


def run(config_dir: Path) -> EvidenceRef:
    """Measure the full matrix and report every placement failure; refuse skipped pairs or invented ANE use."""
    _model, dataset, export = prepare(config_dir)
    project = Path(dataset["_project"])
    gate2_ref = reference(project / "artifacts/gate2.json")
    gate2 = read(gate2_ref)
    if gate2["status"] != "passed":
        raise ValueError("Gate 3 requires passed Gate 2 evidence.")
    bench = load_config(config_dir.resolve() / "bench.yaml")
    bench["_project"] = str(project)
    if (
        bench["warmup_iterations"] != 20
        or bench["timed_iterations"] < 500
        or bench["batch_size"] != 1
    ):
        raise ValueError(
            "Gate 3 requires twenty warmups and at least five hundred batch-one samples."
        )
    artifacts: dict[str, Artifact] = {}
    caches = []
    for row in gate2["exports"]:
        precision = row["precision"]
        path = project / export[precision]["path"]
        artifact, _ = load_artifact(path)
        if artifact.sha256 != row["artifact_sha256"]:
            raise ValueError(f"Gate 2 artifact changed: {precision}")
        artifacts[precision] = artifact
        metrics = restore(row["metrics"])
        caches.append(
            PredictionCache(
                artifact,
                restore(row.get("predictions", gate2["exports"][0].get("predictions", {})))
                if row.get("predictions")
                else restore(read(reference(project / "artifacts/gate1.json"))["predictions"]),
                metrics,
                read(metrics)["evaluator_sha256"],
            )
        )
    partitions = restore(read(reference(project / "artifacts/gate1.json"))["partitions"])
    providers = [
        ProviderSpec(
            id=item["id"],
            name=item["name"],
            options=item["options"],
            require_ane_evidence=item["require_ane_evidence"],
        )
        for item in bench["providers"]
    ]
    required_ids = required_provider_ids(bench)
    measurements = []
    failures = []
    diagnostic_failures = []
    provider_parity = []
    parity_errors = []
    diagnostic_parity_errors = []
    with (project / "artifacts/gate3.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another Gate 3 process is running.") from exc
        for provider in providers:
            required_pair = provider.id in required_ids
            pair_parity_errors = parity_errors if required_pair else diagnostic_parity_errors
            pair_failures = failures if required_pair else diagnostic_failures
            for precision in ("fp32", "fp16", "int8"):
                artifact = artifacts[precision]
                role = "required" if required_pair else "diagnostic"
                print(f"Gate 3 measures {role} pair {precision}/{provider.id}.", flush=True)
                try:
                    parity = verify_provider(
                        artifact,
                        provider,
                        partitions,
                        dict(export["parity"], _bench=bench),
                    )
                    provider_parity.append(serialize(parity))
                except ParityError as exc:
                    provider_parity.append(serialize(exc.report))
                    pair_parity_errors.append(f"{precision}/{provider.id}: {exc}")
                except Exception as exc:  # noqa: BLE001 - benchmark still attempts the required pair.
                    pair_parity_errors.append(
                        f"{precision}/{provider.id}: {type(exc).__name__}: {exc}"
                    )
                try:
                    measurement = benchmark(artifact, provider, partitions, bench)
                    measurements.append(measurement)
                    result = read(measurement.summary)
                    print(
                        f"Measured {precision}/{provider.id}: p99 {result['p99_ms']:.3f} ms.",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - every required pair retains its failure.
                    failure = record(
                        project
                        / f"artifacts/bench/failures/{artifact.path.stem}-{provider.id}.json",
                        {
                            "precision": precision,
                            "provider_id": provider.id,
                            "artifact_sha256": artifact.sha256,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    pair_failures.append(serialize(failure))
                    print(f"Failed {precision}/{provider.id}: {exc}", flush=True)
        matrix = None
        matrix_errors = []
        try:
            matrix = verify_measurement_matrix(measurements, bench)
        except MeasurementMatrixError as exc:
            matrix = exc.report
            matrix_errors = exc.errors
        latency = write_latency_tables(
            measurements, caches, bench, project / bench["outputs"]["table"]
        )
        passed = not failures and not matrix_errors and not parity_errors
        report = record(
            project / "artifacts/gate3.json",
            {
                "gate": 3,
                "status": "passed" if passed else "failed",
                "gate2": serialize(gate2_ref),
                "matrix": serialize(matrix),
                "latency_report": serialize(latency),
                "measurement_count": len(measurements),
                "required_measurement_count": 6,
                "required_providers": sorted(required_ids),
                "decisive_measurement_count": 3 * len(required_ids),
                "measurement_failures": failures,
                "diagnostic_measurement_failures": diagnostic_failures,
                "provider_parity": provider_parity,
                "provider_parity_errors": parity_errors,
                "diagnostic_parity_errors": diagnostic_parity_errors,
                "policy_errors": matrix_errors,
                "diagnostic_policy_errors": read(matrix).get("diagnostic_errors", []),
                "decision": "docs/decisions/0007-coreml-diagnostic-provider.md",
            },
        )
        if not passed:
            raise ValueError(f"Gate 3 failed. Evidence: {report.path}")
        return report
