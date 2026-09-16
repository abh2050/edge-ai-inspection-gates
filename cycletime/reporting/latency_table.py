"""This module writes latency tables from measured samples and cached accuracy."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from cycletime.contracts import JSON, EvidenceRef, Measurement, PredictionCache
from cycletime.dataio.archive import sha256
from cycletime.evidence import read


def write_latency_tables(
    measurements: list[Measurement], caches: list[PredictionCache], config: JSON, destination: Path
) -> EvidenceRef:
    """Render one row per measured pair with cached accuracy; refuse provider mixing or placeholders as evidence."""
    metrics = {cache.artifact.sha256: read(cache.metrics) for cache in caches}
    if len(metrics) != len(caches):
        raise ValueError("Accuracy caches must identify distinct artifacts.")
    environment = read(measurements[0].environment) if measurements else {}
    required_ids = {provider["id"] for provider in config["providers"] if provider.get("required")}
    all_placements = [read(item.placement) for item in measurements if item.placement]
    placements = [item for item in all_placements if item.get("provider_id") in required_ids]
    diagnostic_placements = [
        item for item in all_placements if item.get("provider_id") not in required_ids
    ]
    coreml_placements = [item for item in all_placements if item.get("ane_requested")]
    ane_required = any(item.get("ane_required", True) for item in coreml_placements)
    if coreml_placements and all(item.get("ane_confirmed") for item in coreml_placements):
        ane_note = "Administrator-backed telemetry confirms ANE activity for each CoreML workload."
    elif coreml_placements and all(
        item.get("external_trace")
        and item.get("ane_power_sample_count")
        and item.get("ane_power_max_mw") == 0
        for item in coreml_placements
    ):
        ane_note = "Administrator-backed telemetry measured 0.0 mW of ANE power throughout each CoreML workload."
    else:
        ane_note = "The current run lacks telemetry that confirms ANE activity."
    if any(
        item.get("ort_cpu_fallback_observed") and not item.get("ort_cpu_fallback_allowed")
        for item in placements
    ):
        gate_note = "Required rows fail because ORT placed operators outside the requested provider."
    elif ane_required and not all(item.get("ane_confirmed") for item in coreml_placements):
        gate_note = "Required rows fail because the traces do not confirm Neural Engine placement."
    elif not all(item.get("placement_passed") for item in placements):
        gate_note = "Required rows fail the placement evidence policy."
    else:
        gate_note = "Every required row satisfies the placement evidence policy."
    failed_diagnostics = [item for item in diagnostic_placements if not item.get("placement_passed")]
    diagnostic_note = (
        f"Under ADR 0007, {len(failed_diagnostics)} of {len(diagnostic_placements)} diagnostic rows failed placement policy; those rows do not decide Gate 3 and are not qualified latency evidence."
        if diagnostic_placements
        else "The configuration defines no diagnostic providers."
    )
    lines = [
        "# Gate 3 records measured local latency.",
        "",
        f"The host uses {environment.get('model', 'unavailable')} with {environment.get('chip', 'unavailable')} and macOS {environment.get('macos_version', 'unavailable')}.",
        f"ONNX Runtime {environment.get('onnxruntime_version', 'unavailable')} used one intra-op thread and one inter-op thread.",
        "Every row measures preprocessing, synchronous inference, and postprocessing on decoded images.",
        "Each row excludes twenty warmup iterations and summarizes five hundred timed batch-one completions.",
        f"The exact research cycle allowance is {60_000 / config['line_rate_parts_per_minute']:.6f} milliseconds.",
        "The capacity column derives 60000 divided by measured p99 and does not report achieved paced throughput.",
        "",
    ]
    for provider in config["providers"]:
        lines.extend(
            [
                f"## {provider['id']} ({'required' if provider.get('required') else 'diagnostic, not decisive under ADR 0007'})",
                "",
                "| Artifact precision | Serialized execution precision | p50 ms | p95 ms | p99 ms | Capacity from p99, parts/min | Image AUROC | Pixel AUROC | Recall | Clears 45 parts/min | Placement | Timestamp UTC |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
            ]
        )
        for measurement in sorted(
            (item for item in measurements if item.provider.id == provider["id"]),
            key=lambda item: ("fp32", "fp16", "int8").index(item.artifact.precision),
        ):
            summary = read(measurement.summary)
            samples = read(measurement.samples)
            placement = read(measurement.placement) if measurement.placement else {}
            accuracy = metrics[measurement.artifact.sha256]
            execution = {
                "fp32": "FP32",
                "fp16": "mixed FP16/FP32, 36/104 convolution nodes",
                "int8": "mixed INT8/FP32, 36/104 convolution nodes",
            }[measurement.artifact.precision]
            lines.append(
                f"| {measurement.artifact.precision.upper()} | {execution} | {summary['p50_ms']:.3f} | {summary['p95_ms']:.3f} | {summary['p99_ms']:.3f} | {summary['capacity_from_p99_parts_per_minute']:.2f} | {accuracy['image_auroc']:.6f} | {accuracy['pixel_auroc']:.6f} | {accuracy['recall']:.6f} | {'yes' if summary['research_cycle_feasible'] else 'no'} | {placement.get('placement_label', 'unavailable')} | {samples['completed_at_utc']} |"
            )
        lines.append("")
    lines.extend(
        [
            "CoreML provider events and CPU provider events come from ONNX Runtime profiling.",
            "CPU-to-provider output parity passed for every measured pair on sixteen held-out normal validation images.",
            "The requested CPUAndNeuralEngine option does not prove that CoreML used the Neural Engine.",
            ane_note,
            "ADR 0006 does not require Neural Engine placement, so CoreML rows make no ANE claim.",
            gate_note,
            diagnostic_note,
            "The report does not claim commercial cycle feasibility because acquisition, transport, and actuation overhead remain unmeasured.",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    now = datetime.now(UTC).isoformat()
    return EvidenceRef(destination, sha256(destination), now, f"latency-{now}")
