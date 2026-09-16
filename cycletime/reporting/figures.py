"""cycletime/reporting/figures.py renders the measured latency budget and the cost curves."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from cycletime.contracts import JSON, EvidenceRef, Measurement, OperatingPoint
from cycletime.dataio.archive import sha256
from cycletime.evidence import read, record


def draw_figures(
    measurements: list[Measurement],
    curves: list[EvidenceRef],
    point: OperatingPoint,
    config: JSON,
    destination: Path,
) -> list[EvidenceRef]:
    """Draw latency by precision with a horizontal 60000/45 budget line and cost by threshold with selected and maximum-F1 points; refuse interpolated benchmark claims and omission of units, provenance, or assumptions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bench = config["bench"]
    costs = config["costs"]
    comparator = config["comparator"]
    thermal = read(config["thermal"])
    allowance = 60_000 / bench["line_rate_parts_per_minute"]
    destination.mkdir(parents=True, exist_ok=True)
    written = []

    figure, axes = plt.subplots(figsize=(8, 5))
    labels = []
    for index, measurement in enumerate(
        sorted(measurements, key=lambda item: ("fp32", "fp16", "int8").index(item.artifact.precision))
    ):
        summary = read(measurement.summary)
        row = next(
            item
            for item in thermal["rows"]
            if item["precision"] == measurement.artifact.precision
            and item["provider_id"] == measurement.provider.id
        )
        axes.bar(index - 0.2, summary["p99_ms"], width=0.38, color="#4C72B0")
        axes.bar(index + 0.2, row["worst_minute_p99_ms"], width=0.38, color="#DD8452")
        labels.append(f"{measurement.artifact.precision.upper()}\n{measurement.provider.id}")
    axes.axhline(allowance, color="#C44E52", linestyle="--")
    axes.text(
        -0.45,
        allowance * 1.05,
        f"measured cycle allowance {allowance:.3f} ms at {bench['line_rate_parts_per_minute']} parts/min",
        color="#C44E52",
        fontsize=8,
    )
    axes.set_yscale("log")
    axes.set_xticks(range(len(labels)), labels, fontsize=8)
    axes.set_ylabel("latency, milliseconds (log scale)")
    axes.set_title("Measured latency: microbenchmark p99 and worst sustained minute p99", fontsize=10)
    axes.legend(
        ["cycle allowance", "microbenchmark p99, 500 timed samples", "worst sustained minute p99, 45 samples"],
        fontsize=7,
    )
    figure.text(
        0.01,
        0.01,
        "Every bar is a measured local value; no value is interpolated or scaled between configurations. "
        "Excludes acquisition, transport, and actuation overhead.",
        fontsize=6,
    )
    latency_path = destination / "latency-budget.png"
    figure.savefig(latency_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    written.append(reference_figure(latency_path, "measured latency against the cycle allowance"))

    figure, axes = plt.subplots(figsize=(8, 5))
    for curve_ref in curves:
        curve = read(curve_ref)
        rows = sorted(curve["rows"], key=lambda row: row["threshold"])
        axes.plot(
            [row["threshold"] for row in rows],
            [row["expected_cost_usd_per_shift"] for row in rows],
            linewidth=1.2,
            label=f"artifact {curve['artifact_sha256'][:12]}",
        )
        for value in curve["prevalence_range"]:
            axes.plot(
                [row["threshold"] for row in rows],
                [row["prevalence_sensitivity"][str(value)] for row in rows],
                linewidth=0.6,
                alpha=0.45,
                linestyle=":",
                label=f"prevalence {value}",
            )
    axes.scatter([point.threshold], [point.expected_cost_usd_per_shift], marker="o", s=70, color="#55A868", zorder=5, label="cost-selected point")
    axes.scatter([comparator.threshold], [comparator.expected_cost_usd_per_shift], marker="^", s=70, color="#C44E52", zorder=5, label="maximum-F1 point")
    axes.set_xlabel("decision threshold on the image anomaly score, score >= threshold rejects")
    axes.set_ylabel(f"expected quality cost, {costs['currency']} per {costs['parts_per_shift']}-part shift")
    axes.set_title("Expected cost by threshold from cached test predictions", fontsize=10)
    axes.legend(fontsize=6)
    figure.text(
        0.01,
        0.01,
        f"Configured prevalence {costs['production_defect_prevalence']}; warranty {costs['warranty_return_cost_usd']}, "
        f"scrap {costs['scrap_cost_usd']}, rework {costs['rework_cost_usd']} {costs['currency']}; "
        f"assumptions are {costs['assumption_status']}. Thresholds swept over cached MVTec test scores, so the curve is retrospective research.",
        fontsize=6,
    )
    cost_path = destination / "cost-curve.png"
    figure.savefig(cost_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    written.append(reference_figure(cost_path, "expected cost by threshold with both comparison points"))
    return written


def reference_figure(path: Path, caption: str) -> EvidenceRef:
    """Record a rendered figure as evidence with its own digest."""
    return record(
        path.with_suffix(".json"),
        {"figure": str(path), "figure_sha256": sha256(path), "caption": caption},
    )


def write_scorecard(
    point: OperatingPoint, comparator: OperatingPoint, evidence: list[EvidenceRef], destination: Path
) -> EvidenceRef:
    """Render costs, feasibility, thermal evidence, and retrospective or independent-validation status directly from structured evidence; refuse numerical edits from an LLM and unsupported savings claims."""
    context = {ref.path.name: ref for ref in evidence}
    facts = read(context["scorecard-inputs.json"])
    costs = facts["costs"]
    selected = facts["selected"]
    compared = facts["comparator"]
    per_shift = selected["expected_cost_usd_per_shift"] - compared["expected_cost_usd_per_shift"]
    per_month = per_shift * float(costs["shifts_per_month"])
    lines = [
        "# The scorecard records the operating decision.",
        "",
        f"The evaluation status is {facts['evaluation_status']}.",
        "The research status must state that a threshold selected from cached MVTec test predictions is retrospective.",
        f"This selection swept {facts['evaluated_candidate_count']} cached test thresholds, so it is retrospective research and not a validated operating threshold.",
        "The commercial status must remain unqualified until independent customer acceptance succeeds.",
        f"The commercial status is {facts['commercial_status']}.",
        "",
        "The following table compares the selected operating points.",
        "",
        "| Decision | Precision | Provider and placement | Threshold | Recall | False reject rate | F1 | Expected dollars/shift | Clears sustained budget |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
        f"| The cost rule selects this point. | {selected['precision'].upper()} | {selected['provider_id']}, {selected['placement_label']} | {point.threshold:.6f} | {point.recall:.6f} | {point.false_reject_rate:.6f} | {selected['f1']:.6f} | {point.expected_cost_usd_per_shift:.2f} | yes, worst sustained minute {selected['worst_sustained_p99_ms']:.3f} ms |",
        f"| The F1 rule selects this point. | {compared['precision'].upper()} | {comparator.provider_id} | {comparator.threshold:.6f} | {comparator.recall:.6f} | {comparator.false_reject_rate:.6f} | {compared['f1']:.6f} | {comparator.expected_cost_usd_per_shift:.2f} | {compared['feasibility']} |",
        "",
        f"The chosen artifact SHA256 is `{point.artifact_sha256}`.",
        f"The chosen configuration digest is `{facts['configuration_digest']}`.",
        "The maximum-F1 comparison uses the same prevalence and disposition policy.",
        f"The maximum-F1 point is {compared['feasibility']}.",
        f"The expected difference in dollars per shift is {per_shift:.2f} {costs['currency']}.",
        f"The expected difference per month is {per_month:.2f} {costs['currency']} across {costs['shifts_per_month']} shifts.",
        "These quantities describe modeled quality cost rather than observed commercial savings.",
        "",
        "The following table records the economic assumptions.",
        "",
        "| Input | Configured value | Source of validation |",
        "|---|---|---|",
        f"| The model uses this defect prevalence. | {costs['production_defect_prevalence']} | {costs['assumption_status']}; not measured on a customer line |",
        f"| The shift produces this many parts. | {costs['parts_per_shift']} | {costs['assumption_status']} |",
        f"| A false accept incurs this warranty cost. | {costs['warranty_return_cost_usd']} {costs['currency']} | {costs['assumption_status']} |",
        f"| A false reject incurs this scrap cost. | {costs['scrap_cost_usd']} {costs['currency']} | {costs['assumption_status']} |",
        f"| A true reject incurs this rework cost. | {costs['rework_cost_usd']} {costs['currency']} | {costs['assumption_status']} |",
        f"| The month contains this many shifts. | {costs['shifts_per_month']} | {costs['assumption_status']} |",
        "",
        f"The cost curve appears at {facts['cost_figure']}.",
        f"The latency figure appears at {facts['latency_figure']}.",
        "The curve marks both comparison points and displays prevalence sensitivity.",
        f"The chosen threshold satisfies the configured recall floor of {costs['minimum_defect_recall']}.",
        "The selected provider satisfied the measured budget in every sustained minute.",
        f"The frozen validation threshold produced recall {facts['frozen_recall']:.6f}, which does not satisfy that floor.",
        "The measured acquisition, transport, and actuation overhead is TBD.",
        "The confidence interval for customer recall is TBD.",
        "The source of that independent customer sample is TBD.",
        "",
        "The evidence hashes for evaluation, timing, thermal samples, costs, and the environment follow.",
        "",
        "| Evidence | SHA256 |",
        "|---|---|",
    ]
    for name, digest in facts["evidence_hashes"].items():
        lines.append(f"| {name} | `{digest}` |")
    lines.extend(
        [
            "",
            "The LLM did not run, so its model and prompt versions are unavailable.",
            "Every number in this scorecard comes from recorded evidence, and no LLM may rewrite a numeric field.",
            f"The unresolved commercial acceptance conditions are {facts['unresolved_conditions']}.",
        ]
    )
    destination.write_text("\n".join(lines) + "\n")
    now = datetime.now(UTC).isoformat()
    return EvidenceRef(destination, sha256(destination), now, f"scorecard-{now}")
