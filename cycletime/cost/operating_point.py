"""cycletime/cost/operating_point.py selects the least costly configuration that satisfies the constraints."""
from __future__ import annotations

from cycletime.contracts import JSON, EvidenceRef, Measurement, OperatingPoint
from cycletime.cost.curve import validate_costs
from cycletime.evidence import read


class NoFeasiblePointError(ValueError):
    """Carry the rejected candidates when no configuration satisfies the constraints."""

    def __init__(self, message: str, candidates: list[JSON]):
        self.candidates = candidates
        super().__init__(message)


def sustained_worst_p99(thermal: EvidenceRef, artifact_sha256: str, provider_id: str) -> float:
    """Return the measured worst-minute p99 for one pair; refuse a pair without passing sustained evidence."""
    summary = read(thermal)
    if summary["status"] != "passed":
        raise ValueError("The operating point requires passed sustained evidence.")
    for row in summary["rows"]:
        if row["provider_id"] != provider_id or not row["required"]:
            continue
        series = read_series(row)
        if series["artifact_sha256"] == artifact_sha256:
            if row["errors"] or row["worst_minute_p99_ms"] is None:
                raise ValueError(f"Sustained evidence failed for {provider_id}.")
            return float(row["worst_minute_p99_ms"])
    raise ValueError(f"No sustained series covers {artifact_sha256[:12]} on {provider_id}.")


def read_series(row: JSON) -> JSON:
    """Read one recorded sustained series from its evidence reference."""
    from cycletime.evidence import restore

    return read(restore(row["series"]))


def candidate_rows(
    curves: list[EvidenceRef], measurements: list[Measurement], thermal: EvidenceRef, bench: JSON
) -> list[JSON]:
    """Pair every cached threshold with the measured sustained latency of its artifact and provider."""
    allowance_ms = 60_000 / bench["line_rate_parts_per_minute"]
    candidates = []
    for curve_ref in curves:
        curve = read(curve_ref)
        for measurement in measurements:
            if measurement.artifact.sha256 != curve["artifact_sha256"]:
                continue
            worst = sustained_worst_p99(thermal, curve["artifact_sha256"], measurement.provider.id)
            for row in curve["rows"]:
                if {"latency_ms", "p99_ms"} & set(row):
                    raise ValueError("A threshold row must not carry its own latency.")
                candidates.append(
                    {
                        "artifact_sha256": curve["artifact_sha256"],
                        "precision": measurement.artifact.precision,
                        "provider_id": measurement.provider.id,
                        "threshold": row["threshold"],
                        "recall": row["recall"],
                        "false_reject_rate": row["false_positive_rate"],
                        "f1": row["f1"],
                        "expected_cost_usd_per_shift": row["expected_cost_usd_per_shift"],
                        "worst_sustained_p99_ms": worst,
                        "cycle_allowance_ms": allowance_ms,
                        "curve": curve_ref.path.as_posix(),
                    }
                )
    if not candidates:
        raise ValueError("The selection requires at least one measured candidate.")
    return candidates


def order_key(candidate: JSON) -> tuple:
    """Break ties deterministically: cost, then recall, then false rejects, then threshold and artifact."""
    return (
        candidate["expected_cost_usd_per_shift"],
        -candidate["recall"],
        candidate["false_reject_rate"],
        candidate["threshold"],
        candidate["artifact_sha256"],
        candidate["provider_id"],
    )


def select_operating_point(
    curves: list[EvidenceRef],
    measurements: list[Measurement],
    thermal: EvidenceRef,
    costs: JSON,
    bench: JSON,
) -> OperatingPoint:
    """Minimize dollars per shift across artifact, provider, and threshold subject to recall and measured sustained latency; record deterministic ties and fail when no candidate qualifies; refuse optimizing latency with threshold changes when computation is unchanged."""
    validate_costs(costs)
    floor = float(costs["minimum_defect_recall"])
    candidates = candidate_rows(curves, measurements, thermal, bench)
    feasible = [
        candidate
        for candidate in candidates
        if candidate["recall"] >= floor
        and candidate["worst_sustained_p99_ms"] <= candidate["cycle_allowance_ms"]
    ]
    if not feasible:
        raise NoFeasiblePointError(
            f"No candidate satisfies recall {floor} and the measured sustained allowance.",
            candidates,
        )
    best = min(feasible, key=order_key)
    ties = [
        candidate
        for candidate in feasible
        if candidate["expected_cost_usd_per_shift"] == best["expected_cost_usd_per_shift"]
    ]
    best["tie_count"] = len(ties)
    best["feasible_candidate_count"] = len(feasible)
    best["evaluated_candidate_count"] = len(candidates)
    return OperatingPoint(
        artifact_sha256=best["artifact_sha256"],
        provider_id=best["provider_id"],
        threshold=best["threshold"],
        expected_cost_usd_per_shift=best["expected_cost_usd_per_shift"],
        recall=best["recall"],
        false_reject_rate=best["false_reject_rate"],
        feasible=True,
        evidence=(*curves, thermal),
    )


def accuracy_comparator(curves: list[EvidenceRef], config: JSON) -> OperatingPoint:
    """Choose the maximum-F1 point and report its cost under identical prevalence and outcome rules; identify feasibility separately and label the comparison metric explicitly; refuse calling AUROC threshold-dependent or concealing an infeasible comparator."""
    validate_costs(config)
    if config.get("accuracy_maximizing_metric", "f1") != "f1":
        raise ValueError("The comparator reports the maximum-F1 point.")
    rows = []
    for curve_ref in curves:
        curve = read(curve_ref)
        for row in curve["rows"]:
            rows.append(dict(row, artifact_sha256=curve["artifact_sha256"]))
    if not rows:
        raise ValueError("The comparator requires at least one cached threshold.")
    best = max(rows, key=lambda row: (row["f1"], row["recall"], -row["false_positive_rate"]))
    # Feasibility here covers the recall floor only; sustained latency belongs to the selection.
    return OperatingPoint(
        artifact_sha256=best["artifact_sha256"],
        provider_id="comparison_only_not_a_provider_choice",
        threshold=best["threshold"],
        expected_cost_usd_per_shift=best["expected_cost_usd_per_shift"],
        recall=best["recall"],
        false_reject_rate=best["false_positive_rate"],
        feasible=best["recall"] >= float(config["minimum_defect_recall"]),
        evidence=tuple(curves),
    )
