"""cycletime/cost/curve.py converts cached confusion rates into expected cost."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cycletime.contracts import JSON, EvidenceRef
from cycletime.evidence import read, record, serialize

REQUIRED_COSTS = (
    "scrap_cost_usd",
    "warranty_return_cost_usd",
    "rework_cost_usd",
    "parts_per_shift",
    "production_defect_prevalence",
    "shifts_per_month",
)


def validate_costs(costs: JSON) -> None:
    """Refuse unknown required costs, non-exclusive outcomes, and dispositions the ADR does not define."""
    missing = [key for key in REQUIRED_COSTS if costs.get(key) is None]
    if missing:
        raise ValueError(f"The cost model requires configured values for {missing}.")
    if not costs.get("count_outcomes_exclusively"):
        raise ValueError("The cost model requires exclusive outcome counting.")
    dispositions = {
        "false_accept_treatment": "warranty_return",
        "false_reject_treatment": "scrap",
        "true_reject_treatment": "rework",
        "true_accept_treatment": "no_incremental_cost",
    }
    for key, expected in dispositions.items():
        if costs.get(key) != expected:
            raise ValueError(f"The cost model requires {key} = {expected}.")
    if not 0 < float(costs["production_defect_prevalence"]) < 1:
        raise ValueError("Configured prevalence must lie strictly between zero and one.")


def expected_cost_per_shift(rates: JSON, costs: JSON, prevalence: float) -> float:
    """Return N*[pi*FNR*Cw + (1-pi)*FPR*Cs + pi*TPR*Cr]; each inspected part contributes exactly one disposition."""
    parts = float(costs["parts_per_shift"])
    warranty = float(costs["warranty_return_cost_usd"])
    scrap = float(costs["scrap_cost_usd"])
    rework = float(costs["rework_cost_usd"])
    return parts * (
        prevalence * rates["false_negative_rate"] * warranty
        + (1 - prevalence) * rates["false_positive_rate"] * scrap
        + prevalence * rates["true_positive_rate"] * rework
    )


def confusion_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> JSON:
    """Count one exclusive outcome per cached test image at the decision rule score >= threshold."""
    rejected = scores >= threshold
    tp = int(np.sum(rejected & (labels == 1)))
    fp = int(np.sum(rejected & (labels == 0)))
    fn = int(np.sum(~rejected & (labels == 1)))
    tn = int(np.sum(~rejected & (labels == 0)))
    positives = tp + fn
    negatives = fp + tn
    if positives == 0 or negatives == 0:
        raise ValueError("The cached evaluation must contain defective and normal test images.")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "true_positive_rate": recall,
        "false_negative_rate": fn / positives,
        "false_positive_rate": fp / negatives,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def expected_cost_curve(metrics: EvidenceRef, costs: JSON) -> EvidenceRef:
    """Calculate N*[pi*FNR*Cw + (1-pi)*FPR*Cs + pi*TPR*Cr] at each threshold from configuration; treat outcomes exclusively and record prevalence sensitivity; refuse hardcoded dollars, test-ratio prevalence, double-counted rework, and unknown required costs."""
    validate_costs(costs)
    cache = read(metrics)
    if not cache.get("full_test_split_complete"):
        raise ValueError("The cost curve requires one complete cached test evaluation.")
    images = cache["images"]
    scores = np.asarray([float(row["score"]) for row in images], dtype=np.float64)
    labels = np.asarray([int(row["label"]) for row in images], dtype=np.int64)
    if not np.isfinite(scores).all() or not set(np.unique(labels)) <= {0, 1}:
        raise ValueError("Cached predictions must carry finite scores and binary labels.")
    prevalence = float(costs["production_defect_prevalence"])
    # The MVTec test split over-represents defects; prevalence comes only from configuration.
    observed_ratio = float(np.mean(labels))
    candidates = sorted({float(value) for value in scores} | {float(scores.min()) - 1e-9})
    rows = []
    for threshold in candidates:
        rates = confusion_at(scores, labels, threshold)
        rates["expected_cost_usd_per_shift"] = expected_cost_per_shift(rates, costs, prevalence)
        rates["prevalence_sensitivity"] = {
            str(value): expected_cost_per_shift(rates, costs, float(value))
            for value in costs["prevalence_range"]
        }
        rows.append(rates)
    # The curve belongs beside the cached evaluation of its own artifact.
    destination = Path(metrics.path).with_name("cost-curve.json")
    return record(
        destination,
        {
            "artifact_sha256": cache["artifact_sha256"],
            "predictions": serialize(metrics),
            "evaluation_scope": "cached_full_mvtec_test_split",
            "threshold_source": "swept_over_cached_test_scores_retrospective_research_only",
            "decision_rule": "score_greater_than_or_equal_to_threshold_is_reject",
            "configured_prevalence": prevalence,
            "prevalence_range": list(costs["prevalence_range"]),
            "observed_test_defect_ratio": observed_ratio,
            "prevalence_source": "config/costs.yaml production_defect_prevalence",
            "test_ratio_used_as_prevalence": False,
            "currency": costs["currency"],
            "assumption_status": costs["assumption_status"],
            "cost_inputs": {key: costs[key] for key in REQUIRED_COSTS},
            "outcome_policy": {
                "false_accept": costs["false_accept_treatment"],
                "false_reject": costs["false_reject_treatment"],
                "true_reject": costs["true_reject_treatment"],
                "true_accept": costs["true_accept_treatment"],
                "exclusive": True,
            },
            "rows": rows,
        },
    )
