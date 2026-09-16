"""These tests specify required behavior and refuse silent scaffold success."""
import json
from pathlib import Path

import pytest

from cycletime.contracts import Artifact, Measurement, ProviderSpec
from cycletime.cost.curve import confusion_at, expected_cost_curve, expected_cost_per_shift
from cycletime.cost.operating_point import (
    NoFeasiblePointError,
    accuracy_comparator,
    select_operating_point,
)
from cycletime.evidence import record

pytestmark = [pytest.mark.contract]

COSTS = {
    "currency": "USD",
    "assumption_status": "illustrative_research_only",
    "scrap_cost_usd": 8.0,
    "warranty_return_cost_usd": 250.0,
    "rework_cost_usd": 3.0,
    "parts_per_shift": 21600,
    "shifts_per_month": 40,
    "production_defect_prevalence": 0.01,
    "prevalence_range": [0.001, 0.01, 0.05],
    "false_accept_treatment": "warranty_return",
    "false_reject_treatment": "scrap",
    "true_reject_treatment": "rework",
    "true_accept_treatment": "no_incremental_cost",
    "count_outcomes_exclusively": True,
    "minimum_defect_recall": 0.95,
}


def cache(tmp_path: Path, scores: list[float], labels: list[int], sha: str = "a" * 64):
    """Record a synthetic completed evaluation cache for one artifact."""
    directory = tmp_path / sha
    directory.mkdir(parents=True, exist_ok=True)
    return record(
        directory / "predictions.json",
        {
            "artifact_sha256": sha,
            "full_test_split_complete": True,
            "images": [
                {"image_id": f"i{index}", "score": score, "label": label}
                for index, (score, label) in enumerate(zip(scores, labels, strict=True))
            ],
        },
    )


def test_exclusive_outcome_costs(tmp_path: Path) -> None:
    """A hand-counted confusion matrix must match the configured warranty, scrap, and true-reject rework formula without duplicate costs."""
    scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
    labels = [1, 1, 0, 1, 0, 0]
    rates = confusion_at(__import__("numpy").asarray(scores), __import__("numpy").asarray(labels), 0.7)
    assert (rates["tp"], rates["fp"], rates["fn"], rates["tn"]) == (2, 1, 1, 2)
    assert rates["true_positive_rate"] == pytest.approx(2 / 3)
    assert rates["false_negative_rate"] == pytest.approx(1 / 3)
    assert rates["false_positive_rate"] == pytest.approx(1 / 3)

    # 21600 * [0.01*(1/3)*250 + 0.99*(1/3)*8 + 0.01*(2/3)*3]
    expected = 21600 * (0.01 * (1 / 3) * 250 + 0.99 * (1 / 3) * 8 + 0.01 * (2 / 3) * 3)
    assert expected_cost_per_shift(rates, COSTS, 0.01) == pytest.approx(expected)

    # A rejected defective part pays rework only, never rework plus scrap.
    rework_only = expected_cost_per_shift(
        {"true_positive_rate": 1.0, "false_negative_rate": 0.0, "false_positive_rate": 0.0},
        COSTS,
        0.01,
    )
    assert rework_only == pytest.approx(21600 * 0.01 * 3.0)

    for missing in ("warranty_return_cost_usd", "parts_per_shift"):
        with pytest.raises(ValueError, match="requires configured values"):
            expected_cost_curve(cache(tmp_path, scores, labels), dict(COSTS, **{missing: None}))
    with pytest.raises(ValueError, match="exclusive outcome"):
        expected_cost_curve(cache(tmp_path, scores, labels), dict(COSTS, count_outcomes_exclusively=False))


def test_prevalence_changes_decision(tmp_path: Path) -> None:
    """Changing configured production prevalence must change expected cost without changing cached measured accuracy."""
    scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
    labels = [1, 1, 0, 1, 0, 0]
    low = json.loads(expected_cost_curve(cache(tmp_path / "low", scores, labels), COSTS).path.read_text())
    high = json.loads(
        expected_cost_curve(
            cache(tmp_path / "high", scores, labels), dict(COSTS, production_defect_prevalence=0.05)
        ).path.read_text()
    )
    assert [row["recall"] for row in low["rows"]] == [row["recall"] for row in high["rows"]]
    assert [row["expected_cost_usd_per_shift"] for row in low["rows"]] != [
        row["expected_cost_usd_per_shift"] for row in high["rows"]
    ]
    # The cached test split is half defective; the model must never adopt that ratio as prevalence.
    assert low["observed_test_defect_ratio"] == pytest.approx(0.5)
    assert low["configured_prevalence"] == 0.01 and low["test_ratio_used_as_prevalence"] is False
    assert set(low["rows"][0]["prevalence_sensitivity"]) == {"0.001", "0.01", "0.05"}


def measurement(precision: str, provider_id: str, sha: str, tmp_path: Path) -> Measurement:
    """Build a measurement whose evidence identifies one artifact and provider."""
    path = tmp_path / f"{precision}.onnx"
    path.write_bytes(b"model")
    artifact = Artifact(path, sha, precision, "pre", "data")
    provider = ProviderSpec(provider_id, "CPUExecutionProvider", {}, False)
    ref = record(tmp_path / f"{precision}-{provider_id}.json", {"p99_ms": 15.0})
    return Measurement(artifact, provider, ref, ref, ref, ref)


def thermal_summary(tmp_path: Path, worst_ms: float, sha: str = "a" * 64, provider_id: str = "cpu"):
    """Record a passed sustained summary for one pair."""
    series = record(tmp_path / f"series-{provider_id}-{worst_ms}.json", {"artifact_sha256": sha})
    from cycletime.evidence import serialize

    return record(
        tmp_path / f"summary-{provider_id}-{worst_ms}.json",
        {
            "status": "passed",
            "rows": [
                {
                    "provider_id": provider_id,
                    "required": True,
                    "series": serialize(series),
                    "worst_minute_p99_ms": worst_ms,
                    "errors": [],
                }
            ],
        },
    )


def test_reject_infeasible_candidates(tmp_path: Path) -> None:
    """A cheaper point that violates recall or worst-minute latency must lose to a feasible point, and an empty feasible set must fail."""
    # Overlapping scores: a high threshold is cheaper but misses defects.
    scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
    labels = [1, 1, 0, 1, 0, 0]
    curve = expected_cost_curve(cache(tmp_path, scores, labels), COSTS)
    bench = {"line_rate_parts_per_minute": 45}
    measurements = [measurement("fp32", "cpu", "a" * 64, tmp_path)]

    point = select_operating_point([curve], measurements, thermal_summary(tmp_path, 30.0), COSTS, bench)
    assert point.recall >= COSTS["minimum_defect_recall"]
    rows = json.loads(curve.path.read_text())["rows"]
    cheaper = [row for row in rows if row["expected_cost_usd_per_shift"] < point.expected_cost_usd_per_shift]
    assert cheaper and all(row["recall"] < COSTS["minimum_defect_recall"] for row in cheaper)

    slow = thermal_summary(tmp_path, 2_000.0)
    with pytest.raises(NoFeasiblePointError, match="No candidate satisfies"):
        select_operating_point([curve], measurements, slow, COSTS, bench)
    with pytest.raises(NoFeasiblePointError):
        select_operating_point(
            [curve], measurements, thermal_summary(tmp_path, 30.0), dict(COSTS, minimum_defect_recall=1.01), bench
        )


def test_avoid_test_leakage(tmp_path: Path) -> None:
    """A threshold chosen from cached MVTec test scores must carry retrospective status and must fail commercial release acceptance."""
    scores = [0.9, 0.8, 0.7, 0.6, 0.2, 0.1]
    labels = [1, 1, 1, 1, 0, 0]
    curve_ref = expected_cost_curve(cache(tmp_path, scores, labels), COSTS)
    curve = json.loads(curve_ref.path.read_text())
    assert curve["threshold_source"] == "swept_over_cached_test_scores_retrospective_research_only"
    assert curve["evaluation_scope"] == "cached_full_mvtec_test_split"

    comparator = accuracy_comparator([curve_ref], dict(COSTS, accuracy_maximizing_metric="f1"))
    assert comparator.provider_id == "comparison_only_not_a_provider_choice"

    # Research selection never becomes release acceptance: the release check refuses the
    # research evidence outright, naming passed research gates as insufficient.
    from pathlib import Path as _Path

    from cycletime.gates import release_check

    with pytest.raises(ValueError, match="not qualified") as refused:
        release_check(_Path("config"))
    assert "research gates passed" in str(refused.value)
    assert "no independent customer validation has been recorded" in str(refused.value)
