"""These tests require complete hardware evidence from the recorded host."""
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.contract, pytest.mark.hardware]


def test_measure_required_matrix() -> None:
    """The actual host must execute every configured precision/provider pair with real sample timestamps and verified placement."""
    report = json.loads(Path("artifacts/gate3.json").read_text())
    matrix = json.loads(Path(report["matrix"]["path"]).read_text())
    assert report["measurement_count"] == report["required_measurement_count"] == 6
    assert not report["provider_parity_errors"]
    assert matrix["status"] == "passed", "\n".join(matrix["errors"])


def test_measure_full_sustained_runs() -> None:
    """The actual host must complete thirty measured minutes per required pair without concurrent benchmark or LLM work."""
    report = json.loads(Path("artifacts/gate4.json").read_text())
    summary = json.loads(Path(report["summary"]["path"]).read_text())
    assert report["status"] == "passed", "\n".join(report["errors"])
    rows = {(row["precision"], row["provider_id"]): row for row in summary["rows"] if row["required"]}
    expected = {(p, provider) for provider in report["required_providers"] for p in ("fp32", "fp16", "int8")}
    assert set(rows) == expected
    for row in rows.values():
        assert row["window_count"] == 30 and row["completed_count"] == 1350
        assert row["missed_slot_count"] == 0 and row["deadline_miss_count"] == 0
        series = json.loads(Path(row["series"]["path"]).read_text())
        assert series["starting_conditions"]["agent_enabled"] is False
        assert series["warmup_excluded"] is True
