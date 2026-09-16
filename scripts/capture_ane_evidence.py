"""Capture administrator-backed ANE power samples during isolated CoreML workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import onnxruntime as ort
import yaml

PROJECT = Path(__file__).resolve().parents[1]
POWER_RE = re.compile(r"ANE\s+Power:\s*([0-9]+(?:\.[0-9]+)?)\s*mW", re.IGNORECASE)


def sha256(path: Path) -> str:
    """Hash a captured trace without treating the trace as an independent assertion."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_paths() -> list[Path]:
    """Resolve the three Gate 2 artifacts and refuse missing or changed exports."""
    gate2 = json.loads((PROJECT / "artifacts/gate2.json").read_text())
    expected = {row["artifact_sha256"] for row in gate2["exports"]}
    selected = [
        PROJECT / gate2["export_config"][precision]["path"]
        for precision in ("fp32", "fp16", "int8")
    ]
    if len(selected) != 3 or {sha256(path) for path in selected} != expected:
        raise RuntimeError("Gate 2 does not identify three unchanged local exports.")
    return selected


def capture(path: Path, provider: dict, destination: Path) -> dict:
    """Run one isolated graph while powermetrics samples the ANE power rail."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(path), providers=[(provider["name"], provider["options"])], sess_options=options
    )
    sample = np.zeros((1, 3, 256, 256), dtype=np.float32)
    for _ in range(20):
        session.run(None, {"image": sample})

    stop = threading.Event()
    completed = 0
    failures = []

    def burn() -> None:
        nonlocal completed
        try:
            while not stop.is_set():
                session.run(None, {"image": sample})
                completed += 1
        except Exception as exc:  # noqa: BLE001 - the capture preserves any workload failure.
            failures.append(exc)
            stop.set()

    worker = threading.Thread(target=burn, daemon=True)
    worker.start()
    started = datetime.now(UTC).isoformat()
    try:
        subprocess.run(
            [
                "/usr/bin/powermetrics",
                "--samplers",
                "cpu_power,gpu_power,ane_power",
                "--show-extra-power-info",
                "--sample-rate",
                "200",
                "--sample-count",
                "40",
                "--output-file",
                str(destination),
            ],
            check=True,
        )
    finally:
        stop.set()
        worker.join()
    completed_at = datetime.now(UTC).isoformat()
    if failures or completed == 0:
        raise RuntimeError("The inference workload failed during ANE capture.") from (
            failures[0] if failures else None
        )
    values = [float(value) for value in POWER_RE.findall(destination.read_text(errors="replace"))]
    if not values:
        raise RuntimeError(
            f"powermetrics exposed no ANE metrics for {path.name}; preserve {destination} "
            "as evidence that ANE execution could not be confirmed."
        )
    return {
        "artifact_path": str(path),
        "artifact_sha256": sha256(path),
        "provider_id": provider["id"],
        "provider_name": provider["name"],
        "requested_options": provider["options"],
        "capture_started_at_utc": started,
        "capture_completed_at_utc": completed_at,
        "workload_completions": completed,
        "trace_path": str(destination),
        "trace_sha256": sha256(destination),
        "ane_power_samples_mw": values,
        "ane_active": any(value > 0 for value in values),
    }


def arguments() -> argparse.Namespace:
    """Parse an explicit CoreML format and refuse arbitrary provider values."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-format", choices=("MLProgram", "NeuralNetwork"), default="MLProgram"
    )
    return parser.parse_args()


def main() -> None:
    """Capture every export and refuse execution without administrator privileges."""
    args = arguments()
    if os.geteuid() != 0:
        raise SystemExit("Run this helper with sudo so powermetrics can sample ANE power.")
    config = yaml.safe_load((PROJECT / "config/bench.yaml").read_text())
    configured = next(row for row in config["providers"] if row["id"] == "coreml_cpu_ane")
    provider = dict(configured, options=dict(configured["options"], ModelFormat=args.model_format))
    suffix = "mlprogram" if args.model_format == "MLProgram" else "neuralnetwork"
    output_dir = PROJECT / (
        "artifacts/placement/ane"
        if args.model_format == "MLProgram"
        else f"artifacts/placement/ane-{suffix}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = {}
    for path in artifact_paths():
        digest = sha256(path)
        entries[digest] = capture(path, provider, output_dir / f"{path.stem}.txt")
    record = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": str(uuid4()),
        "capture_tool": "/usr/bin/powermetrics",
        "sampler": "cpu_power,gpu_power,ane_power",
        "sample_rate_ms": 200,
        "sample_count": 40,
        "model_format": args.model_format,
        "artifacts": entries,
    }
    destination = PROJECT / (
        "artifacts/placement/ane-evidence.json"
        if args.model_format == "MLProgram"
        else f"artifacts/placement/ane-evidence-{suffix}.json"
    )
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if os.environ.get("SUDO_UID") and os.environ.get("SUDO_GID"):
        owner = (int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"]))
        for path in [destination, *output_dir.glob("*.txt")]:
            os.chown(path, *owner)
    print(destination)
    for entry in entries.values():
        print(
            f"{Path(entry['artifact_path']).name}: "
            f"ANE maximum {max(entry['ane_power_samples_mw']):.3f} mW; "
            f"activity observed={entry['ane_active']}"
        )
    if not all(entry["ane_active"] for entry in entries.values()):
        raise SystemExit("Capture completed, but ANE activity was not observed for every artifact.")


if __name__ == "__main__":
    main()
