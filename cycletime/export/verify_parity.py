"""This module compares every declared output on held-out normal validation images."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from cycletime.contracts import JSON, Artifact, EvidenceRef, ProviderSpec
from cycletime.dataio.calibration import validate_partitions
from cycletime.dataio.images import load_image
from cycletime.evidence import record
from cycletime.export.metadata import load_artifact
from cycletime.model.train import load_checkpoint


class ParityError(ValueError):
    """Carry failed parity evidence without turning a failure into a pass."""

    def __init__(self, report: EvidenceRef):
        self.report = report
        super().__init__(f"Output parity failed. Evidence: {report.path}")


def compare_outputs(
    expected: list[np.ndarray], actual: list[np.ndarray], tolerances: dict
) -> list[dict]:
    """Compare both outputs elementwise against the reference magnitude; refuse omitted, nonfinite, or mismatched outputs."""
    if len(expected) != 2 or len(actual) != 2:
        raise ValueError("Both image_score and anomaly_map outputs are required.")
    if any(not math.isfinite(tolerances[k]) or tolerances[k] < 0 for k in ("atol", "rtol")):
        raise ValueError("Tolerances must be finite and nonnegative.")
    rows = []
    for name, a, b in zip(["image_score", "anomaly_map"], expected, actual, strict=True):
        if (
            a.shape != b.shape
            or a.size == 0
            or not np.isfinite(a).all()
            or not np.isfinite(b).all()
        ):
            raise ValueError(f"Invalid {name} output shapes or values.")
        error = np.abs(a.astype(np.float64) - b.astype(np.float64))
        allowance = tolerances["atol"] + tolerances["rtol"] * np.abs(a.astype(np.float64))
        violations = error > allowance
        rows.append(
            {
                "output": name,
                "element_count": a.size,
                "violations": int(violations.sum()),
                "maximum_absolute_error": float(error.max()),
                "maximum_excess_error": float(np.maximum(error - allowance, 0).max()),
                "passed": not bool(violations.any()),
            }
        )
    return rows


def cpu_session(artifact: Artifact):
    """Open the exact artifact on CPU; refuse changed bytes or missing declared outputs."""
    verified, _ = load_artifact(artifact.path)
    if verified != artifact:
        raise ValueError("Artifact provenance changed.")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(artifact.path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if [n.name for n in session.get_outputs()] != ["image_score", "anomaly_map"]:
        raise ValueError("The export must declare image_score and anomaly_map in order.")
    return session


def verify_export(
    reference: EvidenceRef | Artifact,
    candidate: Artifact,
    validation: EvidenceRef,
    tolerances: JSON,
) -> EvidenceRef:
    """Check PyTorch/FP32 or FP32/reduced parity on normal validation inputs; refuse omitted outputs, test sweeps, or tolerance widening."""
    parts = validate_partitions(validation)
    config = parts["dataset_config"]
    project = Path(config["_project"])
    count = tolerances["sample_count"]
    if (
        tolerances["outputs"] != ["image_score", "anomaly_map"]
        or tolerances["input_source"] != "held_out_normal_validation"
        or tolerances["allow_automatic_tolerance_widening"]
        or not 0 < count <= len(parts["validation"])
    ):
        raise ValueError("Unsupported parity protocol or insufficient validation samples.")
    if isinstance(reference, Artifact):
        if reference.precision != "fp32" or candidate.precision not in ("fp16", "int8"):
            raise ValueError("Reduced exports must compare against canonical FP32.")
        if (reference.dataset_sha256, reference.preprocessing_sha256) != (
            candidate.dataset_sha256,
            candidate.preprocessing_sha256,
        ):
            raise ValueError("Export provenance differs from its FP32 reference.")
        session = cpu_session(reference)

        def infer(x):
            return session.run(["image_score", "anomaly_map"], {"image": x})

        bounds = tolerances[f"{candidate.precision}_vs_fp32"]
        reference_hash = reference.sha256
    else:
        model, _ = load_checkpoint(reference)
        if candidate.precision != "fp32":
            raise ValueError("A PyTorch checkpoint must compare against FP32.")

        def infer(x):
            with torch.inference_mode():
                return [v.numpy() for v in model(torch.from_numpy(x))]

        bounds = tolerances["fp32_vs_pytorch"]
        reference_hash = reference.sha256
    candidate_session = cpu_session(candidate)
    observations = []
    for name in parts["validation"][:count]:
        x = load_image(project / config["root"] / name, config)[None]
        actual = candidate_session.run(["image_score", "anomaly_map"], {"image": x})
        observations.append(
            {"image_id": name, "outputs": compare_outputs(infer(x), actual, bounds)}
        )
    passed = all(output["passed"] for row in observations for output in row["outputs"])
    report = record(
        project / f"artifacts/parity/{candidate.path.stem}.json",
        {
            "status": "passed" if passed else "failed",
            "reference_sha256": reference_hash,
            "candidate_sha256": candidate.sha256,
            "validation_sha256": validation.sha256,
            "sample_count": count,
            "tolerances": bounds,
            "observations": observations,
            "onnxruntime_version": ort.__version__,
            "provider": "CPUExecutionProvider",
        },
    )
    if not passed:
        raise ParityError(report)
    return report


def verify_provider(
    artifact: Artifact, provider: ProviderSpec, validation: EvidenceRef, tolerances: JSON
) -> EvidenceRef:
    """Compare a provider with CPU on normal validation inputs; refuse test-set inference or missing outputs."""
    parts = validate_partitions(validation)
    config = parts["dataset_config"]
    project = Path(config["_project"])
    bench = tolerances["_bench"]
    if tolerances["input_source"] != "held_out_normal_validation":
        raise ValueError("Provider parity may use only held-out normal validation images.")
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(bench["intra_op_threads"])
    options.inter_op_num_threads = int(bench["inter_op_threads"])
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    reference = ort.InferenceSession(
        str(artifact.path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    candidate = ort.InferenceSession(
        str(artifact.path),
        sess_options=options,
        providers=[(provider.name, provider.options)],
    )
    if provider.name not in candidate.get_providers():
        raise ValueError(f"Provider parity could not activate {provider.name}.")
    count = tolerances["sample_count"]
    observations = []
    for name in parts["validation"][:count]:
        value = load_image(project / config["root"] / name, config)[None]
        expected = reference.run(["image_score", "anomaly_map"], {"image": value})
        actual = candidate.run(["image_score", "anomaly_map"], {"image": value})
        observations.append(
            {
                "image_id": name,
                "outputs": compare_outputs(expected, actual, tolerances["provider_vs_cpu"]),
            }
        )
    passed = all(output["passed"] for row in observations for output in row["outputs"])
    report = record(
        project / f"artifacts/parity/providers/{artifact.path.stem}-{provider.id}.json",
        {
            "status": "passed" if passed else "failed",
            "artifact_sha256": artifact.sha256,
            "provider_id": provider.id,
            "requested_provider": provider.name,
            "active_providers": candidate.get_providers(),
            "reference_provider": "CPUExecutionProvider",
            "validation_sha256": validation.sha256,
            "sample_count": count,
            "tolerances": tolerances["provider_vs_cpu"],
            "observations": observations,
            "placement_claimed": False,
        },
    )
    if not passed:
        raise ParityError(report)
    return report
