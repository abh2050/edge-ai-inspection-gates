"""This module exports and checks the canonical FP32 graph without test access."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from cycletime.contracts import JSON, Artifact, EvidenceRef
from cycletime.dataio.archive import sha256
from cycletime.dataio.calibration import validate_partitions
from cycletime.dataio.images import load_image
from cycletime.evidence import digest, read, record, reference, restore
from cycletime.model.train import load_checkpoint


def export_fp32(checkpoint: EvidenceRef, export_config: JSON) -> Artifact:
    """Export both networks and scores with batch-one shapes and verify validation parity; refuse unrecorded external state or replacement of a canonical artifact."""
    data = read(checkpoint)
    config = data["dataset_config"]
    project = Path(config["_project"])
    path = project / export_config["fp32"]["path"]
    metadata = path.with_suffix(".json")
    identity = digest(
        {"checkpoint": checkpoint.sha256, "export": export_config, "code": sha256(Path(__file__))}
    )
    if path.exists() or metadata.exists():
        if not path.exists() or not metadata.exists():
            raise ValueError("Incomplete canonical export requires an explicit recovery decision.")
        saved = read(reference(metadata))
        if saved["identity"] != identity or sha256(path) != saved["artifact"]["sha256"]:
            raise ValueError("Canonical export changed or belongs to a different experiment.")
        return Artifact(**dict(saved["artifact"], path=path))
    if export_config["input_shape"] != [1, 3, 256, 256] or not export_config["static_shapes"]:
        raise ValueError("The model requires static batch-one 256-pixel input.")
    model, _ = load_checkpoint(checkpoint)
    parts = validate_partitions(restore(data["partitions"]))
    count = export_config["parity"]["sample_count"]
    if count > len(parts["validation"]):
        raise ValueError("Insufficient held-out validation images for parity.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending.onnx")
    torch.onnx.export(
        model,
        torch.zeros(*export_config["input_shape"]),
        str(temporary),
        opset_version=export_config["opset_version"],
        input_names=[export_config["input_name"]],
        output_names=export_config["output_names"],
        dynamo=False,
        external_data=False,
    )
    graph = onnx.load(temporary)
    onnx.checker.check_model(graph)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
        raise ValueError("The canonical export must contain all weights.")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(temporary), sess_options=options, providers=["CPUExecutionProvider"]
    )
    tolerances = export_config["parity"]["fp32_vs_pytorch"]
    maximum_errors = [0.0, 0.0]
    with torch.inference_mode():
        for name in parts["validation"][:count]:
            x = load_image(project / config["root"] / name, config)[None]
            expected = [v.numpy() for v in model(torch.from_numpy(x))]
            actual = session.run(export_config["output_names"], {export_config["input_name"]: x})
            for i, (a, b) in enumerate(zip(expected, actual, strict=True)):
                if (
                    a.shape != b.shape
                    or not np.isfinite(b).all()
                    or not np.allclose(a, b, **tolerances)
                ):
                    raise ValueError(f"FP32 export parity failed on {name}, output {i}.")
                maximum_errors[i] = max(maximum_errors[i], float(np.max(np.abs(a - b))))
    temporary.replace(path)
    artifact = Artifact(
        path, sha256(path), "fp32", data["preprocessing_sha256"], data["dataset_sha256"]
    )
    record(
        metadata,
        {
            "identity": identity,
            "artifact": dict(asdict(artifact), path=str(path)),
            "checkpoint_sha256": checkpoint.sha256,
            "parity": {
                "passed": True,
                "sample_count": count,
                "maximum_absolute_errors": maximum_errors,
                "tolerances": tolerances,
            },
            "onnxruntime_version": ort.__version__,
            "onnx_version": onnx.__version__,
        },
    )
    return artifact
