"""These helpers preserve export identity and report precision coverage honestly."""

from dataclasses import asdict
from pathlib import Path

import onnx

from cycletime.contracts import Artifact
from cycletime.dataio.archive import sha256
from cycletime.evidence import read, record, reference


def load_artifact(path: Path) -> tuple[Artifact, dict]:
    """Load a verified export and its metadata; refuse changed or external weights."""
    metadata = read(reference(path.with_suffix(".json")))
    artifact = Artifact(**dict(metadata["artifact"], path=path))
    if sha256(path) != artifact.sha256:
        raise ValueError("Export hash mismatch.")
    graph = onnx.load(path, load_external_data=False)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
        raise ValueError("Unrecorded external weights are forbidden.")
    return artifact, metadata


def reuse(path: Path, identity: str) -> Artifact | None:
    """Reuse an identical completed export; refuse replacement or incomplete evidence."""
    if not path.exists() and not path.with_suffix(".json").exists():
        return None
    artifact, metadata = load_artifact(path)
    if metadata["identity"] != identity:
        raise ValueError(
            "Existing export belongs to a different conversion; record a new decision first."
        )
    return artifact


def save_export(
    path: Path, temporary: Path, source: Artifact, precision: str, identity: str, details: dict
) -> Artifact:
    """Publish a checked graph with source provenance; refuse an unchanged precision label."""
    onnx.checker.check_model(str(temporary), full_check=True)
    checksum = sha256(temporary)
    if checksum == source.sha256:
        raise ValueError("An unchanged FP32 graph cannot receive a reduced precision label.")
    _, metadata = load_artifact(source.path)
    artifact = Artifact(
        path, checksum, precision, source.preprocessing_sha256, source.dataset_sha256
    )
    temporary.replace(path)
    record(
        path.with_suffix(".json"),
        dict(
            details,
            identity=identity,
            artifact=dict(asdict(artifact), path=str(path)),
            source_fp32_sha256=source.sha256,
            checkpoint_sha256=metadata["checkpoint_sha256"],
        ),
    )
    return artifact


def int8_coverage(model: onnx.ModelProto) -> dict:
    """Identify signed INT8 QDQ compute boundaries and retained ops; refuse absent static quantization."""
    graph = model.graph
    producers = {out: n for n in graph.node for out in n.output}
    initializers = {t.name: t for t in graph.initializer}
    covered, retained = [], []

    def signed_dq(name):
        node = producers.get(name)
        if node is None or node.op_type != "DequantizeLinear" or len(node.input) < 3:
            return False
        zero = initializers.get(node.input[2])
        return zero is not None and zero.data_type == onnx.TensorProto.INT8

    for node in graph.node:
        if node.op_type == "DynamicQuantizeLinear":
            raise ValueError("Dynamic quantization cannot substitute for static INT8.")
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            if node.input[1] not in initializers:
                raise ValueError("Quantization scales must be static initializers.")
            continue
        row = {"name": node.name, "op_type": node.op_type}
        if (
            node.op_type in {"Conv", "Gemm", "MatMul"}
            and len(node.input) >= 2
            and all(signed_dq(n) for n in node.input[:2])
        ):
            covered.append(row)
        else:
            retained.append(row)
    int8_weights = [
        t.name
        for t in graph.initializer
        if t.data_type == onnx.TensorProto.INT8 and len(t.dims) >= 2
    ]
    if not covered or not int8_weights:
        raise ValueError("The graph contains no signed INT8 QDQ compute or weights.")
    return {
        "qdq_compute_nodes": covered,
        "other_nodes": retained,
        "int8_weight_tensors": int8_weights,
        "qdq_compute_count": len(covered),
        "other_node_count": len(retained),
        "scope": "serialized_graph_boundaries_not_runtime_kernel_placement",
    }
