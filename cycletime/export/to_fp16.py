"""This module converts eligible graph tensors to FP16 and preserves configured IO."""

from __future__ import annotations

from pathlib import Path

import onnx
import onnxconverter_common
from onnxconverter_common.float16 import convert_float_to_float16

from cycletime.contracts import JSON, Artifact
from cycletime.dataio.archive import sha256
from cycletime.evidence import digest
from cycletime.export.metadata import load_artifact, reuse, save_export


def isolate_shared_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """Separate internal consumers from public output casts; refuse preexisting alias collisions."""
    graph = model.graph
    all_names = {name for node in graph.node for name in list(node.input) + list(node.output)}
    for output in graph.output:
        name = output.name
        if not any(name in node.input for node in graph.node):
            continue
        internal = name + "__internal_before_io_cast"
        if internal in all_names:
            raise ValueError("The output alias already exists.")
        for node in graph.node:
            for values in (node.input, node.output):
                for index, value in enumerate(values):
                    if value == name:
                        values[index] = internal
        for value in graph.value_info:
            if value.name == name:
                value.name = internal
        graph.node.append(
            onnx.helper.make_node("Identity", [internal], [name], name=name + "__public_output")
        )
    return model


def export_fp16(fp32: Artifact, export_config: JSON) -> Artifact:
    """Convert eligible operations with configured IO types and record retained FP32 nodes; refuse unsupported conversion or an unchanged graph."""
    source, _ = load_artifact(fp32.path)
    if source != fp32 or source.precision != "fp32":
        raise ValueError("FP16 conversion requires the verified canonical FP32 artifact.")
    config = export_config["fp16"]
    path = fp32.path.parents[2] / config["path"]
    identity = digest(
        {
            "source": fp32.sha256,
            "config": config,
            "converter": onnxconverter_common.__version__,
            "code": sha256(Path(__file__)),
            "metadata_code": sha256(Path(__file__).with_name("metadata.py")),
        }
    )
    if cached := reuse(path, identity):
        return cached
    source_model = isolate_shared_outputs(onnx.load(fp32.path))
    selected = [
        node.name
        for node in source_model.graph.node
        if node.op_type == "Conv" and config["selected_node_substring"] in node.name
    ]
    candidates = [node.name for node in source_model.graph.node if node.op_type == "Conv"]
    if (
        config["node_selection"] != "paired_squeeze_excitation_convolutions"
        or len(selected) != config["expected_converted_conv_count"]
        or len(candidates) != config["total_candidate_conv_count"]
    ):
        raise ValueError("FP16 node selection does not match the recorded graph decision.")
    blocked = [node.name for node in source_model.graph.node if node.name not in selected]
    model = convert_float_to_float16(
        source_model,
        keep_io_types=config["keep_io_types"],
        op_block_list=config["blocked_ops"],
        node_block_list=blocked,
        min_positive_val=1e-30,
        max_finite_val=65504,
    )
    tensor_types = {
        v.name: v.type.tensor_type.elem_type
        for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    }
    fp16_weights = [
        t.name for t in model.graph.initializer if t.data_type == onnx.TensorProto.FLOAT16
    ]
    if not fp16_weights:
        raise ValueError("FP16 conversion produced no FP16 weights.")
    retained = [
        {"name": n.name, "op_type": n.op_type}
        for n in model.graph.node
        if any(tensor_types.get(o) == onnx.TensorProto.FLOAT for o in n.output)
    ]
    unknown = [n.name for n in model.graph.node if not all(o in tensor_types for o in n.output)]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending.onnx")
    onnx.save(model, temporary)
    return save_export(
        path,
        temporary,
        fp32,
        "fp16",
        identity,
        {
            "conversion": config,
            "converter_version": onnxconverter_common.__version__,
            "fp16_weight_count": len(fp16_weights),
            "converted_conv_nodes": selected,
            "converted_conv_count": len(selected),
            "total_candidate_conv_count": len(candidates),
            "execution_precision": config["execution_precision"],
            "fp32_output_nodes": retained,
            "nodes_with_unknown_output_type": unknown,
            "numeric_conversion_bounds": {"min_positive_val": 1e-30, "max_finite_val": 65504},
            "scope": "graph_tensor_types_not_runtime_kernel_placement",
            "shared_output_isolation": True,
        },
    )
