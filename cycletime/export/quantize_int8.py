"""This module performs static INT8 QDQ quantization from verified fit images."""

from __future__ import annotations

from pathlib import Path

import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from cycletime.contracts import JSON, Artifact, EvidenceRef
from cycletime.dataio.archive import sha256
from cycletime.dataio.calibration import calibration_reader, validate_partitions
from cycletime.evidence import digest
from cycletime.export.metadata import int8_coverage, load_artifact, reuse, save_export


def export_int8(fp32: Artifact, partitions: EvidenceRef, export_config: JSON) -> Artifact:
    """Quantize with static signed QDQ and verified fit calibration; refuse test calibration, dynamic substitution, or concealed coverage."""
    source, _ = load_artifact(fp32.path)
    if source != fp32 or source.precision != "fp32":
        raise ValueError("INT8 conversion requires the verified canonical FP32 artifact.")
    parts = validate_partitions(partitions)
    config = export_config["int8"]
    if (
        config["method"] != "post_training_static"
        or config["format"] != "QDQ"
        or config["calibration_split"] != "training_fit_partition"
        or config["calibration_method"] != "MinMax"
        or config["activation_type"] != "QInt8"
        or config["weight_type"] != "QInt8"
    ):
        raise ValueError("The declared static INT8 configuration is unsupported.")
    if parts["dataset_config"]["_dataset_sha256"] != fp32.dataset_sha256:
        raise ValueError("Calibration dataset differs from the export provenance.")
    reader = calibration_reader(partitions, config["calibration_sample_count"])
    path = fp32.path.parents[2] / config["path"]
    identity = digest(
        {
            "source": fp32.sha256,
            "config": config,
            "partitions": partitions.sha256,
            "ort": ort.__version__,
            "code": sha256(Path(__file__)),
            "metadata_code": sha256(Path(__file__).with_name("metadata.py")),
        }
    )
    if cached := reuse(path, identity):
        return cached
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared = path.with_name("fp32.quantization-input.onnx")
    quant_pre_process(
        fp32.path,
        prepared,
        skip_symbolic_shape=True,
        skip_optimization=False,
        save_as_external_data=False,
    )
    prepared_model = onnx.load(prepared)
    candidates = [node.name for node in prepared_model.graph.node if node.op_type == "Conv"]
    selected = [
        name for name in candidates if config["selected_node_substring"] in name
    ]
    if (
        config["node_selection"] != "paired_squeeze_excitation_convolutions"
        or len(selected) != config["expected_quantized_conv_count"]
        or len(candidates) != config["total_candidate_conv_count"]
    ):
        raise ValueError("INT8 node selection does not match the recorded graph decision.")
    temporary = path.with_suffix(".pending.onnx")
    quantize_static(
        prepared,
        temporary,
        reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=config["quantize_ops"],
        nodes_to_quantize=selected,
        per_channel=config["per_channel"],
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        calibration_providers=["CPUExecutionProvider"],
        use_external_data_format=False,
    )
    model = onnx.load(temporary)
    coverage = int8_coverage(model)
    if coverage["qdq_compute_count"] != len(selected):
        raise ValueError("Serialized INT8 coverage differs from the selected node set.")
    ids = parts["fit"][: config["calibration_sample_count"]]
    return save_export(
        path,
        temporary,
        fp32,
        "int8",
        identity,
        {
            "quantization": config,
            "onnxruntime_version": ort.__version__,
            "coverage": coverage,
            "selected_conv_nodes": selected,
            "selected_conv_count": len(selected),
            "total_candidate_conv_count": len(candidates),
            "execution_precision": config["execution_precision"],
            "calibration_ids": ids,
            "calibration_file_hashes": {n: parts["file_hashes"][n] for n in ids},
            "partitions_sha256": partitions.sha256,
            "preprocessed_fp32_sha256": sha256(prepared),
            "preprocessing": {
                "symbolic_shape_inference": False,
                "onnx_shape_inference": True,
                "ort_optimization": "basic",
            },
        },
    )
