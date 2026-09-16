"""Compare reduced-precision experiments on held-out normal validation images only."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common.float16 import convert_float_to_float16
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from cycletime.config import load_config
from cycletime.dataio.images import load_image
from cycletime.evidence import read, reference, restore
from cycletime.export.to_fp16 import isolate_shared_outputs
from cycletime.export.verify_parity import compare_outputs

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "artifacts/experiments/gate2"


def session(path: Path) -> ort.InferenceSession:
    """Open a CPU session with fixed thread settings."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def summary(
    reference: ort.InferenceSession,
    candidate: ort.InferenceSession,
    inputs: list[np.ndarray],
    bounds: dict,
) -> dict:
    """Summarize exact elementwise parity without accessing test data."""
    rows = []
    for value in inputs:
        expected = reference.run(["image_score", "anomaly_map"], {"image": value})
        actual = candidate.run(["image_score", "anomaly_map"], {"image": value})
        rows.extend(compare_outputs(expected, actual, bounds))
    return {
        name: {
            "violations": sum(row["violations"] for row in rows if row["output"] == name),
            "elements": sum(row["element_count"] for row in rows if row["output"] == name),
            "maximum_absolute_error": max(
                row["maximum_absolute_error"] for row in rows if row["output"] == name
            ),
        }
        for name in ("image_score", "anomaly_map")
    }


class Reader(CalibrationDataReader):
    """Feed an immutable list of distinct normal fit tensors."""

    def __init__(self, values: list[np.ndarray]):
        self.values = values
        self.index = 0

    def get_next(self):
        if self.index == len(self.values):
            return None
        value = self.values[self.index]
        self.index += 1
        return {"image": value}


def main() -> None:
    """Write diagnostic models and a result table; refuse any test-split input."""
    config = load_config(ROOT / "config/export.yaml")
    dataset = load_config(ROOT / "config/dataset.yaml")
    gate0 = read(reference(ROOT / "artifacts/gate0.json"))
    dataset.update(_project=str(ROOT), _dataset_sha256=gate0["dataset_sha256"])
    gate1 = read(reference(ROOT / "artifacts/gate1.json"))
    partitions = read(restore(gate1["partitions"]))
    if any("/test/" in name for name in partitions["fit"] + partitions["validation"]):
        raise ValueError("Diagnostics may not read test images.")
    root = ROOT / dataset["root"]
    validation = [load_image(root / name, dataset)[None] for name in partitions["validation"][:16]]
    calibration = [load_image(root / name, dataset)[None] for name in partitions["fit"][:64]]
    fp32_path = ROOT / config["fp32"]["path"]
    fp32_session = session(fp32_path)
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    results = []

    for minimum in (1e-7, 1e-12, 1e-30):
        name = f"fp16_min_{minimum:.0e}".replace("-", "m")
        path = EXPERIMENTS / f"{name}.onnx"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = convert_float_to_float16(
                isolate_shared_outputs(onnx.load(fp32_path)),
                keep_io_types=True,
                op_block_list=[],
                min_positive_val=minimum,
                max_finite_val=65504,
            )
        onnx.checker.check_model(model, full_check=True)
        onnx.save(model, path)
        result = {"name": name, "precision": "fp16", "min_positive_val": minimum}
        result["parity"] = summary(
            fp32_session, session(path), validation, config["parity"]["fp16_vs_fp32"]
        )
        results.append(result)
        print(json.dumps(result), flush=True)

    prepared = ROOT / "artifacts/exports/fp32.quantization-input.onnx"
    conv_names = [node.name for node in onnx.load(prepared).graph.node if node.op_type == "Conv"]
    teacher = conv_names[:52]
    student = conv_names[52:]
    variants = {
        "int8_first_pair": [teacher[0], student[0]],
        "int8_last_pair": [teacher[-1], student[-1]],
        "int8_stage3_pair": [teacher[10], student[10]],
        "int8_first_4_pairs": teacher[:4] + student[:4],
        "int8_last_4_pairs": teacher[-4:] + student[-4:],
        "int8_se_pairs": [name for name in conv_names if "/fc" in name],
    }
    for name, nodes in variants.items():
        path = EXPERIMENTS / f"{name}.onnx"
        quantize_static(
            prepared,
            path,
            Reader(calibration),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            nodes_to_quantize=nodes,
            calibrate_method=CalibrationMethod.MinMax,
            calibration_providers=["CPUExecutionProvider"],
        )
        result = {
            "name": name,
            "precision": "int8",
            "quantized_conv_count": len(nodes),
            "nodes": nodes,
        }
        result["parity"] = summary(
            fp32_session, session(path), validation, config["parity"]["int8_vs_fp32"]
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    (EXPERIMENTS / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
