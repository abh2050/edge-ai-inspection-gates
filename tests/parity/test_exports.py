"""These tests check output parity, precision coverage, and shared-output conversion."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto as T
from onnx import helper as h
from onnxconverter_common.float16 import convert_float_to_float16

from cycletime.export.metadata import int8_coverage
from cycletime.export.to_fp16 import isolate_shared_outputs
from cycletime.export.verify_parity import compare_outputs


def test_compare_all_outputs():
    expected = [np.ones(1), np.ones((1, 1, 2, 2))]
    actual = [np.ones(1), np.ones((1, 1, 2, 2)) * 1.2]
    rows = compare_outputs(expected, actual, {"atol": 0.01, "rtol": 0.01})
    assert rows[0]["passed"] and not rows[1]["passed"]
    assert rows[1]["violations"] == 4
    with pytest.raises(ValueError, match="Both"):
        compare_outputs(expected, actual[:1], {"atol": 0.01, "rtol": 0.01})


@pytest.mark.parametrize("bad", [np.array([np.nan]), np.array([np.inf]), np.ones(2)])
def test_reject_nonfinite_or_wrong_shape(bad):
    with pytest.raises(ValueError, match="Invalid"):
        compare_outputs([np.ones(1), np.ones(1)], [bad, np.ones(1)], {"atol": 0.01, "rtol": 0.01})


def test_tolerance_uses_reference_magnitude():
    rows = compare_outputs([np.ones(1)] * 2, [np.ones(1) * 2] * 2, {"atol": 0, "rtol": 0.6})
    assert not any(row["passed"] for row in rows)


def test_shared_output_fp16_keeps_internal_types():
    graph = h.make_graph(
        [
            h.make_node("Mul", ["image", "weight"], ["anomaly_map"]),
            h.make_node("Flatten", ["anomaly_map"], ["flat"], axis=1),
            h.make_node("ReduceMax", ["flat", "axis"], ["image_score"], keepdims=0),
        ],
        "shared",
        [h.make_tensor_value_info("image", T.FLOAT, [1, 1, 2, 2])],
        [
            h.make_tensor_value_info("image_score", T.FLOAT, [1]),
            h.make_tensor_value_info("anomaly_map", T.FLOAT, [1, 1, 2, 2]),
        ],
        [h.make_tensor("weight", T.FLOAT, [1], [2]), h.make_tensor("axis", T.INT64, [1], [1])],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=10)
    converted = convert_float_to_float16(
        isolate_shared_outputs(model), keep_io_types=True, op_block_list=[]
    )
    onnx.checker.check_model(converted, full_check=True)
    session = ort.InferenceSession(
        converted.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    outputs = session.run(None, {"image": np.ones((1, 1, 2, 2), np.float32)})
    assert all(value.dtype == np.float32 for value in outputs)
    assert all(np.all(value == 2) for value in outputs)
    assert any(t.data_type == T.FLOAT16 for t in converted.graph.initializer)


def test_mixed_precision_selection_is_explicit():
    """The configuration must state partial graph coverage instead of implying full conversion."""
    import yaml

    with open("config/export.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for precision, count_key in (
        ("fp16", "expected_converted_conv_count"),
        ("int8", "expected_quantized_conv_count"),
    ):
        settings = config[precision]
        assert settings["node_selection"] == "paired_squeeze_excitation_convolutions"
        assert settings[count_key] == 36
        assert settings["total_candidate_conv_count"] == 104
        assert settings["execution_precision"].startswith("mixed_")


def quantized_graph():
    nodes = [
        h.make_node("QuantizeLinear", ["image", "scale", "zero"], ["quantized"]),
        h.make_node("DequantizeLinear", ["quantized", "scale", "zero"], ["activation"]),
        h.make_node("DequantizeLinear", ["weight", "scale", "zero"], ["float_weight"]),
        h.make_node("Conv", ["activation", "float_weight"], ["output"], name="conv"),
    ]
    graph = h.make_graph(
        nodes,
        "qdq",
        [],
        [],
        [
            h.make_tensor("scale", T.FLOAT, [], [0.1]),
            h.make_tensor("zero", T.INT8, [], [0]),
            h.make_tensor("weight", T.INT8, [1, 1, 1, 1], [1]),
        ],
    )
    return h.make_model(graph)


def test_record_quantization_coverage():
    model = quantized_graph()
    coverage = int8_coverage(model)
    assert coverage["qdq_compute_count"] == 1
    assert coverage["int8_weight_tensors"] == ["weight"]
    model.graph.node[-1].input[0] = "image"
    with pytest.raises(ValueError, match="no signed INT8"):
        int8_coverage(model)


def test_dynamic_quantization_is_rejected():
    model = quantized_graph()
    model.graph.node[0].op_type = "DynamicQuantizeLinear"
    with pytest.raises(ValueError, match="Dynamic quantization"):
        int8_coverage(model)


def test_provider_parity_uses_both_declared_outputs():
    """Provider comparison must fail when either declared output violates tolerance."""
    expected = [np.array([1.0]), np.ones((1, 1, 2, 2))]
    changed_score = [np.array([1.2]), np.ones((1, 1, 2, 2))]
    changed_map = [np.array([1.0]), np.ones((1, 1, 2, 2)) * 1.2]
    bounds = {"atol": 0.001, "rtol": 0.01}
    assert not compare_outputs(expected, changed_score, bounds)[0]["passed"]
    assert not compare_outputs(expected, changed_map, bounds)[1]["passed"]
