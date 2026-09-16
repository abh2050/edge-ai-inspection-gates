# Gate 3 records measured local latency.

The host uses Mac15,6 with Apple M3 Pro and macOS 26.6.2.
ONNX Runtime 1.30.0 used one intra-op thread and one inter-op thread.
Every row measures preprocessing, synchronous inference, and postprocessing on decoded images.
Each row excludes twenty warmup iterations and summarizes five hundred timed batch-one completions.
The exact research cycle allowance is 1333.333333 milliseconds.
The capacity column derives 60000 divided by measured p99 and does not report achieved paced throughput.

## cpu

| Artifact precision | Serialized execution precision | p50 ms | p95 ms | p99 ms | Capacity from p99, parts/min | Image AUROC | Pixel AUROC | Recall | Clears 45 parts/min | Placement | Timestamp UTC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| FP32 | FP32 | 12.719 | 13.368 | 14.627 | 4102.12 | 0.981746 | 0.959094 | 0.904762 | yes | CPUExecutionProvider | 2026-09-15T20:23:04.681365+00:00 |
| FP16 | mixed FP16/FP32, 36/104 convolution nodes | 13.545 | 14.222 | 14.771 | 4061.90 | 0.980952 | 0.959086 | 0.904762 | yes | CPUExecutionProvider | 2026-09-15T20:23:12.950015+00:00 |
| INT8 | mixed INT8/FP32, 36/104 convolution nodes | 12.825 | 13.174 | 13.289 | 4514.88 | 0.980159 | 0.958930 | 0.904762 | yes | CPUExecutionProvider | 2026-09-15T20:23:20.702290+00:00 |

## coreml_cpu_ane

| Artifact precision | Serialized execution precision | p50 ms | p95 ms | p99 ms | Capacity from p99, parts/min | Image AUROC | Pixel AUROC | Recall | Clears 45 parts/min | Placement | Timestamp UTC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| FP32 | FP32 | 9.315 | 9.607 | 9.710 | 6179.09 | 0.981746 | 0.959094 | 0.904762 | yes | CoreML mixed placement; CPU nodes observed; ANE telemetry measured 0 mW | 2026-09-15T20:23:30.276805+00:00 |
| FP16 | mixed FP16/FP32, 36/104 convolution nodes | 9.600 | 9.864 | 10.009 | 5994.90 | 0.980952 | 0.959086 | 0.904762 | yes | CoreML mixed placement; CPU nodes observed; ANE telemetry measured 0 mW | 2026-09-15T20:23:40.136949+00:00 |
| INT8 | mixed INT8/FP32, 36/104 convolution nodes | 10.726 | 11.068 | 11.578 | 5182.29 | 0.980159 | 0.958930 | 0.904762 | yes | CoreML mixed placement; CPU nodes observed; ANE telemetry measured 0 mW | 2026-09-15T20:23:51.537132+00:00 |

CoreML provider events and CPU provider events come from ONNX Runtime profiling.
CPU-to-provider output parity passed for every measured pair on sixteen held-out normal validation images.
The requested CPUAndNeuralEngine option does not prove that CoreML used the Neural Engine.
Administrator-backed telemetry measured 0.0 mW of ANE power throughout each CoreML workload.
Gate 3 remains failed because the traces do not confirm Neural Engine placement.
The report does not claim commercial cycle feasibility because acquisition, transport, and actuation overhead remain unmeasured.
