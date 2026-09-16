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
| FP32 | FP32 | 13.119 | 14.461 | 14.731 | 4073.04 | 0.981746 | 0.959094 | 0.904762 | yes | CPUExecutionProvider | 2026-09-15T20:49:08.379386+00:00 |
| FP16 | mixed FP16/FP32, 36/104 convolution nodes | 13.835 | 15.278 | 15.448 | 3884.00 | 0.980952 | 0.959086 | 0.904762 | yes | CPUExecutionProvider | 2026-09-15T20:49:16.868612+00:00 |
| INT8 | mixed INT8/FP32, 36/104 convolution nodes | 13.994 | 14.387 | 14.474 | 4145.46 | 0.980159 | 0.958930 | 0.904762 | yes | CPUExecutionProvider | 2026-09-15T20:49:25.084198+00:00 |

## coreml_cpu_ane

| Artifact precision | Serialized execution precision | p50 ms | p95 ms | p99 ms | Capacity from p99, parts/min | Image AUROC | Pixel AUROC | Recall | Clears 45 parts/min | Placement | Timestamp UTC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| FP32 | FP32 | 9.424 | 10.562 | 10.765 | 5573.43 | 0.981746 | 0.959094 | 0.904762 | yes | CoreML mixed placement; ORT CPU nodes observed; ANE telemetry measured 0 mW | 2026-09-15T20:49:35.637329+00:00 |
| FP16 | mixed FP16/FP32, 36/104 convolution nodes | 9.549 | 10.958 | 11.109 | 5401.20 | 0.980952 | 0.959086 | 0.904762 | yes | CoreML mixed placement; ORT CPU nodes observed; ANE telemetry measured 0 mW | 2026-09-15T20:49:46.346894+00:00 |
| INT8 | mixed INT8/FP32, 36/104 convolution nodes | 11.778 | 12.306 | 12.706 | 4722.29 | 0.980159 | 0.958930 | 0.904762 | yes | CoreML mixed placement; ORT CPU nodes observed; ANE telemetry measured 0 mW | 2026-09-15T20:49:59.363610+00:00 |

CoreML provider events and CPU provider events come from ONNX Runtime profiling.
CPU-to-provider output parity passed for every measured pair on sixteen held-out normal validation images.
The requested CPUAndNeuralEngine option does not prove that CoreML used the Neural Engine.
Administrator-backed telemetry measured 0.0 mW of ANE power throughout each CoreML workload.
Gate 3 remains failed because ORT placed operators outside CoreML, which the provider policy forbids.
The report does not claim commercial cycle feasibility because acquisition, transport, and actuation overhead remain unmeasured.
