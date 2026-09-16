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
| FP32 | FP32 | 12.647 | 13.131 | 13.334 | 4499.79 | 0.981746 | 0.959094 | 0.904762 | yes | CPUExecutionProvider | 2026-09-15T20:05:20.409462+00:00 |
| FP16 | mixed FP16/FP32, 36/104 convolution nodes | 13.449 | 14.142 | 14.429 | 4158.17 | 0.980952 | 0.959086 | 0.904762 | yes | CPUExecutionProvider | 2026-09-15T20:05:27.923017+00:00 |
| INT8 | mixed INT8/FP32, 36/104 convolution nodes | 12.649 | 13.274 | 15.593 | 3847.89 | 0.980159 | 0.958930 | 0.904762 | yes | CPUExecutionProvider | 2026-09-15T20:05:35.043566+00:00 |

## coreml_cpu_ane

| Artifact precision | Serialized execution precision | p50 ms | p95 ms | p99 ms | Capacity from p99, parts/min | Image AUROC | Pixel AUROC | Recall | Clears 45 parts/min | Placement | Timestamp UTC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| FP32 | FP32 | 9.369 | 9.758 | 10.683 | 5616.18 | 0.981746 | 0.959094 | 0.904762 | yes | CoreML mixed placement; CPU nodes observed; ANE requested but unconfirmed | 2026-09-15T20:05:42.521692+00:00 |
| FP16 | mixed FP16/FP32, 36/104 convolution nodes | 9.657 | 9.911 | 10.150 | 5911.37 | 0.980952 | 0.959086 | 0.904762 | yes | CoreML mixed placement; CPU nodes observed; ANE requested but unconfirmed | 2026-09-15T20:05:50.259949+00:00 |
| INT8 | mixed INT8/FP32, 36/104 convolution nodes | 10.681 | 11.054 | 11.601 | 5172.13 | 0.980159 | 0.958930 | 0.904762 | yes | CoreML mixed placement; CPU nodes observed; ANE requested but unconfirmed | 2026-09-15T20:05:59.217506+00:00 |

CoreML provider events and CPU provider events come from ONNX Runtime profiling.
The requested CPUAndNeuralEngine option does not prove that CoreML used the Neural Engine.
The current run lacks privileged ANE power telemetry, so CoreML placement does not pass the required evidence policy.
Gate 3 therefore remains failed even though every inference pair completed and has measured latency.
The report does not claim commercial cycle feasibility because acquisition, transport, and actuation overhead remain unmeasured.
