# Gate 3 records measured local latency.

The host uses Mac15,6 with Apple M3 Pro and macOS 26.6.2.
ONNX Runtime 1.30.0 used one intra-op thread and one inter-op thread.
Every row measures preprocessing, synchronous inference, and postprocessing on decoded images.
Each row excludes twenty warmup iterations and summarizes five hundred timed batch-one completions.
The exact research cycle allowance is 1333.333333 milliseconds.
The capacity column derives 60000 divided by measured p99 and does not report achieved paced throughput.

## cpu (required)

| Artifact precision | Serialized execution precision | p50 ms | p95 ms | p99 ms | Capacity from p99, parts/min | Image AUROC | Pixel AUROC | Recall | Clears 45 parts/min | Placement | Timestamp UTC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| FP32 | FP32 | 12.378 | 13.648 | 15.437 | 3886.77 | 0.981746 | 0.959094 | 0.904762 | yes | CPUExecutionProvider | 2026-09-16T03:19:30.364912+00:00 |
| FP16 | mixed FP16/FP32, 36/104 convolution nodes | 12.934 | 13.351 | 14.085 | 4259.89 | 0.980952 | 0.959086 | 0.904762 | yes | CPUExecutionProvider | 2026-09-16T03:19:38.241787+00:00 |
| INT8 | mixed INT8/FP32, 36/104 convolution nodes | 12.086 | 12.505 | 12.831 | 4676.06 | 0.980159 | 0.958930 | 0.904762 | yes | CPUExecutionProvider | 2026-09-16T03:19:45.635069+00:00 |

## coreml_cpu_ane (diagnostic, not decisive under ADR 0007)

| Artifact precision | Serialized execution precision | p50 ms | p95 ms | p99 ms | Capacity from p99, parts/min | Image AUROC | Pixel AUROC | Recall | Clears 45 parts/min | Placement | Timestamp UTC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| FP32 | FP32 | 9.031 | 9.463 | 10.495 | 5716.97 | 0.981746 | 0.959094 | 0.904762 | yes | CoreML mixed placement; ORT CPU nodes observed; ANE telemetry measured 0 mW | 2026-09-16T03:19:55.734882+00:00 |
| FP16 | mixed FP16/FP32, 36/104 convolution nodes | 9.293 | 9.712 | 9.915 | 6051.20 | 0.980952 | 0.959086 | 0.904762 | yes | CoreML mixed placement; ORT CPU nodes observed; ANE telemetry measured 0 mW | 2026-09-16T03:20:06.117832+00:00 |
| INT8 | mixed INT8/FP32, 36/104 convolution nodes | 10.284 | 10.744 | 10.941 | 5483.93 | 0.980159 | 0.958930 | 0.904762 | yes | CoreML mixed placement; ORT CPU nodes observed; ANE telemetry measured 0 mW | 2026-09-16T03:20:18.182500+00:00 |

CoreML provider events and CPU provider events come from ONNX Runtime profiling.
CPU-to-provider output parity passed for every measured pair on sixteen held-out normal validation images.
The requested CPUAndNeuralEngine option does not prove that CoreML used the Neural Engine.
Administrator-backed telemetry measured 0.0 mW of ANE power throughout each CoreML workload.
ADR 0006 does not require Neural Engine placement, so CoreML rows make no ANE claim.
Every required row satisfies the placement evidence policy.
Under ADR 0007, 3 of 3 diagnostic rows failed placement policy; those rows do not decide Gate 3 and are not qualified latency evidence.
The report does not claim commercial cycle feasibility because acquisition, transport, and actuation overhead remain unmeasured.
