# ADR 0006 removes the Neural Engine placement requirement for the current model.

Gate 3 measured all six artifact and provider pairs and passed provider output parity.
Gate 3 failed only because administrator-backed powermetrics traces measured 0.0 mW of ANE power for every CoreML workload.
The CoreML rows therefore describe CoreML execution on the CPU, not Neural Engine execution.

A diagnostic investigation on 2026-09-15 examined why CoreML did not select the Neural Engine.
The investigation used held-out normal validation images and did not inspect the MVTec test split.
ONNX Runtime 1.30 rejects HardSwish in the CoreML execution provider, which splits each MobileNet into many CoreML partitions.
The scoring head adds ReduceL2, Expand, and Flatten operators that ONNX Runtime places on its CPU provider.
An export that decomposes HardSwish and rewrites the head produced one CoreML partition with unchanged PyTorch outputs.
CoreML still executed that FP32 graph on the CPU, because CPUAndNeuralEngine and CPUOnly produced identical outputs and latency.
FP16 graphs ran two to three times faster under CPUAndNeuralEngine, which indicates Neural Engine execution without proving it.
Those FP16 graphs violated the configured anomaly-map parity tolerance, even with an FP32 head.
Teacher feature stages carried most of that error, which matches the full-FP16 failure recorded in ADR 0005.
The repository must not widen parity tolerances or change requested precision to obtain Neural Engine placement.

The coreml_cpu_ane provider no longer requires confirmed ANE activity.
The provider keeps its identifier and its requested CPUAndNeuralEngine option so that the recorded telemetry still matches the measured configuration.
The report must continue to attach the powermetrics evidence and label each row with its observed ANE power.
No report, table, or agent output may describe these rows as Neural Engine timings.
The measured p99 values support research feasibility for CoreML on the CPU only.

This decision does not relax the ORT CPU fallback policy in ADR 0004.
The adapter now enforces allow_ort_cpu_fallback, which the configuration already set to false.
CoreML rows with operators placed on the ORT CPU provider therefore still fail Gate 3.
A revised export that removes that fallback requires a new canonical FP32 artifact, new FP16 and INT8 artifacts, and one evaluation for each new artifact hash.

A future model or export may restore the Neural Engine requirement when it passes parity under unchanged tolerances and powermetrics confirms ANE activity.
