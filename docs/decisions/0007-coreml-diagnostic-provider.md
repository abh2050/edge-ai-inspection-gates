# ADR 0007 makes CoreML a diagnostic provider for Gate 3.

Status: accepted by the project owner on 2026-09-15 and applied to config/bench.yaml and the Gate 3 code.

Gate 3 measured every artifact on the CPU execution provider within the research cycle allowance.
The measured CPU p99 values range from 14.474 to 15.448 milliseconds against a 1333.333 millisecond allowance.
The CPU rows therefore establish research latency feasibility without any accelerator.

ADR 0006 records that CoreML executed every workload on the CPU and that Neural Engine execution cannot pass parity for this model.
A diagnostic investigation showed that a revised export removes ORT CPU fallback for the FP32 and FP16 artifacts.
ONNX Runtime 1.30 rejects every QuantizeLinear and DequantizeLinear node in the CoreML execution provider for this graph.
Per-channel and per-tensor quantization with signed and unsigned types all placed the quantized convolutions on the ORT CPU provider.
The INT8 and CoreML pair therefore cannot satisfy the fallback policy without a change of precision or format, which the repository forbids.

Each provider declares whether Gate 3 requires it.
The CPU execution provider remains required for every precision.
The CoreML execution provider becomes diagnostic for every precision.
Gate 3 still attempts, measures, parity-checks, and labels every diagnostic pair.
Gate 3 records every diagnostic failure in the matrix and gate records and in the latency report.
Only failures of required pairs determine the Gate 3 status.
No report may cite a diagnostic CoreML row as qualified latency evidence.
Gate 4 sustained testing would cover required pairs, and any diagnostic sustained run would carry the same label.

This decision does not change tolerances, precisions, placement labels, or the ORT CPU fallback policy.
This decision does not create exports or consume evaluations.
A future accelerator requirement must restore CoreML as a required provider through a new decision.
The Gate 3 record before this decision appears in artifacts/gate3_v3.json, artifacts/bench/matrix_v3.json, and docs/latency-v3.md.
