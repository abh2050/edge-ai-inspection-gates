# Gate 3 passes on the required CPU matrix and records CoreML as a diagnostic provider.

The runner measured all six artifact and provider pairs on the local Apple M3 Pro.
Each pair discarded twenty warmups and retained five hundred timed batch-one completions.
The timer covered preprocessing, synchronous inference, and postprocessing on decoded images.
CPU-to-provider output parity passed for every pair on sixteen held-out normal validation images.

ADR 0007 makes the CPU execution provider required and the CoreML execution provider diagnostic.
The required CPU p99 values range from 14.834 to 15.801 milliseconds against the 1333.333 millisecond research allowance.
Every required row satisfies the placement evidence policy.
These measurements do not include camera acquisition, transport, or actuation.

The diagnostic CoreML p99 values range from 11.356 to 12.688 milliseconds.
All three CoreML rows fail the ORT CPU fallback policy, and artifacts/gate3.json records those failures as diagnostic policy errors.
The CoreML rows are not qualified latency evidence.
ADR 0006 records that powermetrics measured 0.0 mW of ANE power, so the CoreML rows make no Neural Engine claim.
ADR 0006 also records that FP16 Neural Engine execution fails anomaly-map parity and that ONNX Runtime 1.30 rejects this INT8 graph in CoreML.

The full measured table appears in docs/latency.md.
The gate record appears in artifacts/gate3.json.
The matrix record appears in artifacts/bench/matrix.json.
Earlier failed records appear as artifacts/gate3_v2.json and artifacts/gate3_v3.json with matching matrix and latency files.
