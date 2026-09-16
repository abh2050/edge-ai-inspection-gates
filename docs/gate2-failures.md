# The first Gate 2 attempt records conversion failures.

The gate retains the original FP32 model and all configured numerical tolerances.
The evaluator completed one full test pass for each of FP32, FP16, and INT8.
The table summarizes comparisons over sixteen held-out normal validation images.

| Precision | Output | Elements outside tolerance | Elements compared | Maximum absolute error |
|---|---|---:|---:|---:|
| fp16 | image_score | 0 | 16 | 0.00462115 |
| fp16 | anomaly_map | 1406 | 1048576 | 0.02736950 |
| int8 | image_score | 16 | 16 | 0.72348106 |
| int8 | anomaly_map | 998015 | 1048576 | 1.59266275 |

The FP16 image scores meet tolerance, but 1,406 anomaly-map values do not.
The FP16 AUROCs remain above the configured floors; those aggregate metrics do not override elementwise parity.
The FP16 graph contains 211 half-precision weight tensors and two FP32 output-cast nodes.
The converter isolates the shared anomaly-map output before inserting IO casts to keep its internal consumer types consistent.
The INT8 graph contains 104 compute nodes with signed INT8 QDQ boundaries.
The graph retains 221 other nodes for activations, arithmetic, shapes, and scoring.
The INT8 image and pixel AUROCs both fall below 0.90.
At the frozen threshold, INT8 rejects all 83 test images, including all twenty normal images.
Its 100% recall therefore does not describe a usable operating point.

The next experiment should compare intermediate feature outputs on the existing normal validation partition to locate the conversion error.
A changed conversion requires a documented decision and a new artifact hash.
The existing three artifacts must not receive another full test inference pass.
The first attempt blocked Gate 3 until ADR 0005 selected revised mixed-precision artifacts.
The current Gate 2 evidence passes; this document preserves the version-one failure analysis.

The reports in artifacts/parity retain the complete per-image comparisons.
The ledger in artifacts/eval_runs.json retains the sole evaluations and their cache hashes.
