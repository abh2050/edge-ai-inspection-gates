# Gate 2 records precision conversion results.

Gate 2 passed its configured checks.

| Precision | Parity | Image AUROC | Pixel AUROC | Recall | Status |
|---|---|---:|---:|---:|---|
| fp32 | passed_in_gate1 | 0.981746 | 0.959094 | 0.904762 | passed |
| fp16 | passed | 0.980952 | 0.959086 | 0.904762 | passed |
| int8 | passed | 0.980159 | 0.958930 | 0.904762 | passed |

The fp16 row represents mixed FP16/FP32 execution in the serialized graph.
The int8 row represents mixed INT8/FP32 execution in the serialized graph.
Each revised graph converts 36 of 104 convolution nodes selected as paired squeeze-and-excitation operations.
All configurations use the threshold frozen before the FP32 test evaluation.
Gate 2 reuses FP32 accuracy and records one complete evaluation for each supported new artifact.
The INT8 calibration uses 64 distinct images from the normal training fit partition.
Parity compares both outputs on 16 held-out normal validation images.
The conversion metadata records graph precision coverage; it does not establish CPU or Neural Engine kernel placement.
ADR 0005 records the validation-only diagnosis and the mixed-precision decision.
This gate does not measure latency.

