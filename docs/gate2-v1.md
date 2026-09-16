# Gate 2 records precision conversion results.

Gate 2 failed its configured checks.

| Precision | Parity | Image AUROC | Pixel AUROC | Recall | Status |
|---|---|---:|---:|---:|---|
| fp32 | passed_in_gate1 | 0.981746 | 0.959094 | 0.904762 | passed |
| fp16 | failed | 0.981746 | 0.959088 | 0.904762 | failed |
| int8 | failed | 0.800000 | 0.768719 | 1.000000 | failed |

All configurations use the threshold frozen before the FP32 test evaluation.
Gate 2 reuses FP32 accuracy and records one complete evaluation for each supported new artifact.
The INT8 calibration uses 64 distinct images from the normal training fit partition.
Parity compares both outputs on 16 held-out normal validation images.
The conversion metadata records graph precision coverage; it does not establish CPU or Neural Engine kernel placement.
This gate does not measure latency.

The fp16 checks reported: Output parity failed. Evidence: /Users/abhishekshah/Desktop/edge/cycletime-inspect/artifacts/parity/fp16.json
The int8 checks reported: Output parity failed. Evidence: /Users/abhishekshah/Desktop/edge/cycletime-inspect/artifacts/parity/int8.json; Gate 1 quality floor failed: image_auroc=0.7999999999999999 (required >= 0.9); pixel_auroc=0.7687190612903132 (required >= 0.9)

The [failure analysis](gate2-failures.md) records the numerical violations and the INT8 rejection behavior.
