# Gate 1 records the FP32 model evaluation.

Gate 1 passed its configured AUROC floors.

| Metric | Measured value | Required floor |
|---|---:|---:|
| Image AUROC | 0.981746 | 0.90 |
| Pixel AUROC | 0.959094 | 0.90 |
| Recall at frozen threshold | 0.904762 | Gate 5 enforces the recall constraint. |

The held-out normal threshold is 2.16133960.
The evaluation used 83 test images and 67230000 original-resolution pixels.
The ledger records one completed evaluation for this artifact.
The pixel metric resizes score maps bilinearly to the original geometry and does not resize ground truth.
The 256-pixel model input can lose small defects; the following counts describe misses at the frozen threshold.

| Test subtype | Images | Rejected | Defect pixels, min–max |
|---|---:|---:|---:|
| broken_large | 20 | 20 | 33132–222091 |
| broken_small | 22 | 22 | 7166–65292 |
| contamination | 21 | 15 | 4663–133876 |
| good | 20 | 2 | 0–0 |

The artifact SHA256 is `880c32b18b41b0b40b1f08a9ca7fe2944769ab949130ebd583f08ca489038c0c`.
The evidence references are recorded in artifacts/gate1.json.
This gate records accuracy and does not establish latency or commercial readiness.

The [error analysis](gate1-error-analysis.md) lists the missed defects and their original mask areas.
