# Gate 1 records missed defects at the frozen threshold.

The FP32 model missed 6 of 63 defective bottle images at threshold 2.16133960.
All missed images belong to the contamination subtype.
The table measures defect area from the original ground truth masks.
These observations do not establish that resizing caused the misses.

| Image | Score | Original defect pixels | Fraction of image |
|---|---:|---:|---:|
| bottle/test/contamination/002.png | 1.982666 | 14913 | 1.8411% |
| bottle/test/contamination/003.png | 2.089352 | 21317 | 2.6317% |
| bottle/test/contamination/015.png | 2.125976 | 33779 | 4.1702% |
| bottle/test/contamination/018.png | 2.132758 | 4663 | 0.5757% |
| bottle/test/contamination/019.png | 2.121354 | 27013 | 3.3349% |
| bottle/test/contamination/020.png | 2.128083 | 121385 | 14.9858% |

The model rejected two of twenty normal test images.
The initial normal-validation threshold does not meet the configured 95% defect recall target.
Gate 5 may compare thresholds using these cached scores as a retrospective research analysis.
Independent acceptance requires a threshold chosen without using the final test labels.

The prediction manifest SHA256 is `1b0a44a753a066abad8c819f74102f3ea7163a4b1b7d64b2f287bdeb4c23bbe5`.
