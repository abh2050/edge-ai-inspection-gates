# ADR 0005 limits reduced precision to paired squeeze-and-excitation convolutions.

Gate 2 first converted every eligible convolution in both feature extractors.
The full FP16 graph preserved aggregate accuracy but violated the configured anomaly-map parity tolerance.
The full INT8 graph violated parity and reduced image and pixel AUROC below their configured floors.

The diagnostic run used sixteen held-out normal validation images.
The run did not inspect the MVTec test split or its cached predictions.
The run compared several fixed node groups under the original parity tolerances.
Quantizing the early spatial convolutions caused the largest errors in the feature-distance output.
Converting paired squeeze-and-excitation convolutions preserved both declared outputs within tolerance.

The revised FP16 and INT8 artifacts convert the same thirty-six convolution nodes across the teacher and student.
The graph contains 104 candidate convolution nodes.
The revised artifacts therefore use mixed numeric precision.
Their metadata records every converted node, every retained node, and the 36-of-104 coverage count.
Reports must use the labels mixed FP16/FP32 and mixed INT8/FP32 when they discuss execution precision.
The short precision field remains fp16 or int8 because the repository contract and result table use those artifact classes.

This decision preserves the configured parity tolerances and quality floors.
It does not claim that partial conversion improves latency.
Gate 3 must measure the revised artifacts on every required provider.
The original failed artifacts and their sole evaluations remain preserved as version-one evidence.
