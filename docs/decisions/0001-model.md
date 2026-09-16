# ADR 0001 selects convolutional feature distillation.

The scaffold accepts a student and teacher that use MobileNet V3 Small.
The teacher remains frozen after initialization from separately acquired pretrained weights.
The student learns the teacher's intermediate features from normal training images.
The inference graph runs both networks and converts their normalized feature differences into an anomaly map.
The graph takes the maximum map value as the image score.
The implementation must verify that the configured feature stages expose the intended spatial resolutions.

The line permits 60000 / 45 = 1333.333... milliseconds per part.
The repository must measure the combined networks, preprocessing, and scoring against that allowance.
This ADR makes an architectural choice without claiming that the model already satisfies that budget.

A PatchCore-style memory bank requires a nearest neighbor search during inference.
Quantizing its convolutional extractor does not remove that search or its memory traffic.
The selected model concentrates its inference work in convolutional networks with fixed input shapes.
The selected graph therefore gives post-training quantization a direct target.
Quantization may still hurt anomaly scores or leave operators in floating point.
The parity and quality gates must establish whether each conversion remains usable.

The first implementation uses native PyTorch and torchvision.
The scaffold does not add anomalib because a direct feature-distillation wrapper keeps the export boundary explicit.
An implementation may adopt anomalib only after a superseding ADR identifies the selected model, graph boundaries, and dependency tradeoffs.
The implementer must record the pretrained teacher's source, hash, and applicable rights before training.
The repository must not distribute those weights.

The input resize may erase small defects.
The implementer must preserve original mask geometry when reporting pixel metrics and document how maps return to that geometry.
The implementer must never resize masks with bilinear interpolation.
An error analysis must identify defect sizes that the resized representation misses.

Gate 1 must evaluate the canonical FP32 export against the configured AUROC floors.
Gate 2 must establish parity and accuracy for FP16 and INT8.
Gate 3 must determine support on the actual CPU and CoreML configurations.
Gate 4 must determine whether the measured budget survives the sustained workload.
A failed gate requires a documented change or an explicit failure report.
The agent must not silently switch models or relax tolerances.
