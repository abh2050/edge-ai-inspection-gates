# ADR 0009 permits one attributed detection example figure.

Status: accepted by the project owner on 2026-09-16.

The repository documentation described the inspection task in words and showed only dashboard screenshots.
A reader could not see what the detector receives or what it produces.
A stock photograph of an unrelated production line was proposed and rejected, because it depicts a different process and carries no evidential value.

The repository therefore renders one figure from the local dataset and the recorded evaluation.
The figure shows one normal part, one defective part, and the anomaly map that gate 1 recorded for the defective part.
Each panel states the recorded image score and the decision the frozen threshold produced.
The script scripts/build_sample_figure.py regenerates the figure from artifacts/gate1.json and the local dataset directory.

Rule 7 previously prohibited shipping MVTec images at all.
Rule 7 now prohibits the archive, bulk images, and trained weights, and permits one documentation figure containing at most three dataset images.
MVTec AD is distributed under CC BY-NC-SA 4.0, which permits redistribution with attribution under noncommercial terms.
The figure therefore carries the dataset name and the licence notice inside the image, and the README repeats that attribution.

The repository still distributes no dataset archive, no bulk images, and no trained weights.
A commercial deployment must use customer-owned data and cannot rely on this figure or the dataset behind it.
The figure is documentation and is never evidence; the gate records remain the only evidence.
