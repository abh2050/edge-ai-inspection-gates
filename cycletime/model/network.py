"""The model distills fixed MobileNet feature stages and refuses a memory bank."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small


class Detector(nn.Module):
    """Compare student and frozen teacher features; do not update normalization during inference."""

    def __init__(self, stages=(3, 8, 12), size=(256, 256)):
        super().__init__()
        if tuple(stages) != (3, 8, 12):
            raise ValueError("The ADR requires stages 3, 8, and 12.")
        self.stages = tuple(stages)
        self.size = tuple(size)
        self.teacher = mobilenet_v3_small(weights=None).features
        self.student = mobilenet_v3_small(weights=None).features
        self.teacher.requires_grad_(False)
        self.register_buffer("scales", torch.ones(len(stages)))

    def features(self, network, x):
        """Return configured feature tensors; refuse architecture changes through construction checks."""
        result = []
        for index, layer in enumerate(network):
            x = layer(x)
            if index in self.stages:
                result.append(x)
        return result

    def distances(self, x):
        """Compute channel-normalized squared distances; never train the teacher."""
        self.teacher.eval()
        with torch.no_grad():
            teacher = self.features(self.teacher, x)
        student = self.features(self.student, x)
        return [
            (F.normalize(t, dim=1, eps=1e-6) - F.normalize(s, dim=1, eps=1e-6))
            .square()
            .sum(1, keepdim=True)
            for t, s in zip(teacher, student, strict=True)
        ]

    def forward(self, x):
        """Return the mean resized distance map and its maximum; never fit on the input."""
        maps = [
            F.interpolate(d / self.scales[i], size=self.size, mode="bilinear", align_corners=False)
            for i, d in enumerate(self.distances(x))
        ]
        anomaly_map = torch.stack(maps).mean(0)
        return anomaly_map.flatten(1).amax(1), anomaly_map
