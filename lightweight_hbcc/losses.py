from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, student_logits: torch.Tensor, target: torch.Tensor, teacher_logits: torch.Tensor | None = None) -> torch.Tensor:
        ce = self.ce(student_logits, target)
        if teacher_logits is None or self.alpha <= 0.0:
            return ce
        temp = self.temperature
        kd = F.kl_div(
            F.log_softmax(student_logits / temp, dim=1),
            F.softmax(teacher_logits / temp, dim=1),
            reduction="batchmean",
        ) * (temp * temp)
        return (1.0 - self.alpha) * ce + self.alpha * kd
