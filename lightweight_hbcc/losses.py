from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def smooth_one_hot(target: torch.Tensor, num_classes: int, smoothing: float = 0.0) -> torch.Tensor:
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    out = torch.full((target.shape[0], num_classes), off_value, device=target.device, dtype=torch.float32)
    out.scatter_(1, target.reshape(-1, 1), on_value)
    return out


def soft_target_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.label_smoothing = float(label_smoothing)

    def forward(self, student_logits: torch.Tensor, target: torch.Tensor, teacher_logits: torch.Tensor | None = None) -> torch.Tensor:
        if target.ndim == 1:
            ce = F.cross_entropy(student_logits, target, label_smoothing=self.label_smoothing)
        else:
            ce = soft_target_cross_entropy(student_logits, target)
        if teacher_logits is None or self.alpha <= 0.0:
            return ce
        temp = self.temperature
        kd = F.kl_div(
            F.log_softmax(student_logits / temp, dim=1),
            F.softmax(teacher_logits / temp, dim=1),
            reduction="batchmean",
        ) * (temp * temp)
        return (1.0 - self.alpha) * ce + self.alpha * kd
