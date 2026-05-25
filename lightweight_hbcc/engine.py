from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .losses import DistillationLoss
from .metrics import AverageMeter, accuracy


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device, strict: bool = True) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=strict)
    return ckpt


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: DistillationLoss,
    epoch: int,
    teacher: nn.Module | None = None,
    amp: bool = True,
    limit_batches: int | None = None,
    pruning_regularization: tuple[float, float] | None = None,
    progress: bool = True,
) -> dict[str, float]:
    model.train()
    if teacher is not None:
        teacher.eval()
    scaler = GradScaler(device.type, enabled=amp and device.type == "cuda")
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    start = time.perf_counter()
    pbar = tqdm(loader, desc=f"train {epoch}", leave=False, disable=not progress, dynamic_ncols=True, mininterval=0.5)
    for step, (images, target) in enumerate(pbar):
        if limit_batches is not None and step >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits = teacher(images) if teacher is not None else None
        with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(images)
            loss = criterion(output, target, teacher_logits)
            if pruning_regularization and hasattr(model, "pruning_regularization"):
                loss = loss + model.pruning_regularization(*pruning_regularization)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        acc1 = accuracy(output.detach(), target, (1,))[0].item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1, images.size(0))
        if progress:
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc1=f"{acc_meter.avg:.2f}")
    return {"train_loss": loss_meter.avg, "train_acc1": acc_meter.avg, "train_time_s": time.perf_counter() - start}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool = True,
    limit_batches: int | None = None,
    progress: bool = True,
) -> dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    start = time.perf_counter()
    for step, (images, target) in enumerate(
        tqdm(loader, desc="eval", leave=False, disable=not progress, dynamic_ncols=True, mininterval=0.5)
    ):
        if limit_batches is not None and step >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(images)
            loss = loss_fn(output, target)
        acc1 = accuracy(output, target, (1,))[0].item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1, images.size(0))
    return {"val_loss": loss_meter.avg, "val_acc1": acc_meter.avg, "val_time_s": time.perf_counter() - start}


def write_metrics(record: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
