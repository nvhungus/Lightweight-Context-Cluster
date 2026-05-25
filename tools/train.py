from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lightweight_hbcc.config import apply_overrides, load_config, save_config
from lightweight_hbcc.data import build_loaders, num_classes_for_dataset
from lightweight_hbcc.engine import evaluate, load_checkpoint, resolve_device, save_checkpoint, train_one_epoch, write_metrics
from lightweight_hbcc.losses import DistillationLoss
from lightweight_hbcc.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR HBCC/CoC models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="runs")
    parser.add_argument("--resume")
    parser.add_argument("--teacher-config")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    dataset_name = cfg.get("data", {}).get("name", "cifar10")
    cfg.setdefault("model", {})["num_classes"] = cfg["model"].get("num_classes", num_classes_for_dataset(dataset_name))
    device = resolve_device(args.device)
    output_dir = Path(args.output) / cfg.get("experiment", {}).get("name", Path(args.config).stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    train_loader, val_loader = build_loaders(cfg.get("data", {}))
    model = build_model(cfg).to(device)
    if args.resume:
        load_checkpoint(model, args.resume, device, strict=False)

    teacher = None
    if args.teacher_config and args.teacher_checkpoint:
        teacher_cfg = load_config(args.teacher_config)
        teacher_cfg.setdefault("model", {})["num_classes"] = cfg["model"]["num_classes"]
        teacher = build_model(teacher_cfg).to(device)
        load_checkpoint(teacher, args.teacher_checkpoint, device, strict=True)
        teacher.eval()

    train_cfg = cfg.get("train", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    epochs = int(train_cfg.get("epochs", 200))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = DistillationLoss(
        temperature=float(train_cfg.get("kd_temperature", 4.0)),
        alpha=float(train_cfg.get("kd_alpha", 0.0 if teacher is None else 0.5)),
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
    )
    amp = bool(train_cfg.get("amp", True))
    pruning_reg = None
    if cfg.get("model", {}).get("pruning_mask"):
        pruning_reg = (
            float(train_cfg.get("pruning_sparsity_weight", 0.0)),
            float(train_cfg.get("pruning_crispness_weight", 0.0)),
        )

    best_acc = 0.0
    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            epoch,
            teacher=teacher,
            amp=amp,
            limit_batches=args.limit_train_batches,
            pruning_regularization=pruning_reg,
        )
        val_metrics = evaluate(model, val_loader, device, amp=amp, limit_batches=args.limit_val_batches)
        scheduler.step()
        record = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], **train_metrics, **val_metrics}
        write_metrics(record, output_dir / "metrics.jsonl")
        is_best = val_metrics["val_acc1"] >= best_acc
        best_acc = max(best_acc, val_metrics["val_acc1"])
        save_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc1": best_acc,
                "config": cfg,
            },
            output_dir / ("best.pth" if is_best else "latest.pth"),
        )
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
