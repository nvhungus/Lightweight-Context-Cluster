from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def num_classes_for_dataset(name: str) -> int:
    name = name.lower()
    if name == "cifar10":
        return 10
    if name == "cifar100":
        return 100
    if name == "fake":
        return 10
    raise ValueError(f"Unsupported dataset: {name}")


def _transforms(name: str, train: bool, augment: bool) -> transforms.Compose:
    if name.lower() == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    ops: list[Any] = []
    if train and augment:
        ops.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(ops)


def build_datasets(cfg: dict[str, Any]) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    name = str(cfg.get("name", "cifar10")).lower()
    root = Path(cfg.get("root", "data"))
    download = bool(cfg.get("download", True))
    augment = bool(cfg.get("augment", True))
    if name == "cifar10":
        train = datasets.CIFAR10(root=root, train=True, transform=_transforms(name, True, augment), download=download)
        val = datasets.CIFAR10(root=root, train=False, transform=_transforms(name, False, False), download=download)
    elif name == "cifar100":
        train = datasets.CIFAR100(root=root, train=True, transform=_transforms(name, True, augment), download=download)
        val = datasets.CIFAR100(root=root, train=False, transform=_transforms(name, False, False), download=download)
    elif name == "fake":
        transform = _transforms("cifar10", False, False)
        train = datasets.FakeData(size=int(cfg.get("fake_train_size", 512)), image_size=(3, 32, 32), num_classes=10, transform=transform)
        val = datasets.FakeData(size=int(cfg.get("fake_val_size", 128)), image_size=(3, 32, 32), num_classes=10, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    train_limit = cfg.get("train_limit")
    val_limit = cfg.get("val_limit")
    if train_limit:
        train = Subset(train, range(min(int(train_limit), len(train))))
    if val_limit:
        val = Subset(val, range(min(int(val_limit), len(val))))
    return train, val


def build_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    train_set, val_set = build_datasets(cfg)
    batch_size = int(cfg.get("batch_size", 128))
    val_batch_size = int(cfg.get("val_batch_size", batch_size))
    workers = int(cfg.get("workers", 2))
    pin_memory = bool(cfg.get("pin_memory", True))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=bool(cfg.get("drop_last", True)),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
