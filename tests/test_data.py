from __future__ import annotations

import torch
from PIL import Image

from lightweight_hbcc import data


class DummyCIFAR(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool, transform=None, download: bool = False) -> None:
        self.train = train
        self.transform = transform
        self.download = download

    def __len__(self) -> int:
        return 50_000 if self.train else 10_000

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 32, 32), index % 10


def test_cifar10_uses_train_val_test_split(monkeypatch) -> None:
    monkeypatch.setattr(data.datasets, "CIFAR10", DummyCIFAR)

    train, val, test = data.build_datasets(
        {"name": "cifar10", "download": False, "val_size": 5000, "split_seed": 123}
    )

    assert len(train) == 45_000
    assert len(val) == 5_000
    assert len(test) == 10_000
    assert set(train.indices).isdisjoint(set(val.indices))


def test_cifar10_split_is_reproducible(monkeypatch) -> None:
    monkeypatch.setattr(data.datasets, "CIFAR10", DummyCIFAR)

    _, val_a, _ = data.build_datasets({"name": "cifar10", "download": False, "split_seed": 123})
    _, val_b, _ = data.build_datasets({"name": "cifar10", "download": False, "split_seed": 123})
    _, val_c, _ = data.build_datasets({"name": "cifar10", "download": False, "split_seed": 321})

    assert val_a.indices == val_b.indices
    assert val_a.indices != val_c.indices


def test_dataset_builder_can_skip_test_split() -> None:
    train, val, test = data.build_datasets(
        {"name": "fake", "fake_train_size": 8, "fake_val_size": 4},
        include_test=False,
    )

    assert len(train) == 8
    assert len(val) == 4
    assert test is None


def test_imagenet_kaggle_cls_loc_layout_reads_flat_val(tmp_path) -> None:
    root = tmp_path / "imagenet-object-localization-challenge"
    train_class = root / "ILSVRC" / "Data" / "CLS-LOC" / "train" / "n00000001"
    val_dir = root / "ILSVRC" / "Data" / "CLS-LOC" / "val"
    train_class.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color="red").save(train_class / "n00000001_1.JPEG")
    Image.new("RGB", (8, 8), color="blue").save(val_dir / "ILSVRC2012_val_00000001.JPEG")
    (root / "LOC_val_solution.csv").write_text(
        "ImageId,PredictionString\nILSVRC2012_val_00000001,n00000001 0 0 1 1\n",
        encoding="utf-8",
    )

    train, val, test = data.build_datasets(
        {
            "name": "imagenet1k",
            "root": str(root),
            "layout": "kaggle_cls_loc",
            "image_size": 8,
            "crop_pct": 1.0,
        },
        include_test=False,
    )

    assert len(train) == 1
    assert len(val) == 1
    assert test is None
    _, target = val[0]
    assert target == 0
