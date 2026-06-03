from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lightweight_hbcc.config import load_config
from lightweight_hbcc.models import build_model
from lightweight_hbcc.models.layers import sign_ste


def test_hbcc_smoke_forward() -> None:
    cfg = load_config("configs/smoke.yaml")
    model = build_model(cfg)
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_all_reference_models_forward() -> None:
    for path in [
        "configs/baselines/resnet18_cifar.yaml",
        "configs/baselines/mobilenet_v2_cifar.yaml",
        "configs/baselines/shufflenet_v2_x1_0_cifar.yaml",
        "configs/hbcc_latency_tiny.yaml",
        "configs/hbcc_current_reference.yaml",
        "configs/coc_cifar_baseline.yaml",
        "configs/hbcc_accuracy_small.yaml",
        "configs/hbcc_accuracy_medium.yaml",
        "configs/ablations/hbcc_accuracy_small_pruning_mask.yaml",
    ]:
        cfg = load_config(path)
        model = build_model(cfg).eval()
        with torch.no_grad():
            y = model(torch.randn(1, 3, 32, 32))
        assert y.shape == (1, 10), path


def test_official_coc_tiny_imagenet_materializes() -> None:
    official_code = Path("docs/context-cluster/models/context_cluster.py")
    if not official_code.exists():
        pytest.skip(f"Official Context-Cluster code not found: {official_code}")
    cfg = load_config("configs/imagenet/official_coc_tiny_imagenet.yaml")
    model = build_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    assert getattr(model, "num_classes") == 1000
    assert total_params > 5_000_000


def test_hbcc_accuracy_imagenet_configs_forward() -> None:
    for path in [
        "configs/imagenet/hbcc_accuracy_small_imagenet.yaml",
        "configs/imagenet/hbcc_accuracy_medium_imagenet.yaml",
    ]:
        cfg = load_config(path)
        model = build_model(cfg).eval()
        with torch.no_grad():
            y = model(torch.randn(1, 3, 224, 224))
        assert y.shape == (1, 1000), path


def test_sign_ste_backward() -> None:
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
    y = sign_ste(x).sum()
    y.backward()
    assert torch.equal(sign_ste(x).detach(), torch.tensor([-1.0, -1.0, 1.0, 1.0]))
    assert torch.equal(x.grad, torch.tensor([0.0, 1.0, 1.0, 0.0]))
