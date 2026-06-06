from __future__ import annotations

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


def test_stl10_configs_forward() -> None:
    for path in [
        "configs/stl10/resnet18_reference_stl10.yaml",
        "configs/stl10/hbcc_small_no_mix_stl10.yaml",
        "configs/stl10/hbcc_small_light_aug_stl10.yaml",
        "configs/stl10/hbcc_local_heavy_stl10.yaml",
    ]:
        cfg = load_config(path)
        model = build_model(cfg).eval()
        with torch.no_grad():
            y = model(torch.randn(1, 3, 96, 96))
        assert y.shape == (1, 10), path


def test_sign_ste_backward() -> None:
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
    y = sign_ste(x).sum()
    y.backward()
    assert torch.equal(sign_ste(x).detach(), torch.tensor([-1.0, -1.0, 1.0, 1.0]))
    assert torch.equal(x.grad, torch.tensor([0.0, 1.0, 1.0, 0.0]))
