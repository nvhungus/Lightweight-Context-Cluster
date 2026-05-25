from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .metrics import count_params, flop_count, model_size_mb
from .models.cluster import ContextClusterOp


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark_model(
    model: nn.Module,
    device: torch.device,
    batch_sizes: list[int],
    image_size: int = 32,
    warmup: int = 30,
    runs: int = 100,
    throughput_runs: int | None = None,
) -> dict[str, Any]:
    model.eval().to(device)
    throughput_runs = throughput_runs or runs
    results: dict[str, Any] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        for _ in range(warmup):
            model(x)
        synchronize(device)

        start = time.perf_counter()
        for _ in range(runs):
            model(x)
            synchronize(device)
        strict_time = time.perf_counter() - start
        latency_ms = strict_time * 1000.0 / runs

        for _ in range(warmup):
            model(x)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(throughput_runs):
            model(x)
        synchronize(device)
        stream_time = time.perf_counter() - start
        throughput = batch_size * throughput_runs / max(stream_time, 1e-12)
        results[f"latency_ms_b{batch_size}"] = latency_ms
        results[f"throughput_b{batch_size}"] = throughput
    if device.type == "cuda":
        results["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        results["peak_memory_mb"] = None
    return results


def profile_operators(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 1,
    image_size: int = 32,
    warmup: int = 5,
    active: int = 10,
    out_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    model.eval().to(device)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=warmup, active=active),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(1 + warmup + active):
            model(x)
            prof.step()
    rows = []
    for item in prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=30).splitlines():
        rows.append({"row": item})
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(row["row"] for row in rows), encoding="utf-8")
    return rows


def model_static_metrics(model: nn.Module, device: torch.device, image_size: int = 32) -> dict[str, Any]:
    record: dict[str, Any] = {}
    record.update(count_params(model))
    record["model_size_mb"] = model_size_mb(model)
    record.update(flop_count(model, (1, 3, image_size, image_size), device))
    record["bops"] = estimate_bops(model, (1, 3, image_size, image_size), device)
    return record


def write_record(record: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.no_grad()
def estimate_bops(model: nn.Module, input_shape: tuple[int, int, int, int], device: torch.device) -> int:
    total = 0
    handles = []

    def hook(module: ContextClusterOp, inputs: tuple[torch.Tensor, ...], _: torch.Tensor) -> None:
        nonlocal total
        total += module.theoretical_bops(tuple(inputs[0].shape))

    for module in model.modules():
        if isinstance(module, ContextClusterOp):
            handles.append(module.register_forward_hook(hook))
    was_training = model.training
    model.eval().to(device)
    model(torch.randn(input_shape, device=device))
    model.train(was_training)
    for handle in handles:
        handle.remove()
    return int(total)


def append_record_csv(record: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for key, value in record.items():
        if isinstance(value, (dict, list, tuple)):
            flat[key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[key] = value
    exists = out.exists()
    with out.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(flat)
