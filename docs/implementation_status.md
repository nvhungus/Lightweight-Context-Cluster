# Implementation Status

This file maps the research plan to concrete code artifacts in this repo.

## Phase 0: Measurement and Profiler

- `tools/benchmark.py`
- `lightweight_hbcc/profiler.py`
- Records device, torch/CUDA version, params, FLOPs best effort, latency, throughput and peak memory.
- Optional `--profile` writes a torch profiler top-operator table.

## Phase 1: Baseline Matrix

- `configs/baselines/resnet18_cifar.yaml`
- `configs/baselines/mobilenet_v2_cifar.yaml`
- `configs/baselines/shufflenet_v2_x1_0_cifar.yaml`
- `configs/coc_cifar_baseline.yaml`
- `configs/hbcc_current_reference.yaml`

## Phase 2: CoC/HBCC Foundation Fixes

- Optional coordinate augmentation: `lightweight_hbcc/models/layers.py::CoordinateAugment`
- Point reducer: `lightweight_hbcc/models/layers.py::PointReducer`
- Stage schedule `16 -> 8 -> 4 -> 2`: `configs/hbcc_latency_tiny.yaml`
- Shape trace: `tools/shape_trace.py`

## Phase 3: Lightweight Search

- Tiny/Small configs: `configs/hbcc_latency_tiny.yaml`, `configs/hbcc_latency_small.yaml`
- Config overrides allow proxy training, e.g. `--override train.epochs=30`.

## Phase 4: Ablations

- Hamming STE: `configs/ablations/hbcc_tiny_hamming_late.yaml`
- LBPConv: `configs/ablations/hbcc_tiny_lbpconv.yaml`
- Pruning masks: `configs/ablations/hbcc_tiny_pruning_mask.yaml`
- Config generator: `tools/ablation_matrix.py`

## Phase 5: Distillation

- `tools/train.py --teacher-config ... --teacher-checkpoint ...`
- KD loss: `lightweight_hbcc/losses.py::DistillationLoss`

## Phase 6: Structured Pruning Export

- Soft channel gates: `lightweight_hbcc/models/layers.py::ChannelGate`
- Export materialized smaller config: `tools/export_pruned.py`

## Phase 7: BitEdge

- Simulated Hamming is implemented for training/accuracy ablation only.
- Real bit-packed XNOR/popcount kernels are intentionally deferred until Track A has a strong Pareto architecture.
