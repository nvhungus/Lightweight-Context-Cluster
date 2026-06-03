# Lightweight HBCC for Context Cluster

This workspace implements the research plan in `docs/lightweight_hbcc_research_plan.md` as a reproducible CIFAR pipeline.

## Environment

The target conda environment is `CoC`. On this machine it has been initialized with Python 3.11 and CUDA PyTorch.
The official Context-Cluster ImageNet scripts in `docs/Context-Cluster` require `timm==0.6.13`.

```powershell
conda activate CoC
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If the env has to be rebuilt:

```powershell
conda env update -n CoC -f environment.yml
```

When `conda activate` is unreliable, call the interpreter directly:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\shape_trace.py --config configs\smoke.yaml
```

## What Is Implemented

- CoC-style context clustering with cosine similarity, hard assignment, sigmoid-scaled aggregation and dispatch.
- Optional RGB+XY coordinate augmentation.
- Early reducer schedule for CIFAR: `32 -> 16 -> 8 -> 4 -> 2`.
- HBCC-Latency Tiny/Small configs.
- Current-reference HBCC config that intentionally keeps the first cluster stage at `32x32`.
- Local branch ablations: identity, DWConv, fixed LBPConv.
- Similarity ablations: cosine and simulated Hamming with STE.
- Structured channel mask training and materialized pruned-config export.
- CE-only and KD training.
- Benchmark protocol for batch `1, 16, 64, 128`, strict latency, streaming throughput, peak memory, FLOPs best effort and torch operator profile.
- Pareto report generation from JSON benchmark records.

## Quick Checks

```powershell
& D:\Anaconda\envs\CoC\python.exe -m pytest
& D:\Anaconda\envs\CoC\python.exe tools\shape_trace.py --config configs\hbcc_latency_tiny.yaml
& D:\Anaconda\envs\CoC\python.exe tools\benchmark.py --config configs\smoke.yaml --batch-sizes 1 4 --warmup 2 --runs 3
```

## Training

For CIFAR-10 and CIFAR-100, the training pipeline uses the original training
split for model selection and keeps the official test split for final reporting:

- `train`: 45k images from the original training split
- `val`: 5k images from the original training split, controlled by `data.split_seed`
- `test`: official 10k test split, evaluated once on `best.pth`

Validation metrics are written into `metrics.jsonl` every epoch. Final test
metrics are written to both `metrics.jsonl` and `test_metrics.json`.

For ImageNet-1k, prepare the dataset manually because ImageNet cannot be
downloaded by this project. The standard `ImageFolder` structure is:

```text
data/imagenet/
  train/<class_id>/*.JPEG
  val/<class_id>/*.JPEG
```

For the common Kaggle ImageNet localization dataset, this repo trainer can also
read the raw Kaggle layout directly, without creating symlinks:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\train.py --config configs\imagenet\hbcc_accuracy_small_imagenet.yaml --output runs_imagenet --override data.root=/kaggle/input/imagenet-object-localization-challenge --override data.layout=kaggle_cls_loc --override data.train_split=ILSVRC/Data/CLS-LOC/train --override data.val_split=ILSVRC/Data/CLS-LOC/val
```

The conversion script is still available when you need an `ImageFolder` layout,
for example for the official Context-Cluster entrypoint:

```powershell
& D:\Anaconda\envs\CoC\python.exe scripts\prepare_kaggle_imagenet.py --source /kaggle/input/imagenet-object-localization-challenge --output data/imagenet --mode symlink
```

The public ImageNet-1k benchmark reports accuracy on the official validation
split. Matching the Context-Cluster paper and official code, ImageNet configs
train on `train/`, evaluate on `val/`, and set `train.skip_test: true`; `val_acc1`
is the reported ImageNet accuracy.

Student CE-only:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\train.py --config configs\hbcc_latency_tiny.yaml --output runs
```

Short proxy training for search:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\train.py --config configs\hbcc_latency_tiny.yaml --output runs_proxy --override train.epochs=30
```

Knowledge distillation:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\train.py --config configs\hbcc_latency_tiny.yaml --output runs_kd --teacher-config configs\baselines\resnet18_cifar.yaml --teacher-checkpoint runs\resnet18_cifar\best.pth --override train.kd_alpha=0.5 --override train.kd_temperature=4.0
```

Official Context-Cluster ImageNet baseline from `docs/Context-Cluster`:

```powershell
& .\scripts\train_official_coc_imagenet.ps1 -ImageNetRoot data\imagenet
```

Selected ImageNet training matrix through this repo's trainer, including
`hbcc_accuracy_small/medium` converted to ImageNet:

```powershell
& .\scripts\train_imagenet_baselines.ps1 -ImageNetRoot data\imagenet
```

The notebook version of this workflow is `notebooks/imagenet_baseline_pipeline.ipynb`.

## Benchmark Matrix

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\benchmark.py --config configs\baselines\resnet18_cifar.yaml --output results\benchmark --profile
& D:\Anaconda\envs\CoC\python.exe tools\benchmark.py --config configs\baselines\mobilenet_v2_cifar.yaml --output results\benchmark --profile
& D:\Anaconda\envs\CoC\python.exe tools\benchmark.py --config configs\coc_cifar_baseline.yaml --output results\benchmark --profile
& D:\Anaconda\envs\CoC\python.exe tools\benchmark.py --config configs\hbcc_current_reference.yaml --output results\benchmark --profile
& D:\Anaconda\envs\CoC\python.exe tools\benchmark.py --config configs\hbcc_latency_tiny.yaml --output results\benchmark --profile
```

Trained accuracy can be added to benchmark JSON records from `runs/*/metrics.jsonl` before building final Pareto tables.
For CIFAR runs, use `test_acc1` for held-out reporting. For ImageNet runs that match Context-Cluster, use `val_acc1` as the reported official validation accuracy.

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\pareto_report.py results\benchmark --output results\pareto.md
```

## Ablations

Generate one config per ablation from the Tiny base:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\ablation_matrix.py --base configs\hbcc_latency_tiny.yaml --output-dir configs\generated_ablations
```

Main manual ablation configs are also available in `configs/ablations`.

## Pruning Export

Train with masks:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\train.py --config configs\ablations\hbcc_tiny_pruning_mask.yaml --output runs_pruning
```

Export a smaller materialized config:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\export_pruned.py --config configs\ablations\hbcc_tiny_pruning_mask.yaml --checkpoint runs_pruning\hbcc_tiny_pruning_mask\best.pth --output configs\generated_ablations\hbcc_tiny_pruned_export.yaml
```

Then fine-tune and benchmark the exported config.
