# CIFAR-100 Experiment Matrix

This directory contains the CIFAR-100 supervised experiment matrix.

- `resnet18_reference_cifar100.yaml`: accuracy reference and default KD teacher.
- `hbcc_small_no_mix_cifar100.yaml`: small HBCC CE-only baseline without mixup/cutmix.
- `hbcc_small_light_aug_cifar100.yaml`: small HBCC recipe with CIFAR-100 regularization.
- `hbcc_medium_light_aug_cifar100.yaml`: medium HBCC capacity check for CIFAR-100.

Run the full matrix from the repo root:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\run_cifar100_experiments.py
```

Use `--smoke` to verify the pipeline without full training.
