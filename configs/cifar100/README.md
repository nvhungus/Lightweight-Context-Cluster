# CIFAR-100 Config Notes

The primary CIFAR-100 pipeline now mirrors the CIFAR-10 method: it reuses the
existing CIFAR-10 configs with overrides for `data.name=cifar100`,
`model.num_classes=100`, and a CIFAR-100-specific `experiment.name`.

These materialized configs are kept as convenience entry points for quick
single-run experiments:

- `resnet18_reference_cifar100.yaml`: accuracy reference and default KD teacher.
- `hbcc_small_no_mix_cifar100.yaml`: small HBCC CE-only baseline without mixup/cutmix.
- `hbcc_small_light_aug_cifar100.yaml`: small HBCC recipe with CIFAR-100 regularization.
- `hbcc_medium_light_aug_cifar100.yaml`: medium HBCC capacity check for CIFAR-100.

Run the full CIFAR-10-style pipeline from the repo root:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\run_cifar100_experiments.py
```

Use `--smoke` to verify the pipeline without full training.
