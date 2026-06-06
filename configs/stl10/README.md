# STL-10 Experiment Matrix

This directory contains the STL-10 supervised experiment matrix.

- `resnet18_reference_stl10.yaml`: accuracy reference and default KD teacher.
- `hbcc_small_no_mix_stl10.yaml`: low-regularization CE-only baseline.
- `hbcc_small_light_aug_stl10.yaml`: mild augmentation recipe for the same architecture.
- `hbcc_small_full_aug_stl10.yaml`: previous strong-regularization recipe for direct comparison.
- `hbcc_local_heavy_stl10.yaml`: more local/CNN-biased early stages.
- `hbcc_medium_light_aug_stl10.yaml`: capacity check for the mild recipe.

Run the full matrix from the repo root:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\run_stl10_experiments.py
```

Use `--smoke` to verify the pipeline without full training.
