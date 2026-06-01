# Pruning HBCC-Accuracy-Small

The pruning target is `hbcc_accuracy_small`, because it has high accuracy (`~94.2%`) and moderate size (`~1.54M` params).

## Important limitation

The current exporter creates a smaller materialized config from learned channel masks. It does not slice and transfer the old checkpoint weights into the smaller model yet. The exported model must be trained/fine-tuned as a new smaller architecture. Treat this as structured architecture export, not exact weight-preserving pruning.

## Run

Train the base model first:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\train.py --config configs\hbcc_accuracy_small.yaml --output runs_accuracy --no-progress --print-every 5
```

Then run the full pruning workflow:

```powershell
.\scripts\prune_accuracy_small.ps1
```

This does:

1. Starts from `runs_accuracy\hbcc_accuracy_small\best.pth`.
2. Trains pruning masks with `configs\ablations\hbcc_accuracy_small_pruning_mask.yaml`.
3. Prints mask keep counts at thresholds `0.5 0.7 0.8 0.9 0.95`.
4. Exports `configs\generated_ablations\hbcc_accuracy_small_pruned_export.yaml`.
5. Trains the exported smaller model.
6. Benchmarks the exported smaller model.

If the exported model loses too much accuracy, lower the threshold from `0.90` to `0.80` in the script.
