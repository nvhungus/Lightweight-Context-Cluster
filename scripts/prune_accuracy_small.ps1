$ErrorActionPreference = "Stop"

$PY = "D:\Anaconda\envs\CoC\python.exe"
$BASE_CONFIG = "configs\hbcc_accuracy_small.yaml"
$PRUNE_CONFIG = "configs\ablations\hbcc_accuracy_small_pruning_mask.yaml"
$BASE_CKPT = "runs_accuracy\hbcc_accuracy_small\best.pth"
$PRUNE_RUN = "runs_pruning_accuracy"
$PRUNE_EXP = "hbcc_accuracy_small_pruning_mask"
$PRUNE_CKPT = "$PRUNE_RUN\$PRUNE_EXP\best.pth"
$EXPORTED_CONFIG = "configs\generated_ablations\hbcc_accuracy_small_pruned_export.yaml"
$FINETUNE_RUN = "runs_pruned_accuracy"
$THRESHOLD = "0.90"

if (!(Test-Path $BASE_CKPT)) {
  throw "Missing $BASE_CKPT. Train hbcc_accuracy_small first."
}

& $PY tools\train.py `
  --config $PRUNE_CONFIG `
  --output $PRUNE_RUN `
  --resume $BASE_CKPT `
  --no-progress `
  --print-every 5

& $PY tools\inspect_pruning_masks.py `
  --config $PRUNE_CONFIG `
  --checkpoint $PRUNE_CKPT `
  --thresholds 0.5 0.7 0.8 0.9 0.95

& $PY tools\export_pruned.py `
  --config $PRUNE_CONFIG `
  --checkpoint $PRUNE_CKPT `
  --output $EXPORTED_CONFIG `
  --threshold $THRESHOLD `
  --min-channels 16 `
  --round-to 8

& $PY tools\train.py `
  --config $EXPORTED_CONFIG `
  --output $FINETUNE_RUN `
  --no-progress `
  --print-every 5

& $PY tools\benchmark.py `
  --config $EXPORTED_CONFIG `
  --checkpoint "$FINETUNE_RUN\hbcc_accuracy_small_pruned_export\best.pth" `
  --output results\benchmark `
  --profile
