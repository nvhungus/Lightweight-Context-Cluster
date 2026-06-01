$ErrorActionPreference = "Stop"

$PY = "D:\Anaconda\envs\CoC\python.exe"
$CONFIG = "configs\coc_cifar_baseline.yaml"
$RUN_DIR = "runs_coc_baseline"
$EXP = "coc_cifar_baseline"
$CKPT = "$RUN_DIR\$EXP\best.pth"

& $PY tools\train.py `
  --config $CONFIG `
  --output $RUN_DIR `
  --no-progress `
  --print-every 5

& $PY tools\benchmark.py `
  --config $CONFIG `
  --checkpoint $CKPT `
  --output results\benchmark `
  --profile

& $PY tools\summarize_training.py "$RUN_DIR\$EXP"
