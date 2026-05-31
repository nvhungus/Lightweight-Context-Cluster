$ErrorActionPreference = "Stop"
$PY = "D:\Anaconda\envs\CoC\python.exe"

& $PY tools\train.py --config configs\recipes\hbcc_accuracy_no_mix.yaml --output runs_accuracy --no-progress --print-every 5
& $PY tools\train.py --config configs\hbcc_accuracy_small.yaml --output runs_accuracy --no-progress --print-every 5
& $PY tools\train.py --config configs\hbcc_accuracy_medium.yaml --output runs_accuracy --no-progress --print-every 5

& $PY tools\benchmark.py --config configs\hbcc_accuracy_small.yaml --checkpoint runs_accuracy\hbcc_accuracy_small\best.pth --output results\benchmark --profile
& $PY tools\benchmark.py --config configs\hbcc_accuracy_medium.yaml --checkpoint runs_accuracy\hbcc_accuracy_medium\best.pth --output results\benchmark --profile
