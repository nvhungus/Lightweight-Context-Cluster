$ErrorActionPreference = "Stop"
$PY = "D:\Anaconda\envs\CoC\python.exe"

& $PY -m pytest
& $PY tools\shape_trace.py --config configs\smoke.yaml
& $PY tools\benchmark.py --config configs\smoke.yaml --batch-sizes 1 4 --warmup 2 --runs 3
