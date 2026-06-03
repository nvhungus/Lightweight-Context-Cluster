param(
  [string]$ImageNetRoot = "data\imagenet",
  [string]$Python = "D:\Anaconda\envs\CoC\python.exe",
  [string]$Output = "runs_imagenet",
  [string]$BenchmarkOutput = "results\benchmark_imagenet"
)

$ErrorActionPreference = "Stop"

$configs = @(
  "configs\imagenet\hbcc_accuracy_small_imagenet.yaml",
  "configs\imagenet\hbcc_accuracy_medium_imagenet.yaml"
)

foreach ($config in $configs) {
  & $Python tools\train.py `
    --config $config `
    --output $Output `
    --override "data.root=$ImageNetRoot" `
    --no-progress `
    --print-every 5
}

$benchmarks = @(
  @("configs\imagenet\hbcc_accuracy_small_imagenet.yaml", "$Output\hbcc_accuracy_small_imagenet\best.pth"),
  @("configs\imagenet\hbcc_accuracy_medium_imagenet.yaml", "$Output\hbcc_accuracy_medium_imagenet\best.pth")
)

foreach ($job in $benchmarks) {
  & $Python tools\benchmark.py `
    --config $job[0] `
    --checkpoint $job[1] `
    --output $BenchmarkOutput `
    --batch-sizes 1 16 64 128 `
    --profile
}
