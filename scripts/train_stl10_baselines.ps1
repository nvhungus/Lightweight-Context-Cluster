param(
  [string]$Python = "D:\Anaconda\envs\CoC\python.exe",
  [string]$Output = "runs_stl10",
  [string]$BenchmarkOutput = "results\benchmark_stl10"
)

$ErrorActionPreference = "Stop"

$configs = @(
  "configs\stl10\hbcc_accuracy_small_stl10.yaml",
  "configs\stl10\hbcc_accuracy_medium_stl10.yaml",
  "configs\stl10\resnet18_stl10.yaml"
)

foreach ($config in $configs) {
  & $Python tools\train.py `
    --config $config `
    --output $Output `
    --no-progress `
    --print-every 5
}

$benchmarks = @(
  @("configs\stl10\hbcc_accuracy_small_stl10.yaml", "$Output\hbcc_accuracy_small_stl10\best.pth"),
  @("configs\stl10\hbcc_accuracy_medium_stl10.yaml", "$Output\hbcc_accuracy_medium_stl10\best.pth"),
  @("configs\stl10\resnet18_stl10.yaml", "$Output\resnet18_stl10\best.pth")
)

foreach ($job in $benchmarks) {
  & $Python tools\benchmark.py `
    --config $job[0] `
    --checkpoint $job[1] `
    --output $BenchmarkOutput `
    --batch-sizes 1 16 64 128 `
    --profile
}
