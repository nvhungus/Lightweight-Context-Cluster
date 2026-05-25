$ErrorActionPreference = "Stop"
$PY = "D:\Anaconda\envs\CoC\python.exe"

$configs = @(
  "configs\baselines\resnet18_cifar.yaml",
  "configs\baselines\mobilenet_v2_cifar.yaml",
  "configs\baselines\shufflenet_v2_x1_0_cifar.yaml",
  "configs\coc_cifar_baseline.yaml",
  "configs\hbcc_current_reference.yaml",
  "configs\hbcc_latency_tiny.yaml",
  "configs\hbcc_latency_small.yaml"
)

foreach ($cfg in $configs) {
  & $PY tools\benchmark.py --config $cfg --output results\benchmark --profile
}
