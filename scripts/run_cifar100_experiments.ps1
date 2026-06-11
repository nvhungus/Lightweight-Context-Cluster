param(
  [string]$Python = "",
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$ErrorActionPreference = "Stop"

if (-not $Python) {
  $DefaultPython = "D:\Anaconda\envs\CoC\python.exe"
  if (Test-Path $DefaultPython) {
    $Python = $DefaultPython
  } else {
    $Python = "python"
  }
}

& $Python tools\run_cifar100_experiments.py @Args
