param(
  [string]$ImageNetRoot = "data\imagenet",
  [string]$Python = "D:\Anaconda\envs\CoC\python.exe",
  [string]$Model = "coc_tiny",
  [string]$Output = "runs_imagenet_official",
  [string]$Experiment = "official_coc_tiny_imagenet",
  [int]$BatchSize = 128,
  [double]$LearningRate = 0.001,
  [double]$DropPath = 0.1
)

$ErrorActionPreference = "Stop"

& $Python docs\Context-Cluster\train.py `
  --data_dir $ImageNetRoot `
  --train-split train `
  --val-split val `
  --model $Model `
  --num-classes 1000 `
  --batch-size $BatchSize `
  --validation-batch-size $BatchSize `
  --lr $LearningRate `
  --drop-path $DropPath `
  --amp `
  --output $Output `
  --experiment $Experiment

$Checkpoint = "$Output\$Experiment\model_best.pth.tar"
if (Test-Path $Checkpoint) {
  & $Python docs\Context-Cluster\validate.py $ImageNetRoot `
    --split val `
    --model $Model `
    --num-classes 1000 `
    --batch-size $BatchSize `
    --checkpoint $Checkpoint `
    --amp
}
