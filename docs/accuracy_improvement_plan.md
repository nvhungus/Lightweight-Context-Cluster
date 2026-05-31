# Accuracy Improvement Plan

The Kaggle results show the current HBCC models are below the CIFAR ResNet-18 reference:

| Model | Best Top-1 |
|---|---:|
| ResNet-18 CIFAR | 94.23 |
| ShuffleNetV2 x1.0 | 92.45 |
| MobileNetV2 | 91.98 |
| HBCC-Latency-Small | 89.67 |
| HBCC-Tiny + LBPConv | 89.03 |
| HBCC-Latency-Tiny CE-only | 88.38 |
| HBCC-Latency-Tiny KD | 88.06 |

Main diagnosis:

- The current HBCC students overfit: train accuracy is near 99%, while validation is 88-90%.
- KD with `alpha=0.5` did not help Tiny, so the current KD setting should not be the next default.
- Hamming and pruning reduce accuracy in the current setup; keep them deferred for accuracy recovery.
- The best proxy was the wider/deeper model, so accuracy likely needs a larger student and better regularization.

## New configs

Two accuracy-oriented configs have been added:

- `configs/hbcc_accuracy_small.yaml`
- `configs/hbcc_accuracy_medium.yaml`

They differ from latency configs by:

- Wider/deeper stages.
- Hybrid stage 1 instead of purely local stage 1.
- LBPConv only in the first local branch.
- Channel shuffle in early hybrid stages.
- Stronger drop path.
- RandAugment, RandomErasing, MixUp and CutMix support.
- 5 epoch LR warmup.

## Recommended run order

First run the no-mix recipe to isolate architecture gains:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\train.py --config configs\recipes\hbcc_accuracy_no_mix.yaml --output runs_accuracy --no-progress --print-every 5
```

Then run the full augmentation recipes:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\train.py --config configs\hbcc_accuracy_small.yaml --output runs_accuracy --no-progress --print-every 5

& D:\Anaconda\envs\CoC\python.exe tools\train.py --config configs\hbcc_accuracy_medium.yaml --output runs_accuracy --no-progress --print-every 5
```

If `hbcc_accuracy_medium` improves over `hbcc_latency_small`, benchmark it:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\benchmark.py --config configs\hbcc_accuracy_medium.yaml --checkpoint runs_accuracy\hbcc_accuracy_medium\best.pth --output results\benchmark --profile
```

## KD retry

KD should be retried only after CE-only accuracy improves. Use a softer setting than the failed Tiny KD:

```powershell
& D:\Anaconda\envs\CoC\python.exe tools\train.py --config configs\hbcc_accuracy_small.yaml --output runs_accuracy_kd --teacher-config configs\baselines\resnet18_cifar.yaml --teacher-checkpoint runs_teacher\resnet18_cifar\best.pth --override train.kd_alpha=0.2 --override train.kd_temperature=3.0 --no-progress --print-every 5
```

Keep KD only if it beats the same config CE-only.
