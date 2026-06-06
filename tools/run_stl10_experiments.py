from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class Experiment:
    name: str
    config: str
    kind: str = "student"
    overrides: tuple[str, ...] = field(default_factory=tuple)
    teacher: bool = False


REFERENCE = Experiment(
    name="resnet18_reference_stl10",
    config="configs/stl10/resnet18_reference_stl10.yaml",
    kind="reference",
)

CE_EXPERIMENTS = [
    Experiment("hbcc_small_no_mix_stl10", "configs/stl10/hbcc_small_no_mix_stl10.yaml"),
    Experiment("hbcc_small_light_aug_stl10", "configs/stl10/hbcc_small_light_aug_stl10.yaml"),
    Experiment("hbcc_small_full_aug_stl10", "configs/stl10/hbcc_small_full_aug_stl10.yaml"),
    Experiment("hbcc_local_heavy_stl10", "configs/stl10/hbcc_local_heavy_stl10.yaml"),
    Experiment("hbcc_medium_light_aug_stl10", "configs/stl10/hbcc_medium_light_aug_stl10.yaml"),
]

KD_EXPERIMENTS = [
    Experiment(
        "hbcc_small_kd_stl10",
        "configs/stl10/hbcc_small_light_aug_stl10.yaml",
        kind="kd",
        teacher=True,
        overrides=("experiment.name=hbcc_small_kd_stl10", "train.kd_alpha=0.2", "train.kd_temperature=3.0"),
    ),
    Experiment(
        "hbcc_local_heavy_kd_stl10",
        "configs/stl10/hbcc_local_heavy_stl10.yaml",
        kind="kd",
        teacher=True,
        overrides=("experiment.name=hbcc_local_heavy_kd_stl10", "train.kd_alpha=0.2", "train.kd_temperature=3.0"),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full STL-10 experiment matrix.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child commands.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="runs_stl10")
    parser.add_argument("--benchmark-output", default="results/benchmark_stl10")
    parser.add_argument("--summary-output", default="results/stl10_summary.csv")
    parser.add_argument("--only", nargs="*", help="Run only these experiment names.")
    parser.add_argument("--epochs", type=int, help="Override train.epochs for every training run.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-students", action="store_true")
    parser.add_argument("--skip-kd", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Write torch operator profiles during benchmarking.")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bars in child training runs.")
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--benchmark-warmup", type=int, default=30)
    parser.add_argument("--benchmark-runs", type=int, default=100)
    parser.add_argument("--benchmark-batch-sizes", nargs="+", type=int, default=[1, 16, 64, 128])
    parser.add_argument("--smoke", action="store_true", help="Run one batch/epoch and tiny benchmark settings.")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _selected(experiments: list[Experiment], only: set[str] | None) -> list[Experiment]:
    if only is None:
        return experiments
    return [exp for exp in experiments if exp.name in only]


def _train_args(args: argparse.Namespace, exp: Experiment) -> list[str]:
    cmd = [
        args.python,
        "tools/train.py",
        "--config",
        exp.config,
        "--output",
        args.output,
        "--print-every",
        str(args.print_every),
    ]
    if not args.progress:
        cmd.append("--no-progress")
    for override in (f"data.root={args.data_root}", *exp.overrides):
        cmd.extend(["--override", override])
    if args.epochs is not None:
        cmd.extend(["--override", f"train.epochs={args.epochs}"])
    if args.smoke:
        cmd.extend(
            [
                "--override",
                "train.epochs=1",
                "--limit-train-batches",
                "1",
                "--limit-val-batches",
                "1",
                "--limit-test-batches",
                "1",
            ]
        )
    if exp.teacher:
        teacher_checkpoint = Path(args.output) / REFERENCE.name / "best.pth"
        if not teacher_checkpoint.exists():
            raise FileNotFoundError(f"Train the reference first or provide its checkpoint: {teacher_checkpoint}")
        cmd.extend(
            [
                "--teacher-config",
                REFERENCE.config,
                "--teacher-checkpoint",
                str(teacher_checkpoint),
            ]
        )
    return cmd


def _benchmark_args(args: argparse.Namespace, exp: Experiment) -> list[str]:
    cmd = [
        args.python,
        "tools/benchmark.py",
        "--config",
        exp.config,
        "--checkpoint",
        str(Path(args.output) / exp.name / "best.pth"),
        "--output",
        args.benchmark_output,
        "--warmup",
        str(1 if args.smoke else args.benchmark_warmup),
        "--runs",
        str(2 if args.smoke else args.benchmark_runs),
        "--batch-sizes",
        *[str(item) for item in ([1, 4] if args.smoke else args.benchmark_batch_sizes)],
        "--override",
        f"data.root={args.data_root}",
    ]
    for override in exp.overrides:
        cmd.extend(["--override", override])
    if args.profile:
        cmd.append("--profile")
    return cmd


def main() -> None:
    args = parse_args()
    only = set(args.only) if args.only else None
    train_experiments: list[Experiment] = []
    if not args.skip_reference:
        train_experiments.append(REFERENCE)
    if not args.skip_students:
        train_experiments.extend(CE_EXPERIMENTS)
    if not args.skip_kd:
        train_experiments.extend(KD_EXPERIMENTS)
    train_experiments = _selected(train_experiments, only)

    if not args.skip_train:
        for exp in train_experiments:
            _run(_train_args(args, exp))

    benchmark_experiments = _selected([REFERENCE, *CE_EXPERIMENTS, *KD_EXPERIMENTS], only)
    if not args.skip_benchmark:
        for exp in benchmark_experiments:
            checkpoint = Path(args.output) / exp.name / "best.pth"
            if not checkpoint.exists():
                print(f"skip benchmark: missing checkpoint {checkpoint}", flush=True)
                continue
            _run(_benchmark_args(args, exp))

    if not args.skip_summary:
        _run(
            [
                args.python,
                "tools/stl10_report.py",
                "--runs",
                args.output,
                "--benchmarks",
                args.benchmark_output,
                "--output",
                args.summary_output,
            ]
        )


if __name__ == "__main__":
    main()
