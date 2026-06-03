from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KNOWN_SOURCE_ROOT = Path("/kaggle/input/imagenet-object-localization-challenge")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def has_image(path: Path) -> bool:
    try:
        for child in path.iterdir():
            if is_image_file(child):
                return True
    except OSError:
        return False
    return False


def class_dirs(split_dir: Path) -> list[Path]:
    if not split_dir.exists():
        return []
    return sorted(child for child in split_dir.iterdir() if child.is_dir() and has_image(child))


def count_flat_images(split_dir: Path, limit: int | None = None) -> int:
    count = 0
    if not split_dir.exists():
        return count
    for child in split_dir.iterdir():
        if is_image_file(child):
            count += 1
            if limit and count >= limit:
                return count
    return count


def score_split_dir(path: Path) -> int:
    classes = len(class_dirs(path))
    flat = count_flat_images(path, limit=100_000)
    return classes * 1_000_000 + flat


def find_split_dir(source: Path, split: str) -> Path:
    known = [
        source / "ILSVRC" / "Data" / "CLS-LOC" / split,
        source / "Data" / "CLS-LOC" / split,
        source / split,
    ]
    candidates = [path for path in known if path.exists()]
    candidates.extend(path for path in source.rglob(split) if path.is_dir())
    unique = sorted(set(candidates), key=lambda p: (len(p.parts), str(p)))
    scored = [(score_split_dir(path), path) for path in unique]
    scored = [(score, path) for score, path in scored if score > 0]
    if not scored:
        raise FileNotFoundError(f"Could not find a usable '{split}' split under {source}")
    return max(scored, key=lambda item: item[0])[1]


def find_solution_csv(source: Path) -> Path | None:
    known = [
        source / "LOC_val_solution.csv",
        source / "LOC_val_solution.csv.zip",
        source / "ILSVRC" / "LOC_val_solution.csv",
        source / "ILSVRC" / "LOC_val_solution.csv.zip",
        source / "ImageSets" / "CLS-LOC" / "val_solution.csv",
        source / "ImageSets" / "CLS-LOC" / "val_solution.csv.zip",
    ]
    for path in known:
        if path.exists():
            return path
    matches = sorted(list(source.rglob("*val*solution*.csv")) + list(source.rglob("*val*solution*.csv.zip")))
    return matches[0] if matches else None


def read_val_solution(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"No CSV file found inside {path}")
            with archive.open(names[0]) as raw:
                text = raw.read().decode("utf-8")
        return read_val_solution_text(text, str(path))

    return read_val_solution_text(path.read_text(encoding="utf-8"), str(path))


def read_val_solution_text(text: str, source_name: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    lines = text.splitlines()
    with io.StringIO("\n".join(lines)) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and {"ImageId", "PredictionString"}.issubset(reader.fieldnames):
            for row in reader:
                image_id = str(row["ImageId"]).strip()
                prediction = str(row["PredictionString"]).strip()
                if image_id and prediction:
                    labels[Path(image_id).stem] = prediction.split()[0]
            return labels

    with io.StringIO("\n".join(lines)) as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return labels
        for row in reader:
            if len(row) < 2:
                continue
            image_id = row[0].strip()
            prediction = row[1].strip()
            if image_id and prediction:
                labels[Path(image_id).stem] = prediction.split()[0]
    return labels


def print_failure_context(source: Path) -> None:
    print("\nDiagnostic context:", file=sys.stderr)
    print(f"  source exists: {source.exists()} -> {source}", file=sys.stderr)
    input_root = Path("/kaggle/input")
    if input_root.exists():
        try:
            print("  /kaggle/input entries:", file=sys.stderr)
            for child in sorted(input_root.iterdir())[:30]:
                print(f"    - {child}", file=sys.stderr)
        except OSError as exc:
            print(f"  could not list /kaggle/input: {exc}", file=sys.stderr)
    if source.exists():
        for rel in [
            "ILSVRC/Data/CLS-LOC/train",
            "ILSVRC/Data/CLS-LOC/val",
            "ILSVRC/Data/CLS-LOC/test",
            "LOC_val_solution.csv",
            "LOC_val_solution.csv.zip",
        ]:
            path = source / rel
            print(f"  {rel}: {path.exists()} -> {path}", file=sys.stderr)


def materialize(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        try:
            os.symlink(src, dst, target_is_directory=src.is_dir())
            return
        except OSError as exc:
            raise RuntimeError(
                f"Could not create symlink {dst} -> {src}. "
                "Use --mode copy if your environment does not allow symlinks."
            ) from exc
    if mode == "hardlink":
        if src.is_dir():
            raise ValueError("Hardlink mode supports files only; use symlink or copy for class directories.")
        os.link(src, dst)
        return
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def prepare_train(train_src: Path, train_out: Path, mode: str) -> tuple[int, int]:
    classes = class_dirs(train_src)
    if not classes:
        raise ValueError(f"Train split is not class-folder based: {train_src}")
    for src_class in classes:
        materialize(src_class, train_out / src_class.name, mode)
    image_count = sum(1 for src_class in classes for file in src_class.iterdir() if is_image_file(file))
    return len(classes), image_count


def prepare_val_from_folders(val_src: Path, val_out: Path, mode: str) -> tuple[int, int]:
    classes = class_dirs(val_src)
    for src_class in classes:
        materialize(src_class, val_out / src_class.name, mode)
    image_count = sum(1 for src_class in classes for file in src_class.iterdir() if is_image_file(file))
    return len(classes), image_count


def prepare_val_from_solution(val_src: Path, val_out: Path, solution_csv: Path, mode: str) -> tuple[int, int]:
    labels = read_val_solution(solution_csv)
    if not labels:
        raise ValueError(f"No validation labels found in {solution_csv}")

    files_by_stem = {file.stem: file for file in val_src.iterdir() if is_image_file(file)}
    missing: list[str] = []
    linked = 0
    for image_id, class_id in sorted(labels.items()):
        src = files_by_stem.get(image_id)
        if src is None:
            missing.append(image_id)
            continue
        materialize(src, val_out / class_id / src.name, mode)
        linked += 1
    if missing:
        preview = ", ".join(missing[:5])
        print(f"WARNING: {len(missing)} validation images from the solution CSV were not found. First missing: {preview}")
    return len({label for label in labels.values()}), linked


def write_summary(output: Path, summary: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "imagenet_kaggle_layout.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Kaggle ImageNet-1k for torchvision ImageFolder.")
    parser.add_argument(
        "--source",
        type=Path,
        default=KNOWN_SOURCE_ROOT,
        help="Kaggle ImageNet dataset root, for example /kaggle/input/imagenet-object-localization-challenge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/imagenet"),
        help="Output root consumed by this repo. It will contain train/ and val/.",
    )
    parser.add_argument("--train-dir", type=Path, default=None, help="Optional explicit train split directory.")
    parser.add_argument("--val-dir", type=Path, default=None, help="Optional explicit val split directory.")
    parser.add_argument("--solution-csv", type=Path, default=None, help="Optional explicit LOC_val_solution.csv path.")
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy", "hardlink"],
        default="symlink",
        help="How to materialize files. Kaggle normally works best with symlink.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists():
        raise FileNotFoundError(f"ImageNet source does not exist: {source}")

    train_src = args.train_dir.resolve() if args.train_dir else find_split_dir(source, "train")
    val_src = args.val_dir.resolve() if args.val_dir else find_split_dir(source, "val")
    solution_csv = args.solution_csv.resolve() if args.solution_csv else find_solution_csv(source)

    print(f"source: {source}")
    print(f"train:  {train_src}")
    print(f"val:    {val_src}")
    print(f"output: {output}")
    if solution_csv:
        print(f"val labels: {solution_csv}")

    train_classes, train_images = prepare_train(train_src, output / "train", args.mode)
    if class_dirs(val_src):
        val_classes, val_images = prepare_val_from_folders(val_src, output / "val", args.mode)
    else:
        if solution_csv is None:
            raise FileNotFoundError(
                "Validation split is flat, but LOC_val_solution.csv was not found. "
                "Pass it explicitly with --solution-csv."
            )
        val_classes, val_images = prepare_val_from_solution(val_src, output / "val", solution_csv, args.mode)

    summary = {
        "source": str(source),
        "output": str(output),
        "mode": args.mode,
        "train_dir": str(train_src),
        "val_dir": str(val_src),
        "solution_csv": str(solution_csv) if solution_csv else None,
        "train_classes": train_classes,
        "train_images": train_images,
        "val_classes": val_classes,
        "val_images": val_images,
    }
    write_summary(output, summary)
    print(json.dumps(summary, indent=2))
    print(f"Ready. Use data.root={output}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            parsed = parse_args()
            print_failure_context(parsed.source)
        except Exception:
            pass
        raise
