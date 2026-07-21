"""Split YOLO training data among FL clients and replicate validation data."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split images/train evenly among client_0, client_1, ... and copy "
            "the complete images/val split to every client."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/root_datasets"),
        help="Source YOLO dataset (default: datasets/root_datasets)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets"),
        help="Parent directory for client_N datasets (default: datasets)",
    )
    parser.add_argument(
        "--num-clients", type=int, default=3, help="Number of clients (default: 3)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible splits"
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Copy files or create space-saving hard links (default: copy)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing client_N directories selected by --num-clients",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and show counts only"
    )
    return parser.parse_args()


def find_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_source(source: Path) -> tuple[dict, dict[str, list[Path]]]:
    yaml_path = source / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing dataset config: {yaml_path}")

    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if "names" not in config:
        raise ValueError(f"{yaml_path} must contain a 'names' mapping or list")

    images_by_split: dict[str, list[Path]] = {}
    problems: list[str] = []
    for split in SPLITS:
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            problems.append(f"Missing images/{split} or labels/{split} directory")
            continue

        images = find_images(image_dir)
        if not images:
            problems.append(f"No images found in {image_dir}")
            continue
        for image in images:
            relative = image.relative_to(image_dir)
            label = label_dir / relative.with_suffix(".txt")
            if not label.is_file():
                problems.append(f"Missing label for {image}: expected {label}")
                if len(problems) >= 20:
                    break
        images_by_split[split] = images

    if problems:
        details = "\n  - ".join(problems)
        raise ValueError(f"Invalid source dataset:\n  - {details}")
    return config, images_by_split


def assign_images(
    images_by_split: dict[str, list[Path]], num_clients: int, seed: int
) -> dict[int, dict[str, list[Path]]]:
    assignments = {
        client_id: {split: [] for split in SPLITS}
        for client_id in range(num_clients)
    }
    train_images = images_by_split["train"].copy()
    random.Random(seed).shuffle(train_images)
    for index, image in enumerate(train_images):
        assignments[index % num_clients]["train"].append(image)

    # Evaluate every global model on exactly the same validation set so that
    # per-client metrics are directly comparable.
    for client_id in range(num_clients):
        assignments[client_id]["val"] = images_by_split["val"].copy()
    return assignments


def transfer(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        destination.hardlink_to(source)


def write_client_dataset(
    source: Path,
    destination: Path,
    source_config: dict,
    assigned: dict[str, list[Path]],
    mode: str,
) -> None:
    for split, images in assigned.items():
        source_image_dir = source / "images" / split
        source_label_dir = source / "labels" / split
        for image in images:
            relative = image.relative_to(source_image_dir)
            transfer(image, destination / "images" / split / relative, mode)
            transfer(
                source_label_dir / relative.with_suffix(".txt"),
                destination / "labels" / split / relative.with_suffix(".txt"),
                mode,
            )

    client_config = {
        "train": "images/train",
        "val": "images/val",
        "names": source_config["names"],
    }
    (destination / "data.yaml").write_text(
        yaml.safe_dump(client_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.num_clients < 1:
        raise ValueError("--num-clients must be at least 1")

    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source_config, images_by_split = validate_source(source)
    assignments = assign_images(images_by_split, args.num_clients, args.seed)
    destinations = [output_dir / f"client_{i}" for i in range(args.num_clients)]

    print(f"Source: {source}")
    print(f"Mode: {args.mode}; seed: {args.seed}")
    for client_id, assigned in assignments.items():
        counts = ", ".join(f"{split}={len(assigned[split])}" for split in SPLITS)
        print(f"client_{client_id}: {counts}")

    if args.dry_run:
        print("Dry run complete; no files were changed.")
        return

    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output already exists: {paths}. Use --overwrite to replace it."
        )
    if args.overwrite:
        for destination in existing:
            shutil.rmtree(destination)

    for client_id, destination in enumerate(destinations):
        write_client_dataset(
            source,
            destination,
            source_config,
            assignments[client_id],
            args.mode,
        )
    print(f"Created {args.num_clients} client datasets in {output_dir}")


if __name__ == "__main__":
    main()
