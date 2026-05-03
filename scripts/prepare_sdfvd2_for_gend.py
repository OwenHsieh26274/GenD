import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector import prepare_model, process_mixed_types


RAW_SPLIT_TO_SOURCE = {
    "SDFVD2.0_real": "real",
    "SDFVD2.0_fake": "fake",
}
SOURCES = ("real", "fake")


def parse_target_size(raw: str):
    if raw.lower() == "none":
        return None
    width, height = map(int, raw.split(","))
    return (width, height)


def parse_ratio(raw: str) -> tuple[int, int, int]:
    train, val, test = map(int, raw.split(","))
    total = train + val + test
    if total <= 0:
        raise ValueError("Split ratio must sum to a positive value.")
    return train, val, test


def link_or_copy(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def extract_dataset(zip_path: Path, raw_root: Path) -> dict[str, int]:
    raw_root.mkdir(parents=True, exist_ok=True)
    extracted_counts = {source: 0 for source in SOURCES}

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            parts = Path(info.filename).parts
            if len(parts) < 3:
                continue

            source = RAW_SPLIT_TO_SOURCE.get(parts[1])
            if source is None:
                continue

            destination = raw_root / source / Path(parts[-1]).name
            destination.parent.mkdir(parents=True, exist_ok=True)

            if destination.exists() and destination.stat().st_size == info.file_size:
                extracted_counts[source] += 1
                continue

            with zf.open(info) as source_file, destination.open("wb") as target:
                shutil.copyfileobj(source_file, target)
            extracted_counts[source] += 1

    return extracted_counts


def detect_raw_source_dirs(raw_root: Path) -> dict[str, Path]:
    canonical = {source: raw_root / source for source in SOURCES}
    if all(path.is_dir() for path in canonical.values()):
        return canonical

    legacy = {source: raw_root / split_name for split_name, source in RAW_SPLIT_TO_SOURCE.items()}
    if all(path.is_dir() for path in legacy.values()):
        return legacy

    raise FileNotFoundError(f"Could not find raw SDFVD2.0 source directories under {raw_root}")


def count_raw_videos(source_dirs: dict[str, Path]) -> dict[str, int]:
    return {source: sum(1 for _ in source_dir.glob("*.mp4")) for source, source_dir in source_dirs.items()}


def prepare_processing_source_dirs(source_dirs: dict[str, Path]):
    if all(path.name == source for source, path in source_dirs.items()):
        return source_dirs, None

    temp_dir = tempfile.TemporaryDirectory(prefix="sdfvd2_raw_")
    temp_root = Path(temp_dir.name)
    mapped = {}
    for source, source_dir in source_dirs.items():
        link = temp_root / source
        link.symlink_to(source_dir.resolve(), target_is_directory=True)
        mapped[source] = link
    return mapped, temp_dir


def seed_from_legacy_processed_root(legacy_root: Path | None, processed_root: Path) -> int:
    if legacy_root is None or not legacy_root.exists():
        return 0

    copied_dirs = 0
    for split_name, source in RAW_SPLIT_TO_SOURCE.items():
        legacy_source_dir = legacy_root / split_name
        if not legacy_source_dir.is_dir():
            continue

        target_source_dir = processed_root / source
        target_source_dir.mkdir(parents=True, exist_ok=True)

        for legacy_video_dir in sorted(path for path in legacy_source_dir.iterdir() if path.is_dir()):
            target_video_dir = target_source_dir / legacy_video_dir.name
            if target_video_dir.is_dir():
                continue
            shutil.copytree(legacy_video_dir, target_video_dir, copy_function=link_or_copy)
            copied_dirs += 1

    return copied_dirs


def base_video_id(video_name: str) -> str:
    name = video_name
    if name.startswith("real_"):
        name = name[len("real_") :]
    elif name.startswith("fake_"):
        name = name[len("fake_") :]

    if "_aug_" in name:
        return name.split("_aug_", 1)[0]
    return name


def list_video_dirs(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []
    return sorted(path for path in source_dir.iterdir() if path.is_dir())


def assign_splits(base_ids: list[str], ratio: tuple[int, int, int], seed: int) -> dict[str, set[str]]:
    train_ratio, val_ratio, test_ratio = ratio
    rng = random.Random(seed)
    shuffled = sorted(base_ids)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio / sum(ratio))
    val_count = int(total * val_ratio / sum(ratio))
    test_count = total - train_count - val_count

    train_ids = set(shuffled[:train_count])
    val_ids = set(shuffled[train_count : train_count + val_count])
    test_ids = set(shuffled[train_count + val_count : train_count + val_count + test_count])

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def collect_split_files(processed_root: Path, ratio: tuple[int, int, int], seed: int):
    split_to_source_files = {
        split: {source: [] for source in SOURCES}
        for split in ("train", "val", "test", "all")
    }
    split_to_source_videos = {
        split: {source: 0 for source in SOURCES}
        for split in ("train", "val", "test", "all")
    }
    split_to_source_groups = {
        split: {source: 0 for source in SOURCES}
        for split in ("train", "val", "test")
    }

    source_to_group_names = {}

    for source in SOURCES:
        video_dirs = list_video_dirs(processed_root / source)
        groups = {}
        for video_dir in video_dirs:
            groups.setdefault(base_video_id(video_dir.name), []).append(video_dir)

        source_to_group_names[source] = sorted(groups)
        assignments = assign_splits(list(groups), ratio, seed)

        for split in ("train", "val", "test"):
            split_to_source_groups[split][source] = len(assignments[split])

        for group_name, group_video_dirs in groups.items():
            group_files = []
            for video_dir in sorted(group_video_dirs):
                video_files = sorted(str(path) for path in video_dir.rglob("*") if path.is_file())
                split_to_source_files["all"][source].extend(video_files)
                split_to_source_videos["all"][source] += 1
                group_files.extend(video_files)

            for split in ("train", "val", "test"):
                if group_name in assignments[split]:
                    split_to_source_files[split][source].extend(group_files)
                    split_to_source_videos[split][source] += len(group_video_dirs)
                    break

    return split_to_source_files, split_to_source_videos, split_to_source_groups, source_to_group_names


def write_file_list(paths: list[str], output_txt: Path) -> int:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")
    return len(paths)


def write_split_file_lists(config_root: Path, split_to_source_files: dict[str, dict[str, list[str]]]):
    counts = {}
    config_files = {}

    for split, source_to_files in split_to_source_files.items():
        counts[split] = {}
        config_files[split] = {}
        for source, files in source_to_files.items():
            output_txt = config_root / split / f"{source}.txt"
            counts[split][source] = write_file_list(files, output_txt)
            config_files[split][source] = str(output_txt)

    return counts, config_files


def verify_splits(split_to_source_files, split_to_source_groups, source_to_group_names):
    all_seen_files = set()
    all_seen_groups = {source: set() for source in SOURCES}

    for split in ("train", "val", "test"):
        split_files = set()
        for source in SOURCES:
            source_files = split_to_source_files[split][source]
            if len(source_files) != len(set(source_files)):
                raise ValueError(f"Duplicate frame paths found within {split}/{source}")

            overlap = split_files.intersection(source_files)
            if overlap:
                raise ValueError(f"Frame paths overlap inside split {split}")
            split_files.update(source_files)

            if all_seen_files.intersection(source_files):
                raise ValueError("Frame paths overlap across train/val/test")
            all_seen_files.update(source_files)

        assigned_groups = sum(split_to_source_groups[split].values())
        if assigned_groups == 0:
            raise ValueError(f"Split {split} is empty")

    for source in SOURCES:
        total_groups = len(source_to_group_names[source])
        assigned_groups = 0
        for split in ("train", "val", "test"):
            assigned_groups += split_to_source_groups[split][source]
        if assigned_groups != total_groups:
            raise ValueError(f"Group count mismatch for source {source}: {assigned_groups} != {total_groups}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("SDFVD2.0 Extension of Small Scale Deep Fake Video Dataset.zip"),
        help="Path to the SDFVD2.0 zip archive.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("raw_datasets/SDFVD2.0"),
        help="Where to extract or find raw SDFVD2.0 videos.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("datasets/SDFVD2.0"),
        help="Canonical processed dataset root.",
    )
    parser.add_argument(
        "--legacy-processed-root",
        type=Path,
        default=Path("datasets/SDFVD2.0_ext"),
        help="Optional legacy processed dataset root to seed canonical outputs from.",
    )
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--scale", type=float, default=1.3)
    parser.add_argument("--target-size", type=str, default="none")
    parser.add_argument("--det-thres", type=float, default=0.1)
    parser.add_argument("--mode", type=str, default="at_least")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-processed-videos", action="store_true")
    parser.add_argument("--skip-processed-frames", action="store_true")
    parser.add_argument("--split-ratio", type=str, default="80,10,10")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    if not args.zip_path.exists() and not args.skip_extract:
        raise FileNotFoundError(f"Zip archive not found: {args.zip_path}")

    ratio = parse_ratio(args.split_ratio)
    target_size = parse_target_size(args.target_size)

    if args.skip_extract and args.raw_root.exists():
        raw_source_dirs = detect_raw_source_dirs(args.raw_root)
        extracted_counts = count_raw_videos(raw_source_dirs)
    else:
        extracted_counts = extract_dataset(args.zip_path, args.raw_root)
        raw_source_dirs = detect_raw_source_dirs(args.raw_root)

    seeded_video_dirs = seed_from_legacy_processed_root(args.legacy_processed_root, args.processed_root)

    processing_source_dirs, temp_dir = prepare_processing_source_dirs(raw_source_dirs)
    try:
        model = prepare_model(args.det_thres)
        for source in SOURCES:
            process_mixed_types(
                input_folder_or_file=str(processing_source_dirs[source]),
                input_mask_folder=None,
                model=model,
                num_workers=args.num_workers,
                scale=args.scale,
                target_size=target_size,
                stride=args.stride,
                num_frames=args.num_frames,
                mode=args.mode,
                output_folder=str(args.processed_root),
                possible_extensions=("mp4",),
                skip_processed_videos=args.skip_processed_videos,
                skip_processed_frames=args.skip_processed_frames,
            )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    split_to_source_files, split_to_source_videos, split_to_source_groups, source_to_group_names = collect_split_files(
        args.processed_root, ratio, args.split_seed
    )
    verify_splits(split_to_source_files, split_to_source_groups, source_to_group_names)

    config_root = Path("config/datasets/SDFVD2.0")
    frame_counts, config_files = write_split_file_lists(config_root, split_to_source_files)

    summary = {
        "zip_path": str(args.zip_path),
        "raw_root": str(args.raw_root),
        "processed_root": str(args.processed_root),
        "legacy_processed_root": str(args.legacy_processed_root) if args.legacy_processed_root else None,
        "extracted_videos": extracted_counts,
        "seeded_video_dirs_from_legacy": seeded_video_dirs,
        "split": {
            "seed": args.split_seed,
            "ratio": {
                "train": ratio[0],
                "val": ratio[1],
                "test": ratio[2],
            },
            "grouping_rule": "strip leading real_/fake_; if _aug_ exists, use prefix before _aug_; otherwise use stem",
            "group_counts": split_to_source_groups,
            "base_video_counts": {source: len(source_to_group_names[source]) for source in SOURCES},
            "video_dir_counts": split_to_source_videos,
        },
        "processed_frames": frame_counts,
        "config_files": config_files,
        "detector_args": {
            "num_workers": args.num_workers,
            "scale": args.scale,
            "target_size": target_size,
            "det_thres": args.det_thres,
            "mode": args.mode,
            "stride": args.stride,
            "num_frames": args.num_frames,
        },
    }

    summary_path = config_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
