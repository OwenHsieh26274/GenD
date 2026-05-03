import argparse
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


SOURCES = ("real", "fake")
REAL_RE = re.compile(r"^(?P<gender>male|female)-Vid-(?P<vid>\d+)_frame_(?P<frame>\d+)\.jpg$")
FAKE_RE = re.compile(
    r"^(?P<person>.+)_(?P<gender>male|female)-Vid-(?P<vid>\d+)_Tech-(?P<tech>\d+)_frame_(?P<frame>\d+)\.jpg$"
)


@dataclass(frozen=True)
class Record:
    source: str
    gender: str
    vid: str
    frame: str
    src_path: str
    dst_path: str
    group_key: str
    video_name: str
    person: str | None = None
    tech: str | None = None


def parse_ratio(raw: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in raw.split(","))
    if len(values) not in (2, 3):
        raise ValueError("--split-ratio must contain two or three comma-separated integers.")
    if sum(values) <= 0:
        raise ValueError("--split-ratio must sum to a positive value.")
    if any(value < 0 for value in values):
        raise ValueError("--split-ratio values must be non-negative.")
    return values


def split_names_for_ratio(ratio: tuple[int, ...]) -> tuple[str, ...]:
    return ("train", "val") if len(ratio) == 2 else ("train", "val", "test")


def assign_splits(group_keys: list[str], ratio: tuple[int, ...], seed: int) -> dict[str, set[str]]:
    rng = random.Random(seed)
    shuffled = sorted(group_keys)
    rng.shuffle(shuffled)

    total = len(shuffled)
    split_names = split_names_for_ratio(ratio)
    ratio_sum = sum(ratio)

    assignments = {}
    start = 0
    for split, split_ratio in zip(split_names[:-1], ratio[:-1]):
        count = int(total * split_ratio / ratio_sum)
        assignments[split] = set(shuffled[start : start + count])
        start += count
    assignments[split_names[-1]] = set(shuffled[start:])
    return assignments


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if mode == "hardlink" and dst.exists() and os.path.samefile(src, dst):
            return "reused"
        dst.unlink()

    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported link mode: {mode}")
    return "created"


def collect_real_records(raw_root: Path, processed_root: Path) -> tuple[list[Record], list[str]]:
    records = []
    ignored = []
    real_root = raw_root / "real"

    for path in sorted(real_root.rglob("*")):
        if not path.is_file():
            continue
        match = REAL_RE.match(path.name)
        if not match:
            ignored.append(str(path))
            continue

        fields = match.groupdict()
        gender = fields["gender"]
        if path.parent.name != gender:
            raise ValueError(f"Real gender mismatch: {path}")

        group_key = f"{gender}_Vid-{fields['vid']}"
        video_name = group_key
        dst_path = processed_root / "real" / video_name / path.name
        records.append(
            Record(
                source="real",
                gender=gender,
                vid=fields["vid"],
                frame=fields["frame"],
                src_path=str(path),
                dst_path=str(dst_path),
                group_key=group_key,
                video_name=video_name,
            )
        )

    return records, ignored


def collect_fake_records(raw_root: Path, processed_root: Path) -> tuple[list[Record], list[str]]:
    records = []
    ignored = []
    fake_root = raw_root / "fake"

    for path in sorted(fake_root.rglob("*")):
        if not path.is_file():
            continue
        match = FAKE_RE.match(path.name)
        if not match:
            ignored.append(str(path))
            continue

        fields = match.groupdict()
        gender = fields["gender"]
        person = fields["person"]
        if path.parent.name != person or path.parent.parent.name != gender:
            raise ValueError(f"Fake path/name mismatch: {path}")

        group_key = f"{gender}_Vid-{fields['vid']}"
        video_name = f"{person}_{gender}_Vid-{fields['vid']}_Tech-{fields['tech']}"
        dst_path = processed_root / "fake" / video_name / path.name
        records.append(
            Record(
                source="fake",
                gender=gender,
                vid=fields["vid"],
                frame=fields["frame"],
                src_path=str(path),
                dst_path=str(dst_path),
                group_key=group_key,
                video_name=video_name,
                person=person,
                tech=fields["tech"],
            )
        )

    return records, ignored


def collect_records(raw_root: Path, processed_root: Path) -> tuple[list[Record], list[str]]:
    for source in SOURCES:
        if not (raw_root / source).is_dir():
            raise FileNotFoundError(f"Missing CDDD source directory: {raw_root / source}")

    top_level_ignored = [
        str(path)
        for path in sorted(raw_root.iterdir())
        if path.is_file()
    ]
    real_records, real_ignored = collect_real_records(raw_root, processed_root)
    fake_records, fake_ignored = collect_fake_records(raw_root, processed_root)
    records = real_records + fake_records
    if not real_records:
        raise ValueError(f"No real CDDD records found under {raw_root / 'real'}")
    if not fake_records:
        raise ValueError(f"No fake CDDD records found under {raw_root / 'fake'}")
    return records, sorted(top_level_ignored + real_ignored + fake_ignored)


def materialize_records(records: list[Record], link_mode: str) -> dict[str, int]:
    counts = {"created": 0, "reused": 0}
    for record in records:
        status = link_or_copy(Path(record.src_path), Path(record.dst_path), link_mode)
        counts[status] += 1
    return counts


def split_records(
    records: list[Record],
    assignments: dict[str, set[str]],
) -> dict[str, dict[str, list[str]]]:
    split_names = tuple(assignments)
    split_to_source_files = {
        split: {source: [] for source in SOURCES}
        for split in (*split_names, "all")
    }

    for record in records:
        split_to_source_files["all"][record.source].append(record.dst_path)
        for split in split_names:
            if record.group_key in assignments[split]:
                split_to_source_files[split][record.source].append(record.dst_path)
                break
        else:
            raise ValueError(f"Record group was not assigned to a split: {record.group_key}")

    for source_to_files in split_to_source_files.values():
        for source, files in source_to_files.items():
            source_to_files[source] = sorted(files)

    return split_to_source_files


def remove_stale_split_dirs(config_root: Path, split_names: tuple[str, ...]):
    if not config_root.exists():
        return
    active = set(split_names) | {"all"}
    for split_dir in config_root.iterdir():
        if split_dir.is_dir() and split_dir.name not in active:
            shutil.rmtree(split_dir)


def write_file_lists(config_root: Path, split_to_source_files: dict[str, dict[str, list[str]]]):
    counts = {}
    config_files = {}

    for split, source_to_files in split_to_source_files.items():
        counts[split] = {}
        config_files[split] = {}
        for source, files in source_to_files.items():
            output_path = config_root / split / f"{source}.txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("\n".join(files) + ("\n" if files else ""), encoding="utf-8")
            counts[split][source] = len(files)
            config_files[split][source] = str(output_path)

    return counts, config_files


def verify(records: list[Record], split_to_source_files: dict[str, dict[str, list[str]]], assignments: dict[str, set[str]]):
    split_names = tuple(assignments)
    all_split_files = set()
    for split in split_names:
        split_files = set()
        for source in SOURCES:
            files = split_to_source_files[split][source]
            if len(files) != len(set(files)):
                raise ValueError(f"Duplicate paths found in {split}/{source}")
            missing = [path for path in files if not Path(path).exists()]
            if missing:
                raise FileNotFoundError(f"Missing prepared files in {split}/{source}: {missing[:5]}")
            overlap = split_files.intersection(files)
            if overlap:
                raise ValueError(f"Path overlap inside split {split}: {sorted(overlap)[:5]}")
            split_files.update(files)

        overlap = all_split_files.intersection(split_files)
        if overlap:
            raise ValueError(f"Path overlap across splits: {sorted(overlap)[:5]}")
        all_split_files.update(split_files)

    assigned_groups = set().union(*(assignments[split] for split in split_names))
    record_groups = {record.group_key for record in records}
    if assigned_groups != record_groups:
        raise ValueError("Split assignments do not match record groups.")

    group_to_splits = {}
    for split in split_names:
        for group_key in assignments[split]:
            group_to_splits.setdefault(group_key, set()).add(split)
    leaking = {group_key: splits for group_key, splits in group_to_splits.items() if len(splits) > 1}
    if leaking:
        raise ValueError(f"Groups assigned to multiple splits: {leaking}")


def summarize(
    raw_root: Path,
    processed_root: Path,
    config_root: Path,
    link_mode: str,
    split_ratio: tuple[int, ...],
    split_seed: int,
    records: list[Record],
    ignored_files: list[str],
    materialized_counts: dict[str, int],
    assignments: dict[str, set[str]],
    frame_counts: dict[str, dict[str, int]],
    config_files: dict[str, dict[str, str]],
) -> dict:
    source_counts = {source: sum(1 for record in records if record.source == source) for source in SOURCES}
    gender_counts = {
        source: {
            gender: sum(1 for record in records if record.source == source and record.gender == gender)
            for gender in ("female", "male")
        }
        for source in SOURCES
    }
    video_dir_counts = {
        source: len({record.video_name for record in records if record.source == source})
        for source in SOURCES
    }

    fake_to_real = source_counts["fake"] / source_counts["real"] if source_counts["real"] else None
    split_names = tuple(assignments)

    return {
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "config_root": str(config_root),
        "link_mode": link_mode,
        "materialized_files": materialized_counts,
        "ignored_files": {
            "count": len(ignored_files),
            "files": ignored_files,
        },
        "source_counts": source_counts,
        "gender_counts": gender_counts,
        "video_dir_counts": video_dir_counts,
        "class_imbalance": {
            "fake_to_real": fake_to_real,
        },
        "split": {
            "seed": split_seed,
            "ratio": {split: ratio for split, ratio in zip(split_names, split_ratio)},
            "grouping_rule": "gender+Vid; all real/fake/person/Tech records with the same group stay in one split",
            "group_counts": {split: len(assignments[split]) for split in split_names},
            "groups": {split: sorted(assignments[split]) for split in split_names},
        },
        "processed_frames": frame_counts,
        "config_files": config_files,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare CDDD images for GenD DeepfakeDataset.")
    parser.add_argument("--raw-root", type=Path, default=Path("raw_data/CDDD"))
    parser.add_argument("--processed-root", type=Path, default=Path("datasets/CDDD"))
    parser.add_argument("--config-root", type=Path, default=Path("config/datasets/CDDD"))
    parser.add_argument("--link-mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    parser.add_argument("--split-ratio", type=str, default="80,20")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--summary-name",
        type=str,
        default="summary.json",
        help="Summary filename written under --config-root.",
    )
    args = parser.parse_args()

    split_ratio = parse_ratio(args.split_ratio)
    records, ignored_files = collect_records(args.raw_root, args.processed_root)
    assignments = assign_splits(sorted({record.group_key for record in records}), split_ratio, args.split_seed)
    remove_stale_split_dirs(args.config_root, split_names_for_ratio(split_ratio))
    materialized_counts = materialize_records(records, args.link_mode)
    split_to_source_files = split_records(records, assignments)
    verify(records, split_to_source_files, assignments)
    frame_counts, config_files = write_file_lists(args.config_root, split_to_source_files)

    summary = summarize(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        config_root=args.config_root,
        link_mode=args.link_mode,
        split_ratio=split_ratio,
        split_seed=args.split_seed,
        records=records,
        ignored_files=ignored_files,
        materialized_counts=materialized_counts,
        assignments=assignments,
        frame_counts=frame_counts,
        config_files=config_files,
    )

    summary_path = args.config_root / args.summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
