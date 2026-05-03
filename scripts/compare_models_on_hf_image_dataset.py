from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import autorootcwd
except Exception:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from run import main
from src.config import Config, load_config


DATASET_NAME = "starganv2"
SOURCES = ("real", "fake")
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

HEADLINE_METRICS = [
    "auroc_frame",
    "mAP_frame",
    "acc_frame",
    "f1_score_frame",
    "eer_frame",
]

ALL_METRICS = HEADLINE_METRICS + [
    "auroc_video",
    "mAP_video",
    "acc_video",
    "f1_score_video",
    "eer_video",
]

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "hf_dino": {
        "label": "HF GenD DINOv3-L",
        "kind": "hf",
        "checkpoint": "yermandy/GenD_DINOv3_L",
    },
    "sdfvd2_dino_local": {
        "label": "SDFVD2.0-trained DINOv3-B",
        "kind": "local_run",
        "run_dir": "runs/sdfvd2/sdfvd2-DINOv3B-LN+L2+UA",
        "checkpoint": "runs/sdfvd2/sdfvd2-DINOv3B-LN+L2+UA/checkpoints/best_mAP.ckpt",
        "hparams": "runs/sdfvd2/sdfvd2-DINOv3B-LN+L2+UA/hparams.yaml",
    },
    "cddd_dino_local": {
        "label": "CDDD-trained DINOv3-B",
        "kind": "local_run",
        "run_dir": "runs/cddd/cddd-DINOv3B-LN+L2+UA",
        "checkpoint": "runs/cddd/cddd-DINOv3B-LN+L2+UA/checkpoints/best_mAP.ckpt",
        "hparams": "runs/cddd/cddd-DINOv3B-LN+L2+UA/hparams.yaml",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HF GenD DINOv3-L and a local GenD checkpoint on a local real/fake image zip."
    )
    parser.add_argument("--zip-path", type=Path, default=Path("starganv2.zip"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/starganv2"))
    parser.add_argument("--config-root", type=Path, default=Path("config/datasets/starganv2/all"))
    parser.add_argument("--comparison-name", default="starganv2-cddd-vs-hf-dino-compare")
    parser.add_argument("--output-root", type=Path, default=Path("runs/compare"))
    parser.add_argument("--model-a", default="hf_dino", choices=sorted(MODEL_PRESETS))
    parser.add_argument("--model-b", default="cddd_dino_local", choices=sorted(MODEL_PRESETS))
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild dataset/config files from the zip.")
    return parser.parse_args()


def ensure_paths_exist(args: argparse.Namespace) -> None:
    needs_zip = args.rebuild or not prepared_dataset_is_usable(args.dataset_root, args.config_root)
    if needs_zip and not args.zip_path.exists():
        raise FileNotFoundError(f"Zip archive not found: {args.zip_path}")

    local_presets = [MODEL_PRESETS[name] for name in (args.model_a, args.model_b) if MODEL_PRESETS[name]["kind"] == "local_run"]
    missing = [
        path
        for preset in local_presets
        for path in [preset["checkpoint"], preset["hparams"]]
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing local model assets: {missing}")


def write_file_list(paths: list[str], output_txt: Path) -> int:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")
    return len(paths)


def load_file_list(file_path: Path) -> list[str]:
    if not file_path.exists():
        return []
    return [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepared_dataset_is_usable(dataset_root: Path, config_root: Path) -> bool:
    summary_path = config_root / "summary.json"
    if not summary_path.exists():
        return False

    for source in SOURCES:
        file_list = load_file_list(config_root / f"{source}.txt")
        if not file_list:
            return False
        if not all(Path(path).exists() for path in file_list):
            return False

    return dataset_root.is_dir()


def reset_prepared_dataset(dataset_root: Path, config_root: Path) -> None:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    if config_root.parent.exists():
        shutil.rmtree(config_root.parent)


def prepare_dataset(zip_path: Path, dataset_root: Path, config_root: Path, rebuild: bool) -> dict[str, Any]:
    if rebuild:
        reset_prepared_dataset(dataset_root, config_root)
    elif prepared_dataset_is_usable(dataset_root, config_root):
        return json.loads((config_root / "summary.json").read_text(encoding="utf-8"))

    dataset_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)

    copied_counts = {source: 0 for source in SOURCES}
    ignored_non_images = 0
    skipped_invalid: list[dict[str, str | int]] = []
    seen_targets: set[Path] = set()

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            member_path = Path(info.filename)
            parts = member_path.parts

            if len(parts) < 3 or parts[0] != DATASET_NAME or parts[1] not in SOURCES:
                continue

            source = parts[1]
            suffix = member_path.suffix.lower()
            if suffix not in VALID_IMAGE_SUFFIXES:
                ignored_non_images += 1
                continue

            if info.file_size <= 0:
                skipped_invalid.append({"path": info.filename, "reason": "zero-byte", "size": info.file_size})
                continue

            target_dir = dataset_root / source / member_path.stem
            target_path = target_dir / f"frame_0000{suffix}"

            if target_path in seen_targets:
                skipped_invalid.append({"path": info.filename, "reason": "duplicate-target", "size": info.file_size})
                continue

            seen_targets.add(target_path)
            target_dir.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            copied_counts[source] += 1

    config_files: dict[str, str] = {}
    source_files: dict[str, list[str]] = {}
    for source in SOURCES:
        files = sorted(str(path) for path in (dataset_root / source).glob("*/frame_0000.*"))
        if not files:
            raise ValueError(f"No valid files prepared for source '{source}' from {zip_path}")
        source_files[source] = files
        config_file = config_root / f"{source}.txt"
        write_file_list(files, config_file)
        config_files[source] = str(config_file)

    summary = {
        "dataset_name": DATASET_NAME,
        "zip_path": str(zip_path),
        "dataset_root": str(dataset_root),
        "config_root": str(config_root),
        "config_files": config_files,
        "copied_images": copied_counts,
        "total_images": sum(copied_counts.values()),
        "ignored_non_images": ignored_non_images,
        "skipped_invalid": {
            "count": len(skipped_invalid),
            "files": skipped_invalid,
        },
    }
    (config_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_test_files(config_root: Path) -> dict[str, list[str]]:
    return {
        DATASET_NAME: [
            str(config_root / "fake.txt"),
            str(config_root / "real.txt"),
        ]
    }


def build_config(
    preset_name: str,
    run_name: str,
    output_root: Path,
    devices: str,
    num_workers: int | None,
    tst_files: dict[str, list[str]],
) -> Config:
    preset = MODEL_PRESETS[preset_name]
    kind = preset["kind"]

    if kind == "hf":
        config = Config(
            checkpoint=preset["checkpoint"],
            run_name=run_name,
            run_dir=str(output_root),
            wandb=False,
            devices=devices,
            tst_files=deepcopy(tst_files),
            throw_exception_if_run_exists=False,
            remove_if_run_exists=True,
            max_epochs=1,
        )
    elif kind == "local_run":
        config = load_config(preset["hparams"])
        config.run_name = run_name
        config.run_dir = str(output_root)
        config.checkpoint = preset["checkpoint"]
        config.wandb = False
        config.devices = devices
        config.tst_files = deepcopy(tst_files)
        config.throw_exception_if_run_exists = False
        config.remove_if_run_exists = True
    else:
        raise ValueError(f"Unknown preset kind: {kind}")

    if num_workers is not None:
        config.num_workers = num_workers

    return Config(**config.model_dump())


def latest_test_metrics(metrics_path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(metrics_path.open()))
    return next(row for row in reversed(rows) if row.get("test/auroc_frame"))


def collect_scope_metrics(row: dict[str, str], scope: str) -> dict[str, float | str]:
    prefix = "test" if scope == "overall" else f"test/dataset/{scope}"
    output: dict[str, float | str] = {"scope": scope}
    for metric in ALL_METRICS:
        value = row.get(f"{prefix}/{metric}")
        output[metric] = float(value) if value not in (None, "") else float("nan")
    return output


def compute_metric_deltas(results: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    if len(results) != 2:
        return []

    baseline, candidate = results
    deltas: list[dict[str, float | str]] = []
    for scope in ["overall", DATASET_NAME]:
        baseline_metrics = baseline["metrics"][scope]
        candidate_metrics = candidate["metrics"][scope]
        deltas.append(
            {
                "scope": scope,
                "baseline": baseline["label"],
                "candidate": candidate["label"],
                **{
                    f"delta_{metric}": candidate_metrics[metric] - baseline_metrics[metric]
                    for metric in ALL_METRICS
                },
            }
        )
    return deltas


def run_eval(
    preset_name: str,
    run_name: str,
    output_root: Path,
    devices: str,
    num_workers: int | None,
    tst_files: dict[str, list[str]],
) -> dict[str, Any]:
    config = build_config(preset_name, run_name, output_root, devices, num_workers, tst_files)
    main(config, train=False)

    run_dir = output_root / run_name
    row = latest_test_metrics(run_dir / "metrics.csv")
    scopes = {
        "overall": collect_scope_metrics(row, "overall"),
        DATASET_NAME: collect_scope_metrics(row, DATASET_NAME),
    }

    return {
        "preset": preset_name,
        "label": MODEL_PRESETS[preset_name]["label"],
        "run_dir": str(run_dir),
        "metrics": scopes,
    }


def save_summary(
    results: list[dict[str, Any]],
    dataset_summary: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    summary_csv = output_dir / "comparison_summary.csv"
    delta_csv = output_dir / "comparison_delta.csv"
    summary_json = output_dir / "comparison_summary.json"

    rows = []
    for result in results:
        for scope, metrics in result["metrics"].items():
            rows.append(
                {
                    "model": result["label"],
                    "preset": result["preset"],
                    "scope": scope,
                    **{metric: metrics[metric] for metric in ALL_METRICS},
                }
            )

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    deltas = compute_metric_deltas(results)
    with delta_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(deltas[0].keys()))
        writer.writeheader()
        writer.writerows(deltas)

    payload = {
        "dataset": dataset_summary,
        "models": results,
        "deltas": deltas,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_csv, delta_csv, summary_json


def print_summary(results: list[dict[str, Any]], dataset_summary: dict[str, Any]) -> None:
    print(
        "Prepared dataset:",
        f"real={dataset_summary['copied_images']['real']}",
        f"fake={dataset_summary['copied_images']['fake']}",
        f"total={dataset_summary['total_images']}",
        f"skipped_invalid={dataset_summary['skipped_invalid']['count']}",
    )

    for scope in ["overall", DATASET_NAME]:
        print(f"\n[{scope}] frame-first summary")
        header = f"{'model':28} {'auroc_f':>8} {'mAP_f':>8} {'acc_f':>8} {'f1_f':>8} {'eer_f':>8}"
        print(header)
        print("-" * len(header))
        for result in results:
            metrics = result["metrics"][scope]
            print(
                f"{result['label'][:28]:28} "
                f"{metrics['auroc_frame']:8.4f} "
                f"{metrics['mAP_frame']:8.4f} "
                f"{metrics['acc_frame']:8.4f} "
                f"{metrics['f1_score_frame']:8.4f} "
                f"{metrics['eer_frame']:8.4f}"
            )

    deltas = compute_metric_deltas(results)
    for delta in deltas:
        print(f"\n[{delta['scope']}] delta ({delta['candidate']} - {delta['baseline']})")
        header = f"{'metric':16} {'delta':>9}"
        print(header)
        print("-" * len(header))
        for metric in HEADLINE_METRICS:
            print(f"{metric:16} {delta[f'delta_{metric}']:9.4f}")

    print(f"\nExpected frame count: {dataset_summary['total_images']}")


def main_cli() -> None:
    args = parse_args()
    ensure_paths_exist(args)

    dataset_summary = prepare_dataset(args.zip_path, args.dataset_root, args.config_root, args.rebuild)
    tst_files = build_test_files(args.config_root)

    output_dir = args.output_root / args.comparison_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        run_eval(args.model_a, "model-a", output_dir, args.devices, args.num_workers, tst_files),
        run_eval(args.model_b, "model-b", output_dir, args.devices, args.num_workers, tst_files),
    ]

    summary_csv, delta_csv, summary_json = save_summary(results, dataset_summary, output_dir)
    print_summary(results, dataset_summary)
    print(f"\nSaved CSV summary to {summary_csv}")
    print(f"Saved delta CSV summary to {delta_csv}")
    print(f"Saved JSON summary to {summary_json}")
    for result in results:
        print(f"{result['label']} outputs: {result['run_dir']}")


if __name__ == "__main__":
    main_cli()
