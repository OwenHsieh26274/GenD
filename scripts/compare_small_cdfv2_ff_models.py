from __future__ import annotations

import argparse
import csv
import json
import sys
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


SMALL_BENCHMARK = {
    "CDFv2": [
        "config/datasets/CDFv2/test/Celeb-real.txt",
        "config/datasets/CDFv2/test/Celeb-synthesis.txt",
        "config/datasets/CDFv2/test/YouTube-real.txt",
    ],
    "FF": [
        "config/datasets/FF/test/DF.txt",
        "config/datasets/FF/test/F2F.txt",
        "config/datasets/FF/test/FS.txt",
        "config/datasets/FF/test/NT.txt",
        "config/datasets/FF/test/real.txt",
    ],
}

METRICS = [
    "auroc_frame",
    "auroc_video",
    "mAP_frame",
    "mAP_video",
    "acc_frame",
    "acc_video",
    "f1_score_frame",
    "f1_score_video",
    "eer_frame",
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two GenD models on the tiny local CDFv2 + FF benchmark."
    )
    parser.add_argument("--model-a", default="hf_dino", choices=sorted(MODEL_PRESETS))
    parser.add_argument("--model-b", default="sdfvd2_dino_local", choices=sorted(MODEL_PRESETS))
    parser.add_argument("--comparison-name", default="small-cdfv2-ff")
    parser.add_argument("--output-root", default="runs/compare")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def ensure_paths_exist() -> None:
    missing = [path for paths in SMALL_BENCHMARK.values() for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset list files: {missing}")


def build_config(
    preset_name: str,
    run_name: str,
    output_root: Path,
    devices: str,
    num_workers: int | None,
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
            tst_files=deepcopy(SMALL_BENCHMARK),
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
        config.tst_files = deepcopy(SMALL_BENCHMARK)
        config.throw_exception_if_run_exists = False
        config.remove_if_run_exists = True
    else:
        raise ValueError(f"Unknown preset kind: {kind}")

    if num_workers is not None:
        config.num_workers = num_workers

    return Config(**config.model_dump())


def latest_test_metrics(metrics_path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(metrics_path.open()))
    return next(row for row in reversed(rows) if row.get("test/auroc_video"))


def collect_scope_metrics(row: dict[str, str], scope: str) -> dict[str, float | str]:
    prefix = "test" if scope == "overall" else f"test/dataset/{scope}"
    output: dict[str, float | str] = {"scope": scope}
    for metric in METRICS:
        value = row.get(f"{prefix}/{metric}")
        output[metric] = float(value) if value not in (None, "") else float("nan")
    return output


def run_eval(
    preset_name: str,
    run_name: str,
    output_root: Path,
    devices: str,
    num_workers: int | None,
) -> dict[str, Any]:
    config = build_config(preset_name, run_name, output_root, devices, num_workers)
    main(config, train=False)

    run_dir = output_root / run_name
    row = latest_test_metrics(run_dir / "metrics.csv")
    scopes = {
        "overall": collect_scope_metrics(row, "overall"),
        "CDFv2": collect_scope_metrics(row, "CDFv2"),
        "FF": collect_scope_metrics(row, "FF"),
    }

    return {
        "preset": preset_name,
        "label": MODEL_PRESETS[preset_name]["label"],
        "run_dir": str(run_dir),
        "metrics": scopes,
    }


def save_summary(results: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    summary_csv = output_dir / "comparison_summary.csv"
    summary_json = output_dir / "comparison_summary.json"

    rows = []
    for result in results:
        for scope, metrics in result["metrics"].items():
            rows.append(
                {
                    "model": result["label"],
                    "preset": result["preset"],
                    "scope": scope,
                    **{metric: metrics[metric] for metric in METRICS},
                }
            )

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_json.write_text(json.dumps(results, indent=2))
    return summary_csv, summary_json


def print_summary(results: list[dict[str, Any]]) -> None:
    for scope in ["overall", "CDFv2", "FF"]:
        print(f"\n[{scope}]")
        header = f"{'model':28} {'auroc_v':>8} {'mAP_v':>8} {'acc_v':>8} {'f1_v':>8} {'eer_v':>8}"
        print(header)
        print("-" * len(header))
        for result in results:
            metrics = result["metrics"][scope]
            print(
                f"{result['label'][:28]:28} "
                f"{metrics['auroc_video']:8.4f} "
                f"{metrics['mAP_video']:8.4f} "
                f"{metrics['acc_video']:8.4f} "
                f"{metrics['f1_score_video']:8.4f} "
                f"{metrics['eer_video']:8.4f}"
            )


def main_cli() -> None:
    args = parse_args()
    ensure_paths_exist()

    output_dir = Path(args.output_root) / args.comparison_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        run_eval(args.model_a, "model-a", output_dir, args.devices, args.num_workers),
        run_eval(args.model_b, "model-b", output_dir, args.devices, args.num_workers),
    ]

    summary_csv, summary_json = save_summary(results, output_dir)
    print_summary(results)
    print(f"\nSaved CSV summary to {summary_csv}")
    print(f"Saved JSON summary to {summary_json}")
    for result in results:
        print(f"{result['label']} outputs: {result['run_dir']}")


if __name__ == "__main__":
    main_cli()
