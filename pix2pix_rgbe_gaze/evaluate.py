#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import RGBEGazePairs, discover_pairs
from metrics import image_metrics, tensor_to_numpy
from models import Pix2PixGenerator
from train import REPRESENTATIONS, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Pix2Pix RGBE-Gaze reconstructions"
    )
    parser.add_argument("--device", choices=("dgx-1",), default="dgx-1")
    parser.add_argument(
        "--rep", choices=tuple(REPRESENTATIONS), default="acc_events"
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty evaluation directory",
    )
    user_group = parser.add_mutually_exclusive_group()
    user_group.add_argument("--users", nargs="+")
    user_group.add_argument("--all-users", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = load_config(args.device)
    dataset_root = (args.dataset_root or Path(config["DATASET_ROOT"])).resolve()
    selected_users = None if args.all_users else (args.users or config.get("USERS"))
    rows, pair_summary = discover_pairs(
        dataset_root,
        input_dir=REPRESENTATIONS[args.rep],
        target_dir="gray_frames",
        users=selected_users,
        experiments=config["SPLITS"][args.split],
        strict_pairs=bool(config.get("STRICT_PAIRS", False)),
    )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    dataset = RGBEGazePairs(rows, image_size=int(config["IMAGE_SIZE"]))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config.get("NUM_WORKERS", 4)),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    generator = Pix2PixGenerator()
    state = checkpoint["generator"] if "generator" in checkpoint else checkpoint
    generator.load_state_dict(state)
    generator = generator.to(device).eval()
    output_dir = args.output_dir.resolve()
    if output_dir.is_dir() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Evaluation directory is not empty: {output_dir}. "
            "Choose another --output-dir or add --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            generated = generator(inputs)
            values = image_metrics(
                tensor_to_numpy(targets[0]), tensor_to_numpy(generated[0])
            )
            relative = Path(batch["relative_path"][0])
            sample_dir = output_dir / relative.parent
            sample_dir.mkdir(parents=True, exist_ok=True)
            suffix = relative.name
            normalized_input = inputs[0].cpu().add(1.0).div(2.0)
            normalized_generated = generated[0].cpu().add(1.0).div(2.0)
            normalized_target = targets[0].cpu().add(1.0).div(2.0)
            save_image(normalized_input, sample_dir / f"event_{suffix}")
            save_image(normalized_generated, sample_dir / f"generated_{suffix}")
            save_image(normalized_target, sample_dir / f"target_{suffix}")
            save_image(
                torch.stack(
                    (normalized_input, normalized_generated, normalized_target)
                ),
                sample_dir / f"comparison_{suffix}",
                nrow=3,
                padding=4,
            )
            metric_rows.append({"relative_path": str(relative), **values})

    metrics_path = output_dir / "per_image_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("relative_path", "mse", "ssim", "psnr")
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "users": selected_users if selected_users is not None else "all",
        "samples": len(metric_rows),
        "pair_summary": pair_summary,
    }
    for metric in ("mse", "ssim", "psnr"):
        values = np.asarray(
            [float(row[metric]) for row in metric_rows if math.isfinite(float(row[metric]))]
        )
        summary[metric] = {
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
            "non_finite": len(metric_rows) - int(values.size),
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
