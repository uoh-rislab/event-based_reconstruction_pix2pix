#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import RGBEGazePairs, discover_pairs
from metrics import tensor_to_numpy
from models import Pix2PixGenerator
from train import REPRESENTATIONS, load_config


LABELS = ("Input (event)", "Generated", "Target")


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor_to_numpy(tensor).squeeze()
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="L").convert(
        "RGB"
    )


def horizontal_comparison(
    panels: list[Image.Image],
    *,
    label_height: int,
    gap: int,
) -> Image.Image:
    sizes = {panel.size for panel in panels}
    if len(sizes) != 1:
        raise ValueError(f"Panel dimensions do not match: {sorted(sizes)}")
    width, height = panels[0].size
    canvas = Image.new(
        "RGB",
        (width * len(panels) + gap * (len(panels) - 1), height + label_height),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(12, label_height - 12))
    for index, (panel, label) in enumerate(zip(panels, LABELS)):
        x = index * (width + gap)
        canvas.paste(panel, (x, label_height))
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        draw.text(
            (
                x + (width - text_width) / 2,
                (label_height - text_height) / 2 - box[1],
            ),
            label,
            fill="white",
            font=font,
        )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate horizontal Pix2Pix RGBE-Gaze comparisons"
    )
    parser.add_argument("--device", choices=("dgx-1",), default="dgx-1")
    parser.add_argument(
        "--rep", choices=tuple(REPRESENTATIONS), default="acc_events"
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--label-height", type=int, default=44)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    user_group = parser.add_mutually_exclusive_group()
    user_group.add_argument("--users", nargs="+")
    user_group.add_argument("--all-users", action="store_true")
    args = parser.parse_args()

    if args.limit < 1 or args.offset < 0:
        raise ValueError("--limit must be positive and --offset must be non-negative")
    if args.label_height < 1 or args.gap < 0 or args.num_workers < 0:
        raise ValueError(
            "--label-height must be positive; --gap and --num-workers must be non-negative"
        )

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
    rows = rows[args.offset : args.offset + args.limit]
    if not rows:
        raise ValueError(
            f"No samples remain after applying offset={args.offset} and limit={args.limit}"
        )
    dataset = RGBEGazePairs(rows, image_size=int(config["IMAGE_SIZE"]))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    generator = Pix2PixGenerator()
    state = checkpoint["generator"] if "generator" in checkpoint else checkpoint
    generator.load_state_dict(state)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device).eval()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else checkpoint_path.parent.parent
        / "samples"
        / f"{args.split}-{checkpoint_path.stem}"
    )
    if output_dir.is_dir() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Sample directory is not empty: {output_dir}. "
            "Choose another --output-dir or add --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_count = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Sampling {args.split}"):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            generated = generator(inputs)
            relative = Path(batch["relative_path"][0])
            sample_dir = output_dir / relative.parent
            sample_dir.mkdir(parents=True, exist_ok=True)
            comparison = horizontal_comparison(
                [
                    tensor_to_pil(inputs[0]),
                    tensor_to_pil(generated[0]),
                    tensor_to_pil(targets[0]),
                ],
                label_height=args.label_height,
                gap=args.gap,
            )
            comparison.save(sample_dir / f"comparison_{relative.name}")
            generated_count += 1

    metadata = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(dataset_root),
        "representation": args.rep,
        "split": args.split,
        "users": selected_users if selected_users is not None else "all",
        "offset": args.offset,
        "samples": generated_count,
        "pair_summary": pair_summary,
        "output_dir": str(output_dir),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated {generated_count} horizontal comparisons in {output_dir}")


if __name__ == "__main__":
    main()
