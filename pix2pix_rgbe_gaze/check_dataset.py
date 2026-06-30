#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import RGBEGazePairs, discover_pairs
from train import REPRESENTATIONS, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate RGBE-Gaze pairs and configured splits"
    )
    parser.add_argument("--device", choices=("dgx-1",), default="dgx-1")
    parser.add_argument(
        "--rep", choices=tuple(REPRESENTATIONS), default="acc_events"
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--strict-pairs", action="store_true")
    user_group = parser.add_mutually_exclusive_group()
    user_group.add_argument("--users", nargs="+")
    user_group.add_argument("--all-users", action="store_true")
    args = parser.parse_args()

    config = load_config(args.device)
    dataset_root = (args.dataset_root or Path(config["DATASET_ROOT"])).resolve()
    selected_users = None if args.all_users else (args.users or config.get("USERS"))
    report: dict[str, object] = {
        "status": "ok",
        "dataset_root": str(dataset_root),
        "representation": args.rep,
        "users": selected_users if selected_users is not None else "all",
        "splits": {},
    }
    for split in ("train", "val", "test"):
        rows, summary = discover_pairs(
            dataset_root,
            input_dir=REPRESENTATIONS[args.rep],
            target_dir="gray_frames",
            users=selected_users,
            experiments=config["SPLITS"][split],
            strict_pairs=args.strict_pairs,
        )
        dataset = RGBEGazePairs(rows, image_size=int(config["IMAGE_SIZE"]))
        sample = dataset[0]
        report["splits"][split] = {
            **summary,
            "first_relative_path": sample["relative_path"],
            "input_shape": list(sample["input"].shape),
            "target_shape": list(sample["target"].shape),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
