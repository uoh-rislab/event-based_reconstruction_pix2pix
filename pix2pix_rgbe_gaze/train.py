#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import RGBEGazePairs, discover_pairs
from metrics import image_metrics, tensor_to_numpy
from models import PatchDiscriminator, Pix2PixGenerator, initialize_weights


REPRESENTATIONS = {"acc_events": "event_accumulate_frames"}


def load_config(device_name: str) -> dict:
    path = HERE / "yaml" / f"config_{device_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["config_path"] = str(path)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def raw_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def set_requires_grad(model: nn.Module, enabled: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


def make_loader(
    dataset: RGBEGazePairs,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    use_cuda: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=use_cuda,
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


@torch.no_grad()
def validate(
    generator: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None,
) -> dict[str, float]:
    generator.eval()
    totals = {"mse": 0.0, "ssim": 0.0, "psnr": 0.0}
    samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        generated = generator(inputs)
        for target, prediction in zip(targets, generated):
            values = image_metrics(
                tensor_to_numpy(target), tensor_to_numpy(prediction)
            )
            for name in totals:
                totals[name] += values[name]
            samples += 1
    generator.train()
    if samples == 0:
        raise ValueError("Validation loader produced no samples")
    return {name: value / samples for name, value in totals.items()} | {
        "samples": samples
    }


@torch.no_grad()
def save_preview(
    generator: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
) -> None:
    generator.eval()
    batch = next(iter(loader))
    inputs = batch["input"][:1].to(device)
    targets = batch["target"][:1].to(device)
    generated = generator(inputs)
    panels = torch.cat((inputs, generated, targets), dim=0).cpu().add(1.0).div(2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(panels, output_path, nrow=3, padding=4, pad_value=0.0)
    generator.train()


def checkpoint_state(
    *,
    epoch: int,
    best_mse: float,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: Adam,
    optimizer_d: Adam,
    scaler: torch.cuda.amp.GradScaler,
    config: dict,
) -> dict:
    return {
        "epoch": epoch,
        "best_mse": best_mse,
        "generator": raw_model(generator).state_dict(),
        "discriminator": raw_model(discriminator).state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Pix2Pix for RGBE-Gaze accumulated events"
    )
    parser.add_argument("--device", choices=("dgx-1",), default="dgx-1")
    parser.add_argument(
        "--rep",
        choices=tuple(REPRESENTATIONS),
        default="acc_events",
        help="Input event representation",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="Visible GPU IDs, for example: --gpu 0 or --gpu 0,1",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path)
    user_group = parser.add_mutually_exclusive_group()
    user_group.add_argument(
        "--users",
        nargs="+",
        help="Users to include, for example: --users user_1 user_2",
    )
    user_group.add_argument(
        "--all-users",
        action="store_true",
        help="Include every user found under the configured experiments",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = load_config(args.device)
    seed_everything(int(config.get("SEED", 44)))
    dataset_root = (args.dataset_root or Path(config["DATASET_ROOT"])).resolve()
    output_root = Path(config["OUTPUT_ROOT"]).resolve()
    selected_users = None if args.all_users else (args.users or config.get("USERS"))
    user_label = (
        "all-users"
        if selected_users is None
        else "-".join(str(user).replace("_", "-") for user in selected_users)
    )
    run_name = args.run_name or (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_pix2pix_rgbe_gaze_{args.rep}_{user_label}"
    )
    output_dir = output_root / run_name
    if output_dir.exists() and args.resume is None:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Choose another --run-name or provide --resume."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    split_experiments = config["SPLITS"]
    datasets: dict[str, RGBEGazePairs] = {}
    pair_summaries: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        rows, summary = discover_pairs(
            dataset_root,
            input_dir=REPRESENTATIONS[args.rep],
            target_dir="gray_frames",
            users=selected_users,
            experiments=split_experiments[split],
            strict_pairs=bool(config.get("STRICT_PAIRS", False)),
        )
        datasets[split] = RGBEGazePairs(
            rows, image_size=int(config["IMAGE_SIZE"])
        )
        pair_summaries[split] = summary

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    loaders = {
        split: make_loader(
            dataset,
            batch_size=int(config["BATCH_SIZE"]),
            workers=int(config.get("NUM_WORKERS", 4)),
            shuffle=split == "train",
            use_cuda=use_cuda,
        )
        for split, dataset in datasets.items()
    }

    generator: nn.Module = Pix2PixGenerator()
    discriminator: nn.Module = PatchDiscriminator()
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    if use_cuda and torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    optimizer_g = Adam(
        generator.parameters(),
        lr=float(config["LEARNING_RATE"]),
        betas=(float(config["BETA1"]), 0.999),
    )
    optimizer_d = Adam(
        discriminator.parameters(),
        lr=float(config["LEARNING_RATE"]),
        betas=(float(config["BETA1"]), 0.999),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(config.get("MIXED_PRECISION", True)) and use_cuda
    )
    adversarial_loss = nn.BCEWithLogitsLoss()
    reconstruction_loss = nn.L1Loss()
    start_epoch = 0
    best_mse = float("inf")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        raw_model(generator).load_state_dict(checkpoint["generator"])
        raw_model(discriminator).load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_mse = float(checkpoint.get("best_mse", float("inf")))

    resolved = {
        **config,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "representation": args.rep,
        "users": selected_users if selected_users is not None else "all",
        "visible_gpus": args.gpu,
        "pair_summaries": pair_summaries,
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
    )
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists() or start_epoch == 0:
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                ("epoch", "d_loss", "g_loss", "l1_loss", "val_mse", "val_ssim", "val_psnr")
            )

    print(f"Dataset root: {dataset_root}")
    print(f"Output directory: {output_dir}")
    print(f"CUDA_VISIBLE_DEVICES={args.gpu}")
    print(f"PyTorch devices: {torch.cuda.device_count()} | training device: {device}")
    print(
        "Samples: "
        + " | ".join(f"{split}={len(dataset)}" for split, dataset in datasets.items())
    )

    epochs = int(config["EPOCHS"])
    lambda_l1 = float(config.get("LAMBDA_L1", 100.0))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, epochs):
        generator.train()
        discriminator.train()
        totals = {"d": 0.0, "g": 0.0, "l1": 0.0}
        batches = 0
        progress = tqdm(loaders["train"], desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            set_requires_grad(discriminator, True)
            optimizer_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                generated = generator(inputs)
                real_logits = discriminator(inputs, targets)
                fake_logits = discriminator(inputs, generated.detach())
                loss_d = 0.5 * (
                    adversarial_loss(real_logits, torch.ones_like(real_logits))
                    + adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
                )
            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)

            set_requires_grad(discriminator, False)
            optimizer_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                generated = generator(inputs)
                fake_logits = discriminator(inputs, generated)
                loss_gan = adversarial_loss(fake_logits, torch.ones_like(fake_logits))
                loss_l1 = reconstruction_loss(generated, targets)
                loss_g = loss_gan + lambda_l1 * loss_l1
            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()

            totals["d"] += float(loss_d.detach())
            totals["g"] += float(loss_g.detach())
            totals["l1"] += float(loss_l1.detach())
            batches += 1
            progress.set_postfix(
                d=f"{totals['d'] / batches:.4f}",
                g=f"{totals['g'] / batches:.4f}",
                l1=f"{totals['l1'] / batches:.4f}",
            )

        val_metrics = validate(
            generator,
            loaders["val"],
            device,
            max_batches=config.get("MAX_VALIDATION_BATCHES"),
        )
        average = {name: value / max(batches, 1) for name, value in totals.items()}
        print(
            f"Epoch {epoch + 1}: D={average['d']:.6f} | G={average['g']:.6f} | "
            f"L1={average['l1']:.6f} | val={val_metrics}"
        )
        with metrics_path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                (
                    epoch + 1,
                    average["d"],
                    average["g"],
                    average["l1"],
                    val_metrics["mse"],
                    val_metrics["ssim"],
                    val_metrics["psnr"],
                )
            )

        is_best = val_metrics["mse"] <= best_mse
        best_mse = min(best_mse, val_metrics["mse"])
        state = checkpoint_state(
            epoch=epoch,
            best_mse=best_mse,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scaler=scaler,
            config=resolved,
        )
        torch.save(state, checkpoint_dir / "last.pt")
        if is_best:
            torch.save(state, checkpoint_dir / "best.pt")
        if (epoch + 1) % int(config["SAVE_EVERY_N_EPOCHS"]) == 0:
            save_preview(
                generator,
                loaders["val"],
                device,
                output_dir / "previews" / f"epoch_{epoch + 1:04d}.png",
            )
            torch.save(
                raw_model(generator).state_dict(),
                checkpoint_dir / f"generator_epoch_{epoch + 1:04d}.pth",
            )


if __name__ == "__main__":
    main()
