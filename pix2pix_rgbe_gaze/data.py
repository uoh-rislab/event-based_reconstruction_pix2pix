from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
USER_PATTERN = re.compile(r"^user_(\d+)$", re.IGNORECASE)
EXPERIMENT_PATTERN = re.compile(r"^exp(?:eriment)?_?(\d+)$", re.IGNORECASE)


def _normalize_user(value: str) -> str:
    match = re.fullmatch(r"(?:user_?)?(\d+)", str(value).strip().lower())
    return f"user_{int(match.group(1))}" if match else str(value).strip().lower()


def _normalize_experiment(value: str) -> str:
    match = re.fullmatch(
        r"(?:exp(?:eriment)?_?)?(\d+)", str(value).strip().lower()
    )
    return f"exp{int(match.group(1))}" if match else str(value).strip().lower()


def _identity(relative_path: Path) -> tuple[str, str]:
    user = "unknown"
    experiment = "unknown"
    for part in relative_path.parts[:-1]:
        user_match = USER_PATTERN.match(part)
        experiment_match = EXPERIMENT_PATTERN.match(part)
        if user_match:
            user = f"user_{int(user_match.group(1))}"
        if experiment_match:
            experiment = f"exp{int(experiment_match.group(1))}"
    return user, experiment


def _image_map(root: Path) -> dict[Path, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    return {
        path.relative_to(root): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def discover_pairs(
    dataset_root: str | Path,
    *,
    input_dir: str,
    target_dir: str,
    users: Iterable[str] | None = None,
    experiments: Iterable[str] | None = None,
    strict_pairs: bool = False,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    dataset_root = Path(dataset_root).resolve()
    inputs = _image_map(dataset_root / input_dir)
    targets = _image_map(dataset_root / target_dir)
    selected_users = {_normalize_user(value) for value in users or []}
    selected_experiments = {
        _normalize_experiment(value) for value in experiments or []
    }

    candidates = set(inputs) | set(targets)
    if selected_users or selected_experiments:
        candidates = {
            relative
            for relative in candidates
            if (
                not selected_users
                or _identity(relative)[0] in selected_users
            )
            and (
                not selected_experiments
                or _identity(relative)[1] in selected_experiments
            )
        }

    missing_inputs = sorted(candidates & (set(targets) - set(inputs)))
    missing_targets = sorted(candidates & (set(inputs) - set(targets)))
    if strict_pairs and (missing_inputs or missing_targets):
        raise ValueError(
            "The selected dataset is not fully paired: "
            f"missing inputs={len(missing_inputs)}, "
            f"missing targets={len(missing_targets)}"
        )

    rows: list[dict[str, str]] = []
    for relative in sorted(candidates & set(inputs) & set(targets)):
        user, experiment = _identity(relative)
        rows.append(
            {
                "input_path": str(inputs[relative]),
                "target_path": str(targets[relative]),
                "relative_path": str(relative),
                "filename": relative.name,
                "user": user,
                "experiment": experiment,
            }
        )
    if not rows:
        raise ValueError("No paired RGBE-Gaze images matched the requested split")
    summary = {
        "paired": len(rows),
        "skipped_without_input": len(missing_inputs),
        "skipped_without_target": len(missing_targets),
    }
    return rows, summary


class RGBEGazePairs(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        image_size: int,
    ) -> None:
        if not rows:
            raise ValueError("RGBEGazePairs cannot be empty")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.rows = rows
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, path: str) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("L")
            if image.size != (self.image_size, self.image_size):
                image = image.resize(
                    (self.image_size, self.image_size), Image.Resampling.LANCZOS
                )
            tensor = TF.to_tensor(image)
        return tensor.mul(2.0).sub(1.0)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        return {
            "input": self._load(row["input_path"]),
            "target": self._load(row["target_path"]),
            "relative_path": row["relative_path"],
            "filename": row["filename"],
            "user": row["user"],
            "experiment": row["experiment"],
        }

