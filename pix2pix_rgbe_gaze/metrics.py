from __future__ import annotations

import math

import cv2
import numpy as np
import torch


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().float().cpu().clamp(-1.0, 1.0).add(1.0).div(2.0).numpy()
    )


def image_metrics(target: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float32).squeeze()
    generated = np.asarray(generated, dtype=np.float32).squeeze()
    mse = float(np.mean((target - generated) ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * math.log10(1.0 / mse))

    c1 = 0.01**2
    c2 = 0.03**2
    mu_target = cv2.GaussianBlur(target, (11, 11), 1.5)
    mu_generated = cv2.GaussianBlur(generated, (11, 11), 1.5)
    target_variance = (
        cv2.GaussianBlur(target * target, (11, 11), 1.5)
        - mu_target * mu_target
    )
    generated_variance = (
        cv2.GaussianBlur(generated * generated, (11, 11), 1.5)
        - mu_generated * mu_generated
    )
    covariance = (
        cv2.GaussianBlur(target * generated, (11, 11), 1.5)
        - mu_target * mu_generated
    )
    numerator = (2 * mu_target * mu_generated + c1) * (2 * covariance + c2)
    denominator = (
        (mu_target * mu_target + mu_generated * mu_generated + c1)
        * (target_variance + generated_variance + c2)
        + 1e-12
    )
    return {"mse": mse, "ssim": float((numerator / denominator).mean()), "psnr": psnr}

