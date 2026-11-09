"""Grad-CAM overlays for the DenseNet-121 chest X-ray model."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from torchvision import transforms

from src.data.datasets import IMAGENET_MEAN, IMAGENET_STD


def last_conv_layer(module: nn.Module) -> nn.Conv2d:
    """Grad-CAM reads the last convolution. For DenseNet that is inside the final dense block."""
    last = None
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            last = layer
    if last is None:
        raise RuntimeError("No convolutional layer was found for Grad-CAM.")
    return last


def _display_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize((image_size, image_size))
        return np.asarray(resized).astype(np.float32) / 255.0


def _tensor(path: Path, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


def _choose_rows(frame: pd.DataFrame, probabilities: np.ndarray, limit: int) -> list[int]:
    order = np.argsort(probabilities)
    half = max(1, limit // 2)
    high = list(order[-half:][::-1])
    low = list(order[:half])
    chosen = []
    for index in high + low:
        index = int(index)
        if index not in chosen:
            chosen.append(index)
        if len(chosen) >= limit:
            break
    return chosen


def save_gradcam(model, frame: pd.DataFrame, probabilities: np.ndarray, root: Path, image_size: int, device, output_dir: Path, limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    target_layer = last_conv_layer(model.net)
    cam = GradCAM(model=model, target_layers=[target_layer])
    for rank, index in enumerate(_choose_rows(frame, probabilities, limit), start=1):
        row = frame.iloc[index]
        stored = Path(row.image_path)
        path = stored if stored.is_absolute() else root / stored
        tensor = _tensor(path, image_size).unsqueeze(0).to(device)
        grayscale = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(0)])[0]
        original = _display_image(path, image_size)
        overlay = show_cam_on_image(original, grayscale, use_rgb=True)

        figure, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(original)
        axes[0].set_title("X-ray")
        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM on the mortality logit")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(
            f"hadm_id={int(row.hadm_id)}  true={int(row.hospital_expire_flag)}  "
            f"predicted={probabilities[index]:.2f}"
        )
        figure.tight_layout()
        out = output_dir / f"gradcam_{rank}_hadm_{int(row.hadm_id)}.png"
        figure.savefig(out, dpi=140)
        plt.close(figure)
        print(f"Wrote {out.name}")
