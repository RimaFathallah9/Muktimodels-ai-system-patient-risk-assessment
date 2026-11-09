"""Pretrained image classifiers. The embedding is the vector before the final linear layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn

from src.data.datasets import ImageAdmissionDataset, image_transform, make_loader
from src.models.loop import collect_embeddings, collect_probabilities, fit_model


class ImageClassifier(nn.Module):
    def __init__(self, timm_name: str, pretrained: bool) -> None:
        super().__init__()
        self.timm_name = timm_name
        self.net = timm.create_model(timm_name, pretrained=pretrained, num_classes=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        features = self.net.forward_features(images)
        pooled = self.net.forward_head(features, pre_logits=True)
        if pooled.ndim > 2:
            pooled = pooled.flatten(start_dim=1)
        return pooled


def build_image_loader(frame: pd.DataFrame, root: Path, cfg: dict, train: bool, seed: int):
    dataset = ImageAdmissionDataset(
        frame,
        root,
        image_transform(int(cfg["image"]["size"]), train=train),
    )
    return make_loader(
        dataset,
        batch_size=int(cfg["image"]["batch_size"]),
        shuffle=train,
        seed=seed,
        num_workers=int(cfg["image"]["num_workers"]),
    )


def train_image_model(frame_train, frame_val, root, cfg, device, timm_name: str, checkpoint_path: Path) -> float:
    model = ImageClassifier(timm_name, pretrained=bool(cfg["image"]["pretrained"])).to(device)
    seed = int(cfg["seed"])
    train_loader = build_image_loader(frame_train, root, cfg, train=True, seed=seed)
    val_loader = build_image_loader(frame_val, root, cfg, train=False, seed=seed)
    score = fit_model(
        model,
        train_loader,
        val_loader,
        device,
        frame_train["hospital_expire_flag"].to_numpy(),
        epochs=int(cfg["image"]["epochs"]),
        learning_rate=float(cfg["image"]["learning_rate"]),
        weight_decay=float(cfg["image"]["weight_decay"]),
        modality="image",
        checkpoint_path=checkpoint_path,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score


def load_image_model(timm_name: str, checkpoint_path: Path, device) -> ImageClassifier:
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing {checkpoint_path}. Run scripts/03_train_images.py first.")
    model = ImageClassifier(timm_name, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def image_probabilities(model, frame, root, cfg, device) -> tuple[np.ndarray, np.ndarray]:
    loader = build_image_loader(frame, root, cfg, train=False, seed=int(cfg["seed"]))
    return collect_probabilities(model, loader, device, "image")


def image_embeddings(model, frame, root, cfg, device) -> tuple[np.ndarray, np.ndarray]:
    loader = build_image_loader(frame, root, cfg, train=False, seed=int(cfg["seed"]))
    return collect_embeddings(model, loader, device, "image")
