"""Bio_ClinicalBERT fine-tuned for one binary label.

The embedding is the [CLS] vector from the last hidden layer, before the linear head.
Notes are truncated to 512 tokens.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from src.data.datasets import NoteAdmissionDataset, make_loader
from src.models.loop import collect_embeddings, collect_probabilities, fit_model


class ClinicalNoteClassifier(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids, attention_mask))


def _tokenizer(cfg: dict):
    return AutoTokenizer.from_pretrained(cfg["text"]["model_name"])


def build_text_loader(frame: pd.DataFrame, cfg: dict, train: bool, seed: int, tokenizer=None):
    if tokenizer is None:
        tokenizer = _tokenizer(cfg)
    dataset = NoteAdmissionDataset(frame, tokenizer, max_length=int(cfg["text"]["max_length"]))
    return make_loader(
        dataset,
        batch_size=int(cfg["text"]["batch_size"]),
        shuffle=train,
        seed=seed,
        num_workers=0,
    )


def train_text_model(frame_train, frame_val, cfg, device, checkpoint_path: Path) -> float:
    model = ClinicalNoteClassifier(cfg["text"]["model_name"]).to(device)
    seed = int(cfg["seed"])
    tokenizer = _tokenizer(cfg)
    train_loader = build_text_loader(frame_train, cfg, train=True, seed=seed, tokenizer=tokenizer)
    val_loader = build_text_loader(frame_val, cfg, train=False, seed=seed, tokenizer=tokenizer)
    return fit_model(
        model,
        train_loader,
        val_loader,
        device,
        frame_train["hospital_expire_flag"].to_numpy(),
        epochs=int(cfg["text"]["epochs"]),
        learning_rate=float(cfg["text"]["learning_rate"]),
        weight_decay=float(cfg["text"]["weight_decay"]),
        modality="text",
        checkpoint_path=checkpoint_path,
    )


def load_text_model(cfg: dict, checkpoint_path: Path, device) -> ClinicalNoteClassifier:
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing {checkpoint_path}. Run scripts/04_train_text.py first.")
    model = ClinicalNoteClassifier(cfg["text"]["model_name"])
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def text_probabilities(model, frame, cfg, device) -> tuple[np.ndarray, np.ndarray]:
    loader = build_text_loader(frame, cfg, train=False, seed=int(cfg["seed"]))
    return collect_probabilities(model, loader, device, "text")


def text_embeddings(model, frame, cfg, device) -> tuple[np.ndarray, np.ndarray]:
    loader = build_text_loader(frame, cfg, train=False, seed=int(cfg["seed"]))
    return collect_embeddings(model, loader, device, "text")
