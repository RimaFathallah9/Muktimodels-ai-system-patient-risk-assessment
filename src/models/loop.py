"""The training loop shared by the image models and the text model.

Logits are shaped (batch, 1). That makes pos_weight apply to the one output,
rather than to the batch dimension.
"""

from __future__ import annotations

from pathlib import Path

from contextlib import nullcontext

import numpy as np
import torch
from torch import nn

from src.eval.metrics import safe_auroc


def _autocast(device):
    # Mixed precision is for the GPU. On CPU the normal float32 path is used.
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def batch_logits(model, batch, device, modality: str) -> torch.Tensor:
    if modality == "image":
        return model(batch["image"].to(device))
    if modality == "text":
        return model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
    raise ValueError(f"Unknown modality: {modality}")


def batch_embedding(model, batch, device, modality: str) -> torch.Tensor:
    if modality == "image":
        return model.embed(batch["image"].to(device))
    if modality == "text":
        return model.embed(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
    raise ValueError(f"Unknown modality: {modality}")


def run_epoch(model, loader, device, loss_fn, optimizer, scaler, train: bool, modality: str):
    model.train(train)
    total_loss = 0.0
    seen = 0
    labels_all = []
    probs_all = []

    for batch in loader:
        labels = batch["label"].to(device).view(-1, 1)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            logits = batch_logits(model, batch, device, modality).view(-1, 1)
            loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss is not finite. Lower the learning rate in config.yaml.")
        if train:
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        seen += batch_size
        probs_all.append(torch.sigmoid(logits.detach()).view(-1).cpu().numpy())
        labels_all.append(labels.detach().view(-1).cpu().numpy())

    y_true = np.concatenate(labels_all)
    y_prob = np.concatenate(probs_all)
    return total_loss / max(seen, 1), safe_auroc(y_true, y_prob)


def fit_model(model, train_loader, val_loader, device, y_train, epochs, learning_rate, weight_decay, modality, checkpoint_path: Path) -> float:
    y_train = np.asarray(y_train).astype(int)
    n_positive = float(y_train.sum())
    n_negative = float(len(y_train) - n_positive)
    if n_positive <= 0 or n_negative <= 0:
        raise RuntimeError("The training split needs both classes.")
    pos_weight = torch.tensor([n_negative / n_positive], dtype=torch.float32, device=device)
    print(f"BCE pos_weight = {n_negative / n_positive:.3f} (train negatives / train positives)")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best = -1.0
    best_is_defined = False
    for epoch in range(1, epochs + 1):
        train_loss, train_auroc = run_epoch(
            model, train_loader, device, loss_fn, optimizer, scaler, True, modality
        )
        val_loss, val_auroc = run_epoch(
            model, val_loader, device, loss_fn, optimizer, scaler, False, modality
        )
        print(
            f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  train_auroc={train_auroc:.4f}  "
            f"val_loss={val_loss:.4f}  val_auroc={val_auroc:.4f}"
        )
        if not np.isnan(val_auroc) and (not best_is_defined or val_auroc > best):
            best = float(val_auroc)
            best_is_defined = True
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  saved {checkpoint_path.name}")

    if not checkpoint_path.exists():
        torch.save(model.state_dict(), checkpoint_path)
        print("Validation AUROC was undefined for every epoch. Saved the last epoch.")

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return best if best_is_defined else float("nan")


@torch.inference_mode()
def collect_probabilities(model, loader, device, modality: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ids = []
    probs = []
    for batch in loader:
        with _autocast(device):
            logits = batch_logits(model, batch, device, modality)
        probs.append(torch.sigmoid(logits.float()).view(-1).cpu().numpy())
        ids.append(batch["hadm_id"].numpy())
    return np.concatenate(ids), np.concatenate(probs)


@torch.inference_mode()
def collect_embeddings(model, loader, device, modality: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ids = []
    vectors = []
    for batch in loader:
        with _autocast(device):
            embedding = batch_embedding(model, batch, device, modality)
        vectors.append(embedding.float().cpu().numpy())
        ids.append(batch["hadm_id"].numpy())
    return np.concatenate(ids).astype(np.int64), np.concatenate(vectors).astype(np.float32)
