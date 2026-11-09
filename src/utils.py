"""Shared helpers: config, paths, and the random seed."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml


def project_root() -> Path:
    """Directory that contains config.yaml. This file lives in src/."""
    return Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path} did not contain a YAML mapping.")
    return cfg


def apply_smoke(cfg: dict) -> dict:
    """Small settings so the whole pipeline can be checked on a laptop."""
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("runtime", {})
    cfg["runtime"]["smoke"] = True
    cfg["cohort"]["max_admissions"] = min(int(cfg["cohort"]["max_admissions"]), 64)
    cfg["synthetic"]["n_subjects"] = min(int(cfg["synthetic"]["n_subjects"]), 80)
    cfg["image"]["epochs"] = 1
    cfg["image"]["batch_size"] = min(int(cfg["image"]["batch_size"]), 8)
    cfg["text"]["epochs"] = 1
    cfg["text"]["batch_size"] = min(int(cfg["text"]["batch_size"]), 2)
    cfg["tabular"]["n_estimators"] = 20
    cfg["tabular"]["early_stopping_rounds"] = 5
    cfg["fusion"]["n_estimators"] = 20
    cfg["fusion"]["early_stopping_rounds"] = 5
    cfg["explain"]["shap_samples"] = 16
    cfg["explain"]["gradcam_images"] = 4
    cfg["explain"]["n_patients"] = 3
    return cfg


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # Data scripts can run before PyTorch is installed.
        pass


def prepare(argv: list[str] | None = None) -> tuple[Path, dict]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny cohort and one epoch so the pipeline can be checked quickly.",
    )
    args = parser.parse_args(argv)

    root = project_root()
    config_path = Path(args.config) if args.config else root / "config.yaml"
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_config(config_path)

    smoke = bool(args.smoke or cfg.get("runtime", {}).get("smoke", False))
    if smoke:
        cfg = apply_smoke(cfg)
        print("SMOKE MODE: small cohort, one epoch. This checks the code, it does not train a final model.")

    if sys.version_info < (3, 11):
        print(f"Warning: this project targets Python 3.11 or newer. You are on {sys.version.split()[0]}.")

    set_seed(int(cfg["seed"]))
    return root, cfg


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the earlier scripts first.")
    return json.loads(path.read_text(encoding="utf-8"))


def update_val_scores(path: Path, key: str, payload: dict) -> None:
    """Store validation scores used to pick checkpoints. These are not test results."""
    data: dict = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    data["note"] = (
        "Validation scores for checkpoint selection. Not test results. "
        "Test numbers are written only by scripts/07_evaluate.py."
    )
    data[key] = _json_ready(payload)
    image_keys = [name for name in ("densenet121", "vit") if name in data]
    if image_keys and key in {"densenet121", "vit"}:
        data["best_image_model"] = max(image_keys, key=lambda name: _score_or_low(data[name]))
    write_json(path, data)


def _score_or_low(payload: dict) -> float:
    value = payload.get("val_auroc")
    if value is None:
        return -1.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -1.0
    if math.isnan(number):
        return -1.0
    return number


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def get_device():
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {name}")
        return torch.device("cuda")
    print("Using CPU. A smoke test can run here. The full cohort needs a GPU.")
    return torch.device("cpu")
