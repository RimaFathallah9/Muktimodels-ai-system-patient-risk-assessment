"""Run the stages in order. Stops at the first script that fails."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, project_root


STAGES = [
    "00_make_synthetic_data.py",
    "01_build_cohort.py",
    "02_train_tabular.py",
    "03_train_images.py",
    "04_train_text.py",
    "05_cache_embeddings.py",
    "06_train_fusion.py",
    "07_evaluate.py",
    "08_explain.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full patient-risk pipeline.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    root = project_root()
    config_path = Path(args.config) if args.config else root / "config.yaml"
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_config(config_path)

    extra: list[str] = []
    if args.config:
        extra.extend(["--config", args.config])
    if args.smoke or bool(cfg.get("runtime", {}).get("smoke", False)):
        extra.append("--smoke")

    stages = list(STAGES)
    if cfg["data"]["source"] != "synthetic":
        stages = [name for name in stages if not name.startswith("00_")]
        print("data.source is mimic, so the synthetic generator will not run.")

    for name in stages:
        script = root / "scripts" / name
        print(f"\n===== {name} =====")
        subprocess.run([sys.executable, str(script), *extra], check=True)

    print("\nPipeline finished.")
    print("Test numbers, if training ran, are in outputs/metrics/results.md")
    if cfg["data"]["source"] == "synthetic":
        print("The dataset was synthetic. Do not report those numbers as results.")


if __name__ == "__main__":
    main()
