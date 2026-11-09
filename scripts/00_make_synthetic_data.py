"""Stage 0. Write a fake MIMIC-shaped dataset under data/synthetic.

This does not read or overwrite the credentialed MIMIC folders.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic import generate_synthetic_dataset
from src.utils import prepare


def main() -> None:
    root, cfg = prepare()
    generate_synthetic_dataset(root, cfg)


if __name__ == "__main__":
    main()
