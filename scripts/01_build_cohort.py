"""Stage 1. Inspect column names, then build the cohort."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cohort import build_cohort
from src.utils import prepare


def main() -> None:
    root, cfg = prepare()
    build_cohort(root, cfg)


if __name__ == "__main__":
    main()
