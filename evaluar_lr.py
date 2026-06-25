"""Evaluacion holdout baseline lineal regularizado."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_operacional.utils.model_workflows import evaluate_lr_model


def main() -> None:
    evaluate_lr_model()


if __name__ == "__main__":
    main()
