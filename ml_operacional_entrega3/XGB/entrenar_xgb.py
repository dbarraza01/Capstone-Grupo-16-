"""Entrenamiento final XGB operacional en dos etapas."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_operacional_entrega3.utils.model_workflows import train_two_stage_model


def main() -> None:
    train_two_stage_model("xgb")


if __name__ == "__main__":
    main()
