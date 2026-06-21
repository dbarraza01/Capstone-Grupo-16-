"""Genera la comparacion final XGB vs RF vs LR."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_operacional.utils.model_workflows import build_model_comparison


def main() -> None:
    comparison = build_model_comparison()
    print("Comparacion final guardada en ml_operacional/reports/")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
