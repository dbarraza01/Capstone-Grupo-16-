"""Paso 0: segmentacion y exportacion de splits para auditoria."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_operacional.utils.pipeline_operacional import split_and_export


def main() -> None:
    outputs = split_and_export(force=True)
    print("Splits exportados en ml_operacional/data_splits/")
    for name, df in outputs.items():
        print(f"  {name}: {df.shape}")


if __name__ == "__main__":
    main()
