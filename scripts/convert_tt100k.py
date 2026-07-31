r"""Command-line wrapper for the TT100K-to-YOLO converter.

Example (PowerShell)::

    python scripts/convert_tt100k.py `
      --src "\\158.132.186.40\isds\huilin\traffic_sign\seg\tt100k_2021" `
      --out "E:\datasets\tt100k_yolo"
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from yolo_data_manager.converters.tt100k import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
