"""Command-line entry point for the standalone TT100K converter.

The conversion implementation is the reusable
``yolo_data_manager.converters.tt100k.convert_tt100k`` function.  This file is
only a convenient repository tool and is unrelated to dataset examples.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from yolo_data_manager.converters.tt100k import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

