"""Compatibility wrapper for the archived editable task script.

For new work, prefer ``python -m example.datasets.run_dataset --data-dir ...``.
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

from example.archive.scripts.run_ydm import main  # noqa: E402
from example.datasets.run_dataset import main as run_dataset_main  # noqa: E402


if __name__ == "__main__":
    # Keep the old no-argument editable behavior, while allowing the new
    # runtime-path runner to be used from this historical location as well.
    if len(sys.argv) > 1:
        raise SystemExit(run_dataset_main())
    raise SystemExit(main())
