"""Compatibility wrapper for :mod:`example.functions.mdet_tools`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .functions.mdet_tools import *  # noqa: F401,F403
except ImportError:  # supports ``python example/mdet_tools.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from example.functions.mdet_tools import *  # noqa: F401,F403
