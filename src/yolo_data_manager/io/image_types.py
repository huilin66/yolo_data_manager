from __future__ import annotations

from pathlib import Path

from PIL import Image


def image_mode_details(mode: str) -> tuple[str, int | None]:
    """Return the NumPy-style dtype and bit depth implied by a Pillow mode."""

    if mode.startswith("I;16"):
        return "uint16", 16
    if mode == "I":
        return "int32", 32
    if mode == "F":
        return "float32", 32
    if mode == "1":
        return "bool", 1
    return "uint8", 8


def read_image_type(path: str | Path) -> dict[str, object]:
    """Read image header information without decoding the complete pixel array."""

    with Image.open(path) as image:
        dtype, bit_depth = image_mode_details(image.mode)
        width, height = image.size
        return {
            "format": image.format or "unknown",
            "mode": image.mode,
            "dtype": dtype,
            "bit_depth": bit_depth,
            "channels": len(image.getbands()),
            "width": width,
            "height": height,
        }
