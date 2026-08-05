"""Interactively draw one temporary YOLO box without changing a label file.

Example::

    python example/manual_draw_box.py /data/images/0001.jpg --class-id 5

The matching label is inferred from ``images/...`` -> ``labels/...``. The
coordinates are printed as JSON after pressing Enter in the drawing window.
Use ``--output`` to save that JSON to a separate file; the label is never
modified by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from yolo_data_manager.vis.manual_box import draw_manual_box
from yolo_data_manager.io.layout import infer_label_path_from_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="image filename")
    parser.add_argument("--label", default=None, help="optional matching YOLO txt path")
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--max-width", type=int, default=1400)
    parser.add_argument("--max-height", type=int, default=900)
    parser.add_argument("--min-pixels", type=int, default=2)
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--output", default=None, help="optional JSON output path")
    args = parser.parse_args()

    if args.output is not None:
        output_path = Path(args.output).resolve()
        protected_paths = {Path(args.image).resolve()}
        protected_paths.add(
            (Path(args.label) if args.label is not None else infer_label_path_from_image(Path(args.image))).resolve()
        )
        if output_path in protected_paths:
            parser.error("--output must be a separate JSON path, not the source image or label")

    try:
        result = draw_manual_box(
            args.image,
            label_path=args.label,
            class_id=args.class_id,
            max_width=args.max_width,
            max_height=args.max_height,
            min_pixels=args.min_pixels,
            precision=args.precision,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"manual_draw_box failed: {exc}", file=sys.stderr)
        return 2
    payload = {"cancelled": True, "image": str(Path(args.image))} if result is None else result.to_dict()
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
