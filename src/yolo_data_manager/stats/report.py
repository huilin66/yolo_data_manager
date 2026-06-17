from __future__ import annotations

import json
import csv
from pathlib import Path


def write_json_report(data: dict[str, object], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_class_counts_csv(data: dict[str, object], path: str | Path) -> None:
    counts = data.get("class_counts", {})
    if not isinstance(counts, dict):
        counts = {}
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["class_name", "count"])
        writer.writeheader()
        for class_name, count in counts.items():
            writer.writerow({"class_name": class_name, "count": count})
