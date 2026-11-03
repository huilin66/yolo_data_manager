"""Split ``vis_crop`` image files into train/val/test lists.

The crop filename written by YDM contains the source image stem and the
1-based annotation index, for example ``frame_001_3.jpg``.  This tool can
therefore either reuse the source dataset split, randomly split source stems,
or randomly split individual crop files.

Examples
--------
Reuse an existing dataset split::

    python tools/split_vis_crops.py \
        --crops-dir dataset/ydm_vis/crop \
        --split-dir dataset \
        --out dataset/ydm_vis/crop_split \
        --mode source_split

Keep every crop from one source image in one random split::

    python tools/split_vis_crops.py \
        --crops-dir dataset/ydm_vis/crop \
        --out dataset/ydm_vis/crop_split \
        --mode stem_random --train 0.8 --val 0.2 --test 0 --seed 233

Randomly assign every crop file independently::

    python tools/split_vis_crops.py \
        --crops-dir dataset/ydm_vis/crop \
        --out dataset/ydm_vis/crop_split \
        --mode crop_random --train 0.8 --val 0.2 --test 0 --seed 233
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable


SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)
CROP_NAME_RE = re.compile(r"^(?P<source>.+)_(?P<annotation_index>[0-9]+)$")
SPLIT_PRIORITY = {"train": 2, "test": 1, "val": 0}


class CropRecord:
    """One crop file and the source information encoded in its filename."""

    __slots__ = ("path", "relative_path", "source_stem", "source_key", "annotation_index")

    def __init__(
        self,
        path: Path,
        relative_path: str,
        source_stem: str | None,
        annotation_index: int | None,
    ) -> None:
        self.path = path
        self.relative_path = relative_path
        self.source_stem = source_stem
        self.source_key = _normalise_key(source_stem) if source_stem else None
        self.annotation_index = annotation_index


def _normalise_key(value: object) -> str:
    return str(value).replace("\\", "/").strip().casefold()


def _stem_from_split_value(value: str) -> str:
    """Extract a source image stem from a relative or absolute split entry."""

    text = str(value).strip().replace("\\", "/")
    if not text:
        return ""
    return Path(PurePosixPath(text).name).stem


def _parse_crop_name(path: Path) -> tuple[str | None, int | None]:
    match = CROP_NAME_RE.match(path.stem)
    if match is None:
        return None, None
    return match.group("source"), int(match.group("annotation_index"))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _iter_crop_files(crops_dir: Path, *, excluded_dir: Path | None = None) -> list[CropRecord]:
    records: list[CropRecord] = []
    for path in sorted(crops_dir.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        resolved = path.resolve()
        if excluded_dir is not None and _is_relative_to(resolved, excluded_dir):
            continue
        source_stem, annotation_index = _parse_crop_name(path)
        records.append(
            CropRecord(
                path=path,
                relative_path=path.relative_to(crops_dir).as_posix(),
                source_stem=source_stem,
                annotation_index=annotation_index,
            )
        )
    return records


def _read_source_split(split_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Read source image stem -> split mappings from train/val/test txt files."""

    mapping: dict[str, str] = {}
    files: dict[str, str] = {}
    found = 0
    for split in SPLITS:
        split_file = split_dir / f"{split}.txt"
        if not split_file.is_file():
            continue
        found += 1
        files[split] = str(split_file)
        for line_number, raw_line in enumerate(
            split_file.read_text(encoding="utf-8-sig").splitlines(),
            start=1,
        ):
            value = raw_line.strip()
            if not value or value.startswith("#"):
                continue
            stem = _stem_from_split_value(value)
            key = _normalise_key(stem)
            previous = mapping.get(key)
            if previous is not None and previous != split:
                raise ValueError(
                    f"source stem {stem!r} appears in both {previous}.txt and "
                    f"{split}.txt ({split_file}:{line_number})"
                )
            mapping[key] = split
    if found == 0:
        raise FileNotFoundError(
            f"no train.txt, val.txt, or test.txt found in split directory: {split_dir}"
        )
    return mapping, files


def _validate_ratios(train: float, val: float, test: float) -> dict[str, float]:
    ratios = {"train": float(train), "val": float(val), "test": float(test)}
    if not all(math.isfinite(value) for value in ratios.values()):
        raise ValueError("train, val, and test ratios must be finite")
    if min(ratios.values()) < 0:
        raise ValueError("train, val, and test ratios must be non-negative")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("train + val + test must equal 1.0")
    return ratios


def _allocate_units(
    units: Iterable[object],
    ratios: dict[str, float],
    *,
    seed: int,
) -> dict[str, set[object]]:
    """Randomly allocate units while keeping zero-ratio splits empty."""

    shuffled = list(units)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    if ratios["test"] == 0.0:
        sizes = {
            "train": int(count * ratios["train"]),
            "val": count - int(count * ratios["train"]),
            "test": 0,
        }
    else:
        raw_sizes = {split: count * ratios[split] for split in SPLITS}
        sizes = {split: int(raw_sizes[split]) for split in SPLITS}
        remainder = count - sum(sizes.values())
        fractional = sorted(
            (split for split in SPLITS if ratios[split] > 0),
            key=lambda split: (
                -(raw_sizes[split] - sizes[split]),
                -SPLIT_PRIORITY[split],
            ),
        )
        for split in fractional[:remainder]:
            if ratios[split] > 0:
                sizes[split] += 1

    result: dict[str, set[object]] = {split: set() for split in SPLITS}
    offset = 0
    for split in SPLITS:
        result[split].update(shuffled[offset : offset + sizes[split]])
        offset += sizes[split]
    return result


def _relative_or_absolute(path: Path, *, absolute_paths: bool) -> str:
    return str(path.resolve()) if absolute_paths else path.as_posix()


def split_vis_crops(
    crops_dir: str | Path,
    out: str | Path | None = None,
    *,
    mode: str = "source_split",
    split_dir: str | Path | None = None,
    train: float = 0.8,
    val: float = 0.2,
    test: float = 0.0,
    seed: int = 233,
    absolute_paths: bool = False,
    copy_files: bool = False,
    unmatched: str = "skip",
) -> dict[str, object]:
    """Create split txt files for a YDM ``vis_crop`` output directory.

    ``source_split`` uses ``split_dir/train.txt`` etc.  ``stem_random``
    assigns source image stems randomly, keeping all their crops together.
    ``crop_random`` assigns every crop file independently.
    """

    mode = str(mode).strip().lower().replace("-", "_")
    mode_aliases = {"source": "source_split", "random_stem": "stem_random", "random_crop": "crop_random"}
    mode = mode_aliases.get(mode, mode)
    if mode not in {"source_split", "stem_random", "crop_random"}:
        raise ValueError("mode must be source_split, stem_random, or crop_random")
    if unmatched not in {"skip", "error"}:
        raise ValueError("unmatched must be 'skip' or 'error'")

    ratios = _validate_ratios(train, val, test)
    crop_root = Path(crops_dir).expanduser().resolve()
    if not crop_root.is_dir():
        raise FileNotFoundError(f"crop directory does not exist: {crop_root}")
    out_root = (
        Path(out).expanduser().resolve()
        if out is not None
        else crop_root.parent / f"{crop_root.name}_split"
    )
    if out_root == crop_root:
        raise ValueError("out must be different from crops_dir")
    out_root.mkdir(parents=True, exist_ok=True)

    records = _iter_crop_files(crop_root, excluded_dir=out_root)
    if not records:
        raise ValueError(f"no crop image files found in {crop_root}")

    source_mapping: dict[str, str] = {}
    source_split_files: dict[str, str] = {}
    if mode == "source_split":
        if split_dir is None:
            raise ValueError("split_dir is required when mode='source_split'")
        source_mapping, source_split_files = _read_source_split(
            Path(split_dir).expanduser().resolve()
        )

    selected: dict[str, list[CropRecord]] = {split: [] for split in SPLITS}
    unmatched_records: list[CropRecord] = []
    invalid_records: list[CropRecord] = []

    if mode == "source_split":
        for record in records:
            split = source_mapping.get(record.source_key or "")
            if split is None:
                unmatched_records.append(record)
            else:
                selected[split].append(record)
    elif mode == "stem_random":
        groups: dict[str, list[CropRecord]] = defaultdict(list)
        for record in records:
            if record.source_key is None:
                invalid_records.append(record)
            else:
                groups[record.source_key].append(record)
        group_splits = _allocate_units(groups.keys(), ratios, seed=seed)
        for split, keys in group_splits.items():
            for key in keys:
                selected[split].extend(groups[key])
    else:
        record_splits = _allocate_units(range(len(records)), ratios, seed=seed)
        for split, indices in record_splits.items():
            selected[split].extend(records[index] for index in indices)

    if mode == "stem_random" and invalid_records:
        unmatched_records.extend(invalid_records)
    if unmatched == "error" and unmatched_records:
        examples = ", ".join(record.relative_path for record in unmatched_records[:5])
        raise ValueError(
            f"{len(unmatched_records)} crop files could not be assigned; examples: {examples}"
        )

    for split in SPLITS:
        selected[split].sort(key=lambda record: record.relative_path.casefold())

    output_entries: dict[str, list[str]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        for record in selected[split]:
            if copy_files:
                target = out_root / split / Path(record.relative_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record.path, target)
                path_value = _relative_or_absolute(
                    target,
                    absolute_paths=absolute_paths,
                )
                if not absolute_paths:
                    path_value = target.relative_to(out_root).as_posix()
            else:
                path_value = _relative_or_absolute(
                    Path(record.relative_path),
                    absolute_paths=False,
                )
                if absolute_paths:
                    path_value = str(record.path.resolve())
            output_entries[split].append(path_value)

        (out_root / f"{split}.txt").write_text(
            "".join(f"{value}\n" for value in output_entries[split]),
            encoding="utf-8",
        )

    split_summary = {
        split: {
            "crops": len(selected[split]),
            "source_stems": len(
                {
                    record.source_key
                    for record in selected[split]
                    if record.source_key is not None
                }
            ),
            "txt": str(out_root / f"{split}.txt"),
        }
        for split in SPLITS
    }
    report: dict[str, object] = {
        "crops_dir": str(crop_root),
        "out": str(out_root),
        "mode": mode,
        "seed": seed if mode != "source_split" else None,
        "ratios": ratios,
        "absolute_paths": absolute_paths,
        "copy_files": copy_files,
        "path_root": "output" if copy_files else "crops_dir",
        "source_split_files": source_split_files,
        "splits": split_summary,
        "total_crops": len(records),
        "assigned_crops": sum(len(values) for values in selected.values()),
        "unmatched_crops": len(unmatched_records),
        "unmatched_examples": [
            record.relative_path for record in unmatched_records[:20]
        ],
    }
    report_path = out_root / "split_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report["report"] = str(report_path)

    print(f"Crop split mode: {mode}")
    print("split | crops | source_stems")
    print("------+-------+-------------")
    for split in SPLITS:
        print(
            f"{split:<5} | {split_summary[split]['crops']:<5} | "
            f"{split_summary[split]['source_stems']}"
        )
    if unmatched_records:
        print(f"Unassigned crops: {len(unmatched_records)}")
    print(f"Report: {report_path}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create train/val/test txt files from YDM vis_crop results."
    )
    parser.add_argument("--crops-dir", required=True, help="vis_crop output directory")
    parser.add_argument(
        "--out",
        default=None,
        help="output directory; defaults to <crops-dir>_split beside the crop directory",
    )
    parser.add_argument(
        "--mode",
        choices=["source_split", "stem_random", "crop_random"],
        default="source_split",
        help="reuse source txt, randomize source stems, or randomize individual crop files",
    )
    parser.add_argument(
        "--split-dir",
        default=None,
        help="dataset directory containing train.txt, val.txt, and test.txt; required for source_split",
    )
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="write absolute crop paths in the generated txt files",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="copy crops into <out>/train, <out>/val, and <out>/test",
    )
    parser.add_argument(
        "--unmatched",
        choices=["skip", "error"],
        default="skip",
        help="how source_split/stem_random handle filenames without a source stem",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        split_vis_crops(
            args.crops_dir,
            args.out,
            mode=args.mode,
            split_dir=args.split_dir,
            train=args.train,
            val=args.val,
            test=args.test,
            seed=args.seed,
            absolute_paths=args.absolute_paths,
            copy_files=args.copy_files,
            unmatched=args.unmatched,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
