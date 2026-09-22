#!/usr/bin/env python3
"""Merge manually reviewed groups and split a YOLO mdet dataset safely.

This script is intentionally independent of ReID inference.  It treats every
folder in ``group_src`` and ``group`` as a hard same-set constraint, maps crop
names such as ``FLIR0602_wall_signboard_det0004.png`` back to the original
frame ``FLIR0602``, and computes connected components across both sources.
Consequently, if one frame occurs in two manually reviewed folders, all images
in both folders are assigned to the same final group.

The split stage assigns whole groups (not individual images) to train/val/test.
It uses a deterministic greedy initialization plus local improvement.  When
manual groups are fixed in train, the objective prioritizes attribute coverage
as train > test > val, then minimizes image-count deviation from the requested
ratios.  Attribute coverage is measured at image level: an image is positive
for an attribute if at least one of its label rows has that attribute equal to 1.

Typical Windows usage::

    python split_by_manual_group.py merge-groups \
        --out-dir "\\\\158.132.186.40\\isds\\huilin\\mayolo\\yolo_rgb_detection5_10_c\\group_merged"

    python split_by_manual_group.py split \
        --groups-dir "\\\\158.132.186.40\\isds\\huilin\\mayolo\\yolo_rgb_detection5_10_c\\group_merged" \
        --manual-groups-split train \
        --make-yolo \
        --out-dir "\\\\158.132.186.40\\isds\\huilin\\mayolo\\yolo_rgb_detection5_10_c\\manual_group_split"

Linux usage is identical after replacing the paths with mounted paths and
using ``python3``.  The original input folders are never modified; an existing
output requires explicit ``--overwrite``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import re
import shutil
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

DEFAULT_BASE = r"\\158.132.186.40\isds\huilin\mayolo\mayolo_v1"
DEFAULT_GROUP_SRC = str(Path(DEFAULT_BASE) / "group_src")
DEFAULT_GROUP = str(Path(DEFAULT_BASE) / "group")
DEFAULT_IMAGES = str(Path(DEFAULT_BASE) / "images")
DEFAULT_LABELS = str(Path(DEFAULT_BASE) / "labels")
DEFAULT_MERGED_GROUP = str(Path(DEFAULT_BASE) / "group_merged")
DEFAULT_SPLIT_OUT = str(Path(DEFAULT_BASE) / "manual_group_split")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FRAME_PATTERN = re.compile(r"^(FLIR\d+)", re.IGNORECASE)
DEFAULT_ATTRIBUTE_NAMES = [
    "surface_missing",
    "surface_incomplete",
    "surface_corroded",
    "frame_corroded",
    "surface_peeling",
    "surface_fade",
    "surface_deformed",
    "frame_deformed",
    "disconnected",
    "added_billboard",
]


def is_image(path: Path) -> bool:
    """Return whether a path has a supported image extension."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def canonical_key(name: str) -> str:
    """Map a full-frame or crop filename to a stable image key."""
    stem = Path(name).stem
    match = FRAME_PATTERN.match(stem)
    return match.group(1).upper() if match else stem.upper()


def frame_sort_key(key: str) -> tuple[int, str]:
    """Sort FLIR keys numerically and fall back to lexical order."""
    match = re.search(r"(\d+)$", key)
    return (int(match.group(1)), key) if match else (10**18, key)


class DisjointSet:
    """Small deterministic union-find implementation."""

    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def discover_group_records(group_dir: str, source_name: str) -> list[dict[str, object]]:
    """Read one manual group root into records of canonical image keys."""
    root = Path(group_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Group directory not found: {root}")

    records: list[dict[str, object]] = []
    for folder in sorted(
        (p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name
    ):
        files = [p for p in folder.rglob("*") if is_image(p)]
        keys = sorted({canonical_key(p.name) for p in files}, key=frame_sort_key)
        records.append(
            {
                "source": source_name,
                "folder": folder.name,
                "path": str(folder),
                "files": len(files),
                "members": keys,
            }
        )
    return records


def merge_records(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Merge records connected by a shared canonical image key."""
    dsu = DisjointSet(len(records))
    key_to_records: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        for key in record["members"]:  # type: ignore[union-attr]
            key_to_records[str(key)].append(index)
    for indices in key_to_records.values():
        for index in indices[1:]:
            dsu.union(indices[0], index)

    component_indices: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        component_indices[dsu.find(index)].append(index)

    components: list[dict[str, object]] = []
    for indices in component_indices.values():
        member_keys = sorted(
            {str(key) for index in indices for key in records[index]["members"]},  # type: ignore[union-attr]
            key=frame_sort_key,
        )
        source_groups = [
            {
                "source": records[index]["source"],
                "folder": records[index]["folder"],
                "files": records[index]["files"],
            }
            for index in sorted(indices)
        ]
        components.append({"members": member_keys, "source_groups": source_groups})

    components.sort(
        key=lambda item: (
            frame_sort_key(item["members"][0]) if item["members"] else (10**18, "")
        )
    )
    for index, component in enumerate(components):
        component["group_id"] = f"group_{index:03d}_size{len(component['members'])}"
    return components


def index_images(images_dir: str) -> tuple[dict[str, Path], list[str]]:
    """Index source images by canonical key and report duplicate keys."""
    root = Path(images_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Images directory not found: {root}")
    index: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(
        (p for p in root.rglob("*") if is_image(p)), key=lambda p: p.name
    ):
        key = canonical_key(path.name)
        if key in index:
            duplicates.append(key)
            # Prefer the PNG source, then the lexicographically earlier path.
            old = index[key]
            if path.suffix.lower() == ".png" and old.suffix.lower() != ".png":
                index[key] = path
        else:
            index[key] = path
    return index, sorted(set(duplicates), key=frame_sort_key)


def prepare_output(path: str, overwrite: bool) -> Path:
    """Create a new output directory, requiring explicit overwrite."""
    output = Path(path).expanduser()
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists; pass --overwrite explicitly: {output}"
            )
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.mkdir(parents=True, exist_ok=False)
    return output.resolve()


def copy_or_link(source: Path, target: Path, link: bool) -> None:
    """Copy an image or create a symbolic link without overwriting."""
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output file: {target}")
    if link:
        os.symlink(str(source.resolve()), str(target))
    else:
        shutil.copy2(str(source), str(target))


def write_json(path: Path, value: object) -> None:
    """Write UTF-8 JSON with stable readable formatting."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def write_group_csv(path: Path, components: Sequence[dict[str, object]]) -> None:
    """Write one row per merged group."""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group_id", "size", "members", "source_groups"])
        for component in components:
            sources = ";".join(
                f"{item['source']}:{item['folder']}"
                for item in component["source_groups"]
            )
            writer.writerow(
                [
                    component["group_id"],
                    len(component["members"]),
                    ";".join(component["members"]),
                    sources,
                ]
            )


def merge_groups(args: argparse.Namespace) -> Path:
    """Merge the two manual group roots into a new canonical group folder."""
    records = discover_group_records(args.group_src, "group_src")
    records.extend(discover_group_records(args.group_dir, "group"))
    components = merge_records(records)
    image_index, duplicate_images = index_images(args.images_dir)

    missing = sorted(
        {
            key
            for component in components
            for key in component["members"]
            if key not in image_index
        },
        key=frame_sort_key,
    )
    if missing:
        raise FileNotFoundError(
            "Manual groups reference images that are absent from --images-dir: "
            + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )

    summary = {
        "group_src": args.group_src,
        "group": args.group_dir,
        "images_dir": args.images_dir,
        "input_group_records": len(records),
        "merged_groups": len(components),
        "merged_images": len(
            {key for component in components for key in component["members"]}
        ),
        "duplicate_source_image_keys": duplicate_images,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return Path(args.out_dir).expanduser()

    output = prepare_output(args.out_dir, args.overwrite)
    for component in components:
        target_dir = output / str(component["group_id"])
        target_dir.mkdir()
        for key in component["members"]:
            source = image_index[key]
            copy_or_link(source, target_dir / source.name, args.link)

    manifest = {
        "schema_version": 1,
        "kind": "manual_group_merge",
        **summary,
        "out_dir": str(output),
        "groups": components,
    }
    write_json(output / "group_manifest.json", manifest)
    write_group_csv(output / "group_manifest.csv", components)
    print(f"Merged group folder created: {output}")
    return output


def load_class_names(class_file: str | None, images_dir: str) -> list[str]:
    """Load class names from class.txt, preserving class IDs by line order."""
    path = (
        Path(class_file).expanduser()
        if class_file
        else Path(images_dir).parent / "class.txt"
    )
    if path.is_file():
        names = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if names:
            return names
    return ["background", "wall_signboard", "projecting_signboard"]


def load_attribute_names(
    attribute_file: str | None, explicit: str | None, count: int | None, images_dir: str
) -> list[str]:
    """Load attribute names from an explicit list or attribute.yaml."""
    if explicit:
        names = [item.strip() for item in explicit.split(",") if item.strip()]
    else:
        path = (
            Path(attribute_file).expanduser()
            if attribute_file
            else Path(images_dir).parent / "attribute.yaml"
        )
        names = []
        if path.is_file():
            try:
                import yaml

                raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                attrs = raw.get("attributes", {}) if isinstance(raw, dict) else {}
                if isinstance(attrs, dict):
                    names = [str(name) for name in attrs.keys()]
            except ImportError:
                # Minimal fallback for the simple `attributes: name:` schema.
                in_attributes = False
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip() == "attributes:":
                        in_attributes = True
                        continue
                    if in_attributes and re.match(r"^\s{2}[A-Za-z0-9_\-]+:\s*$", line):
                        names.append(line.strip().rstrip(":").strip())
                    elif in_attributes and line and not line.startswith(" "):
                        break
    if not names:
        names = list(DEFAULT_ATTRIBUTE_NAMES)
    if count is not None:
        if count <= 0:
            raise ValueError("--num-attributes must be positive")
        if len(names) != count:
            raise ValueError(
                f"Attribute name count ({len(names)}) does not match --num-attributes ({count})"
            )
    return names


def discover_split_units(
    groups_dir: str, image_index: dict[str, Path]
) -> list[dict[str, object]]:
    """Load merged manual groups and add singleton units for all other images."""
    records = discover_group_records(groups_dir, "merged_group")
    if not records:
        raise ValueError(f"No group subdirectories found in {groups_dir}")
    components = merge_records(records)
    grouped_keys = {key for component in components for key in component["members"]}
    missing = sorted(grouped_keys - set(image_index), key=frame_sort_key)
    if missing:
        raise FileNotFoundError(
            "Merged groups reference images absent from --images-dir: "
            + ", ".join(missing[:20])
        )

    units: list[dict[str, object]] = []
    for component in components:
        units.append(
            {
                "unit_id": component["group_id"],
                "kind": "manual_group",
                "members": list(component["members"]),
                "source_groups": component["source_groups"],
            }
        )
    for key in sorted(set(image_index) - grouped_keys, key=frame_sort_key):
        units.append(
            {
                "unit_id": f"singleton_{key}",
                "kind": "singleton",
                "members": [key],
                "source_groups": [],
            }
        )
    units.sort(key=lambda unit: frame_sort_key(unit["members"][0]))
    return units


def parse_attribute_labels(
    label_file: Path, attribute_start: int, num_attributes: int
) -> tuple[list[bool], int, list[str]]:
    """Read image-level attribute positives from all rows in one label file."""
    positives = [False] * num_attributes
    if not label_file.is_file():
        return positives, 0, ["missing_label_file"]

    rows = 0
    issues: list[str] = []
    for line_number, raw_line in enumerate(
        label_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < attribute_start + num_attributes:
            issues.append(f"line_{line_number}_too_short_{len(fields)}")
            continue
        rows += 1
        for offset in range(num_attributes):
            try:
                value = float(fields[attribute_start + offset])
            except ValueError:
                issues.append(f"line_{line_number}_attribute_{offset}_not_numeric")
                continue
            if value > 0.5:
                positives[offset] = True
            if value not in (0.0, 1.0):
                issues.append(
                    f"line_{line_number}_attribute_{offset}_nonbinary_{value:g}"
                )
    return positives, rows, issues


def parse_ratios(text: str) -> tuple[float, float, float]:
    """Parse and validate train/val/test ratios."""
    values = tuple(float(value.strip()) for value in text.split(","))
    if (
        len(values) != 3
        or any(value < 0 for value in values)
        or not math.isclose(sum(values), 1.0, abs_tol=1e-8)
    ):
        raise ValueError(
            "--ratios must contain three non-negative values summing to 1, e.g. 0.8,0.1,0.1"
        )
    if any(value == 0 for value in values):
        raise ValueError(
            "Each split ratio must be greater than zero so coverage can be evaluated"
        )
    return values  # type: ignore[return-value]


def target_counts(total_images: int, ratios: tuple[float, float, float]) -> list[int]:
    """Convert ratios to integer targets with a largest-remainder rule."""
    raw = [total_images * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    for index in sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)[
        : total_images - sum(counts)
    ]:
        counts[index] += 1
    return counts


def calculate_state(
    assignment: Sequence[int], units: Sequence[dict[str, object]], num_attributes: int
) -> tuple[list[int], list[list[int]]]:
    """Calculate image counts and positive-image counts for each split."""
    counts = [0, 0, 0]
    coverage = [[0] * num_attributes for _ in range(3)]
    for unit, split in zip(units, assignment):
        if split < 0:
            continue
        counts[split] += len(unit["members"])
        positives = unit["attribute_positive"]
        for attr in range(num_attributes):
            coverage[split][attr] += int(positives[attr])
    return counts, coverage


def objective(
    counts: Sequence[int],
    coverage: Sequence[Sequence[int]],
    targets: Sequence[int],
    coverage_priority: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Return a coverage-first objective, optionally prioritizing split order."""
    missing_by_split = [
        sum(value == 0 for value in coverage[split]) for split in range(3)
    ]
    empty = sum(1 for count in counts if count == 0)
    size_error = sum(abs(counts[i] - targets[i]) for i in range(3))
    if coverage_priority is None:
        return sum(missing_by_split), empty, size_error
    return tuple(missing_by_split[split] for split in coverage_priority) + (
        empty,
        size_error,
    )


def choose_initial_assignment(
    units: Sequence[dict[str, object]],
    targets: Sequence[int],
    num_attributes: int,
    seed: int,
    fixed_splits: dict[int, int] | None = None,
    coverage_order: Sequence[int] = (1, 2, 0),
) -> list[int]:
    """Create one deterministic coverage-aware assignment with fixed units."""
    rng = random.Random(seed)
    assignment = [-1] * len(units)
    counts = [0, 0, 0]
    coverage = [[0] * num_attributes for _ in range(3)]
    fixed_splits = fixed_splits or {}
    for index, split in fixed_splits.items():
        if not 0 <= index < len(units):
            raise IndexError(f"Fixed unit index out of range: {index}")
        if split not in range(3):
            raise ValueError(f"Fixed split must be 0, 1, or 2; got {split}")
        assignment[index] = split
        counts[split] += len(units[index]["members"])
        for attr in range(num_attributes):
            coverage[split][attr] += int(units[index]["attribute_positive"][attr])

    frequencies = [
        sum(1 for unit in units if unit["attribute_positive"][attr])
        for attr in range(num_attributes)
    ]
    attr_order = list(range(num_attributes))
    rng.shuffle(attr_order)
    attr_order.sort(key=lambda attr: frequencies[attr])

    def assign(index: int, split: int) -> None:
        assignment[index] = split
        counts[split] += len(units[index]["members"])
        for attr in range(num_attributes):
            coverage[split][attr] += int(units[index]["attribute_positive"][attr])

    # Reserve splits in the caller-requested coverage priority order.
    for split in coverage_order:
        for attr in attr_order:
            if coverage[split][attr] > 0:
                continue
            candidates = [
                index
                for index, unit in enumerate(units)
                if assignment[index] < 0 and unit["attribute_positive"][attr]
            ]
            if not candidates:
                continue

            def rank(index: int) -> tuple[float, float, int, int, int, float]:
                unit = units[index]
                gain = sum(
                    1
                    for other_attr in range(num_attributes)
                    if coverage[split][other_attr] == 0
                    and unit["attribute_positive"][other_attr]
                )
                rare_gain = sum(
                    1.0 / max(frequencies[other_attr], 1)
                    for other_attr in range(num_attributes)
                    if unit["attribute_positive"][other_attr]
                )
                size = len(unit["members"])
                new_count = counts[split] + size
                overshoot = max(0, new_count - targets[split])
                distance = abs(targets[split] - new_count)
                return gain, rare_gain, -overshoot, -distance, -size, rng.random()

            assign(max(candidates, key=rank), split)

    # Fill all remaining units by prioritizing uncovered attributes and deficits.
    remaining = [index for index, split in enumerate(assignment) if split < 0]
    rng.shuffle(remaining)
    for index in remaining:
        unit = units[index]
        options = []
        for split in range(3):
            gain = sum(
                1
                for attr in range(num_attributes)
                if coverage[split][attr] == 0 and unit["attribute_positive"][attr]
            )
            new_count = counts[split] + len(unit["members"])
            deficit = targets[split] - counts[split]
            distance = abs(targets[split] - new_count)
            options.append(
                (gain, deficit, -distance, -len(unit["members"]), rng.random(), split)
            )
        assign(index, max(options)[-1])
    return assignment


def improve_assignment(
    assignment: list[int],
    units: Sequence[dict[str, object]],
    targets: Sequence[int],
    num_attributes: int,
    fixed_splits: dict[int, int] | None = None,
    coverage_priority: Sequence[int] | None = None,
) -> list[int]:
    """Improve image-count fit without moving fixed units."""
    counts, coverage = calculate_state(assignment, units, num_attributes)
    current = objective(counts, coverage, targets, coverage_priority)
    fixed_indices = set((fixed_splits or {}).keys())
    for _ in range(max(100, len(units) * 3)):
        best = current
        best_move = None
        for index, unit in enumerate(units):
            if index in fixed_indices:
                continue
            source = assignment[index]
            for destination in range(3):
                if destination == source:
                    continue
                candidate_counts = counts[:]
                candidate_coverage = [row[:] for row in coverage]
                size = len(unit["members"])
                candidate_counts[source] -= size
                candidate_counts[destination] += size
                for attr in range(num_attributes):
                    value = int(unit["attribute_positive"][attr])
                    candidate_coverage[source][attr] -= value
                    candidate_coverage[destination][attr] += value
                candidate = objective(
                    candidate_counts, candidate_coverage, targets, coverage_priority
                )
                if candidate < best:
                    best = candidate
                    best_move = (
                        index,
                        source,
                        destination,
                        candidate_counts,
                        candidate_coverage,
                    )
        if best_move is None:
            break
        index, _, destination, counts, coverage = best_move
        assignment[index] = destination
        current = best
    return assignment


def assign_units(
    units: list[dict[str, object]],
    ratios: tuple[float, float, float],
    seed: int,
    attempts: int,
    fixed_splits: dict[int, int] | None = None,
    coverage_priority: Sequence[int] | None = None,
) -> tuple[list[int], list[int], tuple[int, ...]]:
    """Find a coverage-first assignment while keeping fixed units in place."""
    total = sum(len(unit["members"]) for unit in units)
    targets = target_counts(total, ratios)
    num_attributes = len(units[0]["attribute_positive"])
    fixed_splits = fixed_splits or {}
    if any(index < 0 or index >= len(units) for index in fixed_splits):
        raise IndexError("A fixed unit index is outside the unit list")
    if any(split not in range(3) for split in fixed_splits.values()):
        raise ValueError("Fixed split values must be 0, 1, or 2")
    best_assignment: list[int] | None = None
    best_objective: tuple[int, ...] | None = None
    coverage_order = (
        tuple(coverage_priority) if coverage_priority is not None else (1, 2, 0)
    )
    for attempt in range(max(1, attempts)):
        assignment = choose_initial_assignment(
            units,
            targets,
            num_attributes,
            seed + attempt,
            fixed_splits=fixed_splits,
            coverage_order=coverage_order,
        )
        assignment = improve_assignment(
            assignment,
            units,
            targets,
            num_attributes,
            fixed_splits=fixed_splits,
            coverage_priority=coverage_priority,
        )
        counts, coverage = calculate_state(assignment, units, num_attributes)
        current = objective(counts, coverage, targets, coverage_priority)
        if best_objective is None or current < best_objective:
            best_assignment, best_objective = assignment[:], current
        missing = sum(sum(value == 0 for value in row) for row in coverage)
        empty = sum(count == 0 for count in counts)
        if missing == 0 and empty == 0:
            # More attempts cannot improve the primary objective; retain the first
            # valid coverage solution, whose size error is already locally optimized.
            break
    if best_assignment is None or best_objective is None:
        raise RuntimeError("Could not construct a group assignment")
    final_counts, _ = calculate_state(best_assignment, units, num_attributes)
    return best_assignment, final_counts, best_objective


def attribute_coverage_rows(
    names: Sequence[str], coverage: Sequence[Sequence[int]], total: Sequence[int]
) -> list[list[object]]:
    """Build tabular positive-image coverage rows."""
    rows: list[list[object]] = []
    for attr, name in enumerate(names):
        values = [coverage[split][attr] for split in range(3)]
        missing = ";".join(
            part for part, value in zip(("train", "val", "test"), values) if value == 0
        )
        rows.append([name, total[attr], *values, missing])
    return rows


ATTRIBUTE_COVERAGE_HEADERS = [
    "attribute",
    "total_positive_images",
    "train",
    "val",
    "test",
    "missing_from",
]


def write_attribute_csv(
    path: Path,
    names: Sequence[str],
    coverage: Sequence[Sequence[int]],
    total: Sequence[int],
) -> None:
    """Write positive-image coverage as a CSV table."""
    rows = attribute_coverage_rows(names, coverage, total)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(ATTRIBUTE_COVERAGE_HEADERS)
        writer.writerows(rows)


def write_attribute_markdown(
    path: Path,
    names: Sequence[str],
    coverage: Sequence[Sequence[int]],
    total: Sequence[int],
) -> None:
    """Write positive-image coverage as a Markdown table."""
    rows = attribute_coverage_rows(names, coverage, total)
    with path.open("w", encoding="utf-8") as file:
        file.write("| " + " | ".join(ATTRIBUTE_COVERAGE_HEADERS) + " |\n")
        file.write(
            "| " + " | ".join("---" for _ in ATTRIBUTE_COVERAGE_HEADERS) + " |\n"
        )
        for row in rows:
            file.write("| " + " | ".join(str(value) for value in row) + " |\n")


def print_attribute_coverage_table(
    names: Sequence[str], coverage: Sequence[Sequence[int]], total: Sequence[int]
) -> None:
    """Print positive-image coverage as an aligned terminal table."""
    rows = [
        [str(value) for value in row]
        for row in attribute_coverage_rows(names, coverage, total)
    ]
    values = [[str(value) for value in ATTRIBUTE_COVERAGE_HEADERS], *rows]
    widths = [
        max(len(row[index]) for row in values)
        for index in range(len(ATTRIBUTE_COVERAGE_HEADERS))
    ]
    print("Attribute coverage (positive images):")
    print(
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(values[0]))
    )
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def write_split_yaml(
    path: Path,
    yolo_dir: Path,
    class_names: Sequence[str],
    attribute_names: Sequence[str],
    nal: int,
) -> None:
    """Write a self-contained mdet dataset YAML without hard-coded paths."""
    lines = [
        f"path: {json.dumps(yolo_dir.as_posix(), ensure_ascii=False)}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(
        f"  {index}: {json.dumps(name, ensure_ascii=False)}"
        for index, name in enumerate(class_names)
    )
    lines.extend([f"nal: {nal}", "attributes:"])
    for name in attribute_names:
        lines.extend([f"  {name}:", "    - no", "    - yes"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_yolo_split(
    yolo_dir: Path,
    split_paths: dict[str, list[Path]],
    labels_dir: Path,
    class_names: Sequence[str],
    attribute_names: Sequence[str],
    nal: int,
    link: bool,
) -> list[str]:
    """Copy/link images and labels into a YOLO directory."""
    missing_labels: list[str] = []
    for split in ("train", "val", "test"):
        image_dir = yolo_dir / "images" / split
        label_dir = yolo_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image in split_paths[split]:
            target_image = image_dir / image.name
            copy_or_link(image, target_image, link)
            label = labels_dir / f"{image.stem}.txt"
            if label.is_file():
                copy_or_link(label, label_dir / label.name, link)
            else:
                missing_labels.append(str(label))
    write_split_yaml(
        yolo_dir / "data.yaml", yolo_dir, class_names, attribute_names, nal
    )
    return missing_labels


def split_dataset(args: argparse.Namespace) -> Path:
    """Assign manual groups and write split reports/dataset files."""
    ratios = parse_ratios(args.ratios)
    if args.nal < 2:
        raise ValueError(
            "--nal must be at least 2; use 2 for the current binary attributes"
        )
    image_index, duplicate_images = index_images(args.images_dir)
    if duplicate_images:
        raise ValueError(
            "Duplicate image keys in --images-dir: " + ", ".join(duplicate_images[:20])
        )
    attribute_names = load_attribute_names(
        args.attribute_file, args.attribute_names, args.num_attributes, args.images_dir
    )
    num_attributes = len(attribute_names)
    units = discover_split_units(args.groups_dir, image_index)
    labels_dir = Path(args.labels_dir).expanduser()

    label_issues: dict[str, list[str]] = {}
    image_positive: dict[str, list[bool]] = {}
    image_rows: dict[str, int] = {}
    for key, image in image_index.items():
        positives, rows, issues = parse_attribute_labels(
            labels_dir / f"{image.stem}.txt", args.attribute_start, num_attributes
        )
        image_positive[key] = positives
        image_rows[key] = rows
        if issues:
            label_issues[key] = issues

    for unit in units:
        unit["attribute_positive"] = [
            sum(1 for key in unit["members"] if image_positive[key][attr])
            for attr in range(num_attributes)
        ]

    fixed_splits = {
        index: 0
        for index, unit in enumerate(units)
        if args.manual_groups_split == "train" and unit["kind"] == "manual_group"
    }
    # With manual groups fixed in train, preserve the requested priority:
    # train coverage first, then test coverage, and val coverage last.
    coverage_priority = (0, 2, 1) if args.manual_groups_split == "train" else None
    assignment, counts, assignment_objective = assign_units(
        units,
        ratios,
        args.seed,
        args.attempts,
        fixed_splits=fixed_splits,
        coverage_priority=coverage_priority,
    )
    targets = target_counts(len(image_index), ratios)
    _, coverage = calculate_state(assignment, units, num_attributes)
    total_coverage = [
        sum(image_positive[key][attr] for key in image_index)
        for attr in range(num_attributes)
    ]
    gaps = {
        attribute_names[attr]: [
            part
            for part, value in zip(
                ("train", "val", "test"), [coverage[s][attr] for s in range(3)]
            )
            if value == 0
        ]
        for attr in range(num_attributes)
    }
    gaps = {name: missing for name, missing in gaps.items() if missing}
    missing_by_split = {
        split_name: sum(value == 0 for value in coverage[split])
        for split, split_name in enumerate(("train", "val", "test"))
    }

    split_names = ("train", "val", "test")
    split_keys = {name: [] for name in split_names}
    for unit, split in zip(units, assignment):
        split_keys[split_names[split]].extend(unit["members"])
    for name in split_names:
        split_keys[name].sort(key=frame_sort_key)

    output = prepare_output(args.out_dir, args.overwrite)
    lists_dir = output / "lists"
    lists_dir.mkdir()
    for split in split_names:
        with (lists_dir / f"{split}.txt").open("w", encoding="utf-8") as file:
            for key in split_keys[split]:
                file.write(str(image_index[key].resolve()) + "\n")

    split_paths = {
        split: [image_index[key] for key in split_keys[split]] for split in split_names
    }
    with (output / "group_assignments.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["unit_id", "kind", "split", "size", "members", "source_groups"]
        )
        for unit, split in zip(units, assignment):
            source_groups = ";".join(
                f"{item['source']}:{item['folder']}" for item in unit["source_groups"]
            )
            writer.writerow(
                [
                    unit["unit_id"],
                    unit["kind"],
                    split_names[split],
                    len(unit["members"]),
                    ";".join(unit["members"]),
                    source_groups,
                ]
            )
    write_attribute_csv(
        output / "attribute_coverage.csv", attribute_names, coverage, total_coverage
    )
    write_attribute_markdown(
        output / "attribute_coverage.md", attribute_names, coverage, total_coverage
    )

    missing_labels = []
    if args.make_yolo:
        missing_labels = write_yolo_split(
            output / "yolo",
            split_paths,
            labels_dir,
            load_class_names(args.class_file, args.images_dir),
            attribute_names,
            args.nal,
            args.link,
        )

    report = {
        "schema_version": 1,
        "kind": "manual_group_dataset_split",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "images_dir": str(Path(args.images_dir).expanduser().resolve()),
        "labels_dir": str(labels_dir.resolve()),
        "groups_dir": str(Path(args.groups_dir).expanduser().resolve()),
        "out_dir": str(output),
        "ratios": list(ratios),
        "target_image_counts": dict(zip(split_names, targets)),
        "actual_image_counts": dict(zip(split_names, counts)),
        "seed": args.seed,
        "attempts": args.attempts,
        "attribute_start": args.attribute_start,
        "nal": args.nal,
        "attribute_names": attribute_names,
        "num_images": len(image_index),
        "num_units": len(units),
        "manual_group_units": sum(unit["kind"] == "manual_group" for unit in units),
        "singleton_units": sum(unit["kind"] == "singleton" for unit in units),
        "manual_groups_split": args.manual_groups_split,
        "manual_group_train_units": len(fixed_splits),
        "manual_group_train_images": sum(
            len(units[index]["members"]) for index in fixed_splits
        ),
        "assignment_objective": {
            "coverage_priority": ["train", "test", "val"]
            if coverage_priority is not None
            else ["total_missing"],
            "missing_attribute_split_pairs": sum(missing_by_split.values()),
            "missing_attributes_by_split": missing_by_split,
            "empty_splits": sum(count == 0 for count in counts),
            "absolute_image_count_error": sum(
                abs(counts[index] - targets[index]) for index in range(3)
            ),
        },
        "attribute_coverage": {
            attribute_names[attr]: {
                "total_positive_images": total_coverage[attr],
                **{split_names[split]: coverage[split][attr] for split in range(3)},
            }
            for attr in range(num_attributes)
        },
        "attribute_coverage_gaps": gaps,
        "label_issue_image_count": len(label_issues),
        "label_issues": label_issues,
        "missing_labels_when_copied": missing_labels,
        "duplicate_images": duplicate_images,
        "make_yolo": args.make_yolo,
        "link": args.link,
    }
    write_json(output / "split_report.json", report)
    print_attribute_coverage_table(attribute_names, coverage, total_coverage)
    print(f"CSV coverage table: {output / 'attribute_coverage.csv'}")
    print(f"Markdown coverage table: {output / 'attribute_coverage.md'}")
    print(f"JSON report: {output / 'split_report.json'}")
    print(f"Split output created: {output}")
    if gaps:
        print(
            "WARNING: some attributes could not be present in all three splits; see attribute_coverage.csv"
        )
    if label_issues:
        print("WARNING: label parsing issues were found; see split_report.json")
    return output


def build_parser() -> argparse.ArgumentParser:
    """Build the merge/split command-line interface."""
    parser = argparse.ArgumentParser(
        description="Merge manually reviewed groups and split mdet data without leakage"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge = subparsers.add_parser(
        "merge-groups", help="merge group_src and group into group_merged"
    )
    merge.add_argument("--group-src", default=DEFAULT_GROUP_SRC)
    merge.add_argument("--group-dir", default=DEFAULT_GROUP)
    merge.add_argument("--images-dir", default=DEFAULT_IMAGES)
    merge.add_argument("--out-dir", default=DEFAULT_MERGED_GROUP)
    merge.add_argument(
        "--link",
        action="store_true",
        help="create links instead of copying original images",
    )
    merge.add_argument("--overwrite", action="store_true")
    merge.add_argument("--dry-run", action="store_true")

    split = subparsers.add_parser(
        "split", help="split the merged groups into train/val/test"
    )
    split.add_argument("--images-dir", default=DEFAULT_IMAGES)
    split.add_argument("--labels-dir", default=DEFAULT_LABELS)
    split.add_argument("--groups-dir", default=DEFAULT_MERGED_GROUP)
    split.add_argument("--out-dir", default=DEFAULT_SPLIT_OUT)
    split.add_argument("--ratios", default="0.80,0.10,0.10")
    split.add_argument("--seed", type=int, default=42)
    split.add_argument(
        "--attempts", type=int, default=32, help="randomized initial assignments to try"
    )
    split.add_argument(
        "--manual-groups-split",
        "--group-split",
        dest="manual_groups_split",
        choices=("train", "any"),
        default="train",
        help="placement policy for merged manual groups; default: all in train",
    )
    split.add_argument("--attribute-file", default=None)
    split.add_argument(
        "--attribute-names",
        default=None,
        help="comma-separated names, overrides attribute.yaml",
    )
    split.add_argument("--num-attributes", type=int, default=None)
    split.add_argument(
        "--attribute-start",
        type=int,
        default=2,
        help="first attribute column in each label row",
    )
    split.add_argument(
        "--nal",
        type=int,
        default=2,
        help="number of levels per attribute; 2 for binary attributes",
    )
    split.add_argument("--class-file", default=None)
    split.add_argument(
        "--make-yolo",
        action="store_true",
        help="copy/link images and labels and write yolo/data.yaml",
    )
    split.add_argument(
        "--link",
        action="store_true",
        help="use links instead of copying for --make-yolo",
    )
    split.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected command."""
    args = build_parser().parse_args(argv)
    if args.command == "merge-groups":
        merge_groups(args)
    elif args.command == "split":
        split_dataset(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
