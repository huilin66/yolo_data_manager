"""Generate a GCA-compatible attribute co-occurrence matrix.

The current MAYOLO mdet labels use the following layout by default::

    class meta attr1 ... attr10 cx cy w h

The second column (``meta``) is the number of attributes, not an object
class.  This script computes co-occurrence from the attribute columns and
writes a CSV that can be loaded by ``ultralytics.nn.modules.head.GAT`` with
``pd.read_csv(path, header=0, index_col=0)``.

The historical ``co_occurrence_matrix6.csv`` used by MAYOLO is the positive
attribute co-occurrence matrix normalized by the square root of the two
diagonal entries::

    C[i, j] / sqrt(C[i, i] * C[j, j])

This is the default ``--mode cross``.  It produces an ``na x na`` matrix,
which is the shape expected by the current ``com_gat`` implementation.

For the paper-defined directed GCA prior, ``--mode conditional`` writes
``P(attribute_j=1 | attribute_i=1)`` row by row.  The optional
``--smoothing`` applies Laplace smoothing to the off-diagonal Bernoulli
probabilities; the diagonal remains one whenever the conditioning attribute
appears in the training split.  This keeps rare-attribute rows from becoming
extreme while preserving the target-row/source-column direction used by GCA.

Examples (PowerShell)::

    python generate_com.py \
        --data-root "\\\\158.132.186.40\\isds\\huilin\\mayolo\\mayolo_v3" \
        --split train \
        --mode conditional \
        --smoothing 1.0 \
        --output "\\\\158.132.186.40\\isds\\huilin\\mayolo\\mayolo_v3\\co_occurrence_matrix_train_conditional.csv"

The default is now ``--split train`` so that validation and test annotations
do not enter the training prior.  Use ``--split all`` explicitly when the
historical all-data matrix is required.  The input images and labels are
never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_DATA_ROOT = r"\\158.132.186.40\isds\huilin\mayolo\mayolo_v3"
DEFAULT_ATTRIBUTE_START = 2
DEFAULT_MODE = "cross"
DEFAULT_SPLIT = "train"


def _as_path(value: str | Path) -> Path:
    """Return a path without resolving a UNC path through the local cwd."""
    return Path(value).expanduser()


def _read_yaml_attribute_names(path: Path) -> list[str]:
    """Read attribute keys from the small project YAML schema.

    PyYAML is preferred when available.  The fallback intentionally handles
    the simple ``attributes:`` mapping used by this dataset, so the utility
    remains usable in a minimal Python environment.
    """
    if not path.is_file():
        return []

    try:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        attributes = raw.get("attributes", {}) if isinstance(raw, dict) else {}
        if isinstance(attributes, dict):
            return [str(name) for name in attributes]
        if isinstance(attributes, list):
            return [str(name) for name in attributes]
        return []
    except ImportError:
        names: list[str] = []
        in_attributes = False
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped == "attributes:":
                in_attributes = True
                continue
            if in_attributes and re.match(r"^\s{2}[A-Za-z0-9_\-]+:\s*$", line):
                names.append(stripped.rstrip(":").strip().strip('"\''))
            elif in_attributes and line and not line.startswith((" ", "\t")):
                break
        return names


def find_attribute_file(data_root: Path, labels_dir: Path, explicit: Optional[str]) -> Optional[Path]:
    """Find the attribute schema, preferring an explicit path."""
    if explicit:
        path = _as_path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Attribute file not found: {path}")
        return path

    for candidate in (data_root / "attribute.yaml", data_root / "attributes.yaml", labels_dir / "attributes.yaml"):
        if candidate.is_file():
            return candidate
    return None


def load_attribute_names(
    data_root: Path,
    labels_dir: Path,
    attribute_file: Optional[str],
    requested_count: Optional[int],
) -> tuple[list[str], Optional[Path]]:
    """Load names and validate an optional explicit attribute count."""
    schema_path = find_attribute_file(data_root, labels_dir, attribute_file)
    names = _read_yaml_attribute_names(schema_path) if schema_path else []

    if requested_count is not None:
        if requested_count <= 0:
            raise ValueError("--num-attributes must be positive")
        if names and len(names) != requested_count:
            raise ValueError(
                f"Attribute name count ({len(names)}) does not match "
                f"--num-attributes ({requested_count})"
            )
        count = requested_count
    elif names:
        count = len(names)
    else:
        count = 0

    if count and not names:
        names = [f"attribute_{index}" for index in range(count)]
    return names, schema_path


def label_files(labels_dir: Path) -> list[Path]:
    """Return annotation TXT files, excluding metadata TXT files."""
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    excluded = {"class.txt", "classes.txt", "attribute.txt", "attributes.txt"}
    files = [
        path
        for path in labels_dir.rglob("*.txt")
        if path.name.lower() not in excluded
    ]
    return sorted(files, key=lambda path: str(path).lower())


def split_stems(data_root: Path, split: str) -> Optional[set[str]]:
    """Load image stems from ``train.txt``, ``val.txt`` or ``test.txt``.

    The dataset list files contain absolute paths from the training machine,
    so only the basename stem is used when matching label files.
    """
    if split == "all":
        return None

    split_file = data_root / f"{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(
            f"Split file not found: {split_file}. Use --split all or provide "
            "the corresponding train/val/test list."
        )

    stems: set[str] = set()
    for raw_line in split_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            stems.add(Path(line).stem.lower())
    if not stems:
        raise ValueError(f"Split file is empty: {split_file}")
    return stems


def selected_label_files(
    files: Iterable[Path],
    selected_stems: Optional[set[str]],
) -> list[Path]:
    """Filter label files by image stem when a split was requested."""
    if selected_stems is None:
        return list(files)
    return [path for path in files if path.stem.lower() in selected_stems]


def parse_label_file(
    path: Path,
    attribute_start: int,
    num_attributes: int,
) -> tuple[list[list[float]], list[str]]:
    """Read positive attribute vectors from one mdet label file."""
    vectors: list[list[float]] = []
    issues: list[str] = []
    required = attribute_start + num_attributes

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < required:
            issues.append(f"{path.name}:line {line_number}: only {len(fields)} columns")
            continue

        values: list[float] = []
        for offset in range(num_attributes):
            field = fields[attribute_start + offset]
            try:
                value = float(field)
            except ValueError as error:
                raise ValueError(
                    f"Non-numeric attribute in {path} line {line_number}, "
                    f"column {attribute_start + offset}: {field!r}"
                ) from error
            if not math.isfinite(value) or value < 0 or value > 1:
                raise ValueError(
                    f"Attribute value must be in [0, 1] for COM generation; "
                    f"got {field!r} in {path} line {line_number}"
                )
            values.append(value)
        vectors.append(values)
    return vectors, issues


def infer_attribute_count(files: Iterable[Path], attribute_start: int) -> int:
    """Infer ``na`` from the second label column when no schema is available."""
    for path in files:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            fields = raw_line.split("#", 1)[0].split()
            if not fields:
                continue
            if len(fields) <= 1:
                continue
            try:
                count = int(float(fields[1]))
            except ValueError:
                continue
            if count > 0 and len(fields) >= attribute_start + count:
                return count
    raise ValueError(
        "Could not infer the number of attributes. Pass --num-attributes "
        "or provide attribute.yaml."
    )


def calculate_matrix(
    vectors: list[list[float]],
    mode: str,
    smoothing: float = 0.0,
) -> list[list[float]]:
    """Calculate count, probability, cross-normalized, or conditional matrices."""
    if not vectors:
        raise ValueError("No valid annotation rows were found")
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
    num_attributes = len(vectors[0])
    counts = [[0.0 for _ in range(num_attributes)] for _ in range(num_attributes)]
    for vector in vectors:
        for row in range(num_attributes):
            for column in range(num_attributes):
                counts[row][column] += vector[row] * vector[column]

    if mode == "count":
        return counts
    if mode == "probability":
        denominator = float(len(vectors))
        return [[value / denominator for value in row] for row in counts]
    if mode == "conditional":
        matrix = [[0.0 for _ in range(num_attributes)] for _ in range(num_attributes)]
        for row in range(num_attributes):
            positive_count = counts[row][row]
            denominator = positive_count + 2.0 * smoothing
            for column in range(num_attributes):
                if row == column:
                    # P(i|i) is known to be one when attribute i has positive
                    # support; zero marks a truly unseen conditioning row.
                    matrix[row][column] = 1.0 if positive_count else 0.0
                elif denominator:
                    matrix[row][column] = (counts[row][column] + smoothing) / denominator
                else:
                    matrix[row][column] = 0.0
        return matrix
    if mode != "cross":
        raise ValueError(f"Unsupported matrix mode: {mode}")

    diagonal = [counts[index][index] for index in range(num_attributes)]
    matrix = [[0.0 for _ in range(num_attributes)] for _ in range(num_attributes)]
    for row in range(num_attributes):
        for column in range(num_attributes):
            denominator = math.sqrt(diagonal[row] * diagonal[column])
            matrix[row][column] = counts[row][column] / denominator if denominator else 0.0
    return matrix


def write_matrix(path: Path, names: list[str], matrix: list[list[float]]) -> None:
    """Write a pandas-compatible indexed CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([""] + names)
        for name, row in zip(names, matrix):
            writer.writerow([name] + [f"{value:.10g}" for value in row])


def write_summary(path: Path, summary: dict[str, object]) -> None:
    """Write a machine-readable generation summary next to the CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _check_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists; pass --overwrite explicitly: {path}"
        )


def generate(args: argparse.Namespace) -> int:
    """Generate the requested COM CSV and return a process status code."""
    data_root = _as_path(args.data_root)
    labels_dir = _as_path(args.labels_dir) if args.labels_dir else data_root / "labels"
    files = label_files(labels_dir)
    if not files:
        raise FileNotFoundError(f"No label TXT files found in: {labels_dir}")

    selected_stems = split_stems(data_root, args.split)
    files = selected_label_files(files, selected_stems)
    if not files:
        raise ValueError(
            f"No label files from {labels_dir} match --split {args.split!r}"
        )

    names, schema_path = load_attribute_names(
        data_root, labels_dir, args.attribute_file, args.num_attributes
    )
    if not names:
        inferred = infer_attribute_count(files, args.attribute_start)
        names = [f"attribute_{index}" for index in range(inferred)]
    num_attributes = len(names)

    vectors: list[list[float]] = []
    issue_messages: list[str] = []
    files_with_rows = 0
    for path in files:
        file_vectors, issues = parse_label_file(path, args.attribute_start, num_attributes)
        issue_messages.extend(issues)
        if file_vectors:
            files_with_rows += 1
            if args.filter_background:
                file_vectors = [vector for vector in file_vectors if sum(vector) != 0]
            vectors.extend(file_vectors)

    if issue_messages:
        preview = "; ".join(issue_messages[:5])
        suffix = " ..." if len(issue_messages) > 5 else ""
        raise ValueError(f"Malformed label rows ({len(issue_messages)}): {preview}{suffix}")

    matrix = calculate_matrix(vectors, args.mode, args.smoothing)
    if args.output:
        output = _as_path(args.output)
    else:
        if args.mode == "conditional":
            default_name = f"co_occurrence_matrix_{args.split}_conditional.csv"
        else:
            default_name = (
                "co_occurrence_matrix6.csv"
                if args.split == "all"
                else f"co_occurrence_matrix_{args.split}.csv"
            )
        output = data_root / default_name
    summary_path = (
        _as_path(args.summary)
        if args.summary
        else output.with_name(f"{output.stem}_summary.json")
    )
    diagonal = [matrix[index][index] for index in range(num_attributes)]
    summary: dict[str, object] = {
        "schema_version": 1,
        "data_root": str(data_root),
        "labels_dir": str(labels_dir),
        "attribute_file": str(schema_path) if schema_path else None,
        "split": args.split,
        "matrix_mode": args.mode,
        "matrix_smoothing": args.smoothing,
        "filter_background": args.filter_background,
        "attribute_start": args.attribute_start,
        "attribute_names": names,
        "num_attributes": num_attributes,
        "label_files": len(files),
        "files_with_rows": files_with_rows,
        "annotation_rows": sum(1 for _ in vectors),
        "diagonal": diagonal,
        "matrix": matrix,
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    _check_output(output, args.overwrite)
    _check_output(summary_path, args.overwrite)
    write_matrix(output, names, matrix)
    write_summary(summary_path, summary)

    print(f"COM written: {output}")
    print(f"Summary written: {summary_path}")
    print(
        f"split={args.split}, files={len(files)}, rows={len(vectors)}, "
        f"attributes={num_attributes}, mode={args.mode}, smoothing={args.smoothing}"
    )
    print("diagonal:", ", ".join(f"{name}={value:.6f}" for name, value in zip(names, diagonal)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Dataset root")
    parser.add_argument("--labels-dir", default=None, help="Label directory; defaults to <data-root>/labels")
    parser.add_argument("--attribute-file", default=None, help="attribute.yaml or attributes.yaml")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV; conditional mode appends _conditional to the split name",
    )
    parser.add_argument("--summary", default=None, help="Summary JSON; defaults beside the output CSV")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default=DEFAULT_SPLIT)
    parser.add_argument(
        "--mode",
        choices=("cross", "count", "probability", "conditional"),
        default=DEFAULT_MODE,
        help="matrix representation; conditional is the directed P(j|i) GCA prior",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="Laplace smoothing for --mode conditional (default: 0)",
    )
    parser.add_argument("--attribute-start", type=int, default=DEFAULT_ATTRIBUTE_START)
    parser.add_argument("--num-attributes", type=int, default=None)
    parser.add_argument(
        "--filter-background",
        action="store_true",
        help="Exclude rows whose attribute vector is all zero",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output files")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the matrix without writing files")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
