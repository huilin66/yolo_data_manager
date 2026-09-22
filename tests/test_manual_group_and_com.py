import csv
import json
from pathlib import Path

from PIL import Image

from yolo_data_manager import YoloManager


def _write_attribute_schema(root: Path) -> None:
    (root / "attribute.yaml").write_text(
        "attributes:\n"
        "  surface_corroded:\n"
        "    - no\n"
        "    - yes\n"
        "  frame_corroded:\n"
        "    - no\n"
        "    - yes\n",
        encoding="utf-8",
    )


def test_manager_generate_attribute_com_matches_script_formula(tmp_path):
    root = tmp_path / "com_dataset"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    _write_attribute_schema(root)
    Image.new("RGB", (20, 20), color="white").save(root / "images" / "a.jpg")
    Image.new("RGB", (20, 20), color="white").save(root / "images" / "b.jpg")
    (root / "labels" / "a.txt").write_text(
        "0 2 1 0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (root / "labels" / "b.txt").write_text(
        "0 2 1 1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (root / "train.txt").write_text("images/a.jpg\nimages/b.jpg\n", encoding="utf-8")

    manager = YoloManager(root, init_layout=False, init_check=False)
    output = root / "co_occurrence.csv"
    summary = root / "co_occurrence_summary.json"

    assert manager.generate_attribute_com(output=output, summary=summary) == 0

    with output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    assert rows[0] == ["", "surface_corroded", "frame_corroded"]
    assert rows[1][0] == "surface_corroded"
    assert rows[1][1:] == ["1", "0.7071067812"]
    assert rows[2][0] == "frame_corroded"
    assert rows[2][1:] == ["0.7071067812", "1"]

    report = json.loads(summary.read_text(encoding="utf-8"))
    assert report["split"] == "train"
    assert report["annotation_rows"] == 2
    assert report["attribute_names"] == ["surface_corroded", "frame_corroded"]


def test_manager_manual_group_merge_and_split_outputs(tmp_path):
    root = tmp_path / "manual_group_dataset"
    images = root / "images"
    labels = root / "labels"
    group_src = root / "group_src"
    group_dir = root / "group"
    images.mkdir(parents=True)
    labels.mkdir()
    (group_src / "review_a").mkdir(parents=True)
    (group_dir / "review_b").mkdir(parents=True)
    (group_src / "review_c").mkdir(parents=True)
    _write_attribute_schema(root)
    (root / "class.txt").write_text("wall_signboard\n", encoding="utf-8")

    for index in range(1, 5):
        name = f"FLIR{index:04d}.png"
        Image.new("RGB", (20, 20), color="white").save(images / name)
        (labels / f"FLIR{index:04d}.txt").write_text(
            f"0 2 {int(index % 2 == 0)} {int(index % 3 == 0)} 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )

    Image.new("RGB", (5, 5), color="white").save(
        group_src / "review_a" / "FLIR0001_wall_signboard_det0001.png"
    )
    Image.new("RGB", (5, 5), color="white").save(
        group_dir / "review_b" / "FLIR0001_projecting_signboard_det0001.png"
    )
    Image.new("RGB", (5, 5), color="white").save(
        group_src / "review_c" / "FLIR0002_wall_signboard_det0001.png"
    )

    manager = YoloManager(root, init_layout=False, init_check=False)
    merged = manager.merge_manual_groups(
        group_src=group_src,
        group_dir=group_dir,
        images_dir=images,
        out_dir=root / "group_merged",
    )
    assert (merged / "group_manifest.json").is_file()
    manifest = json.loads((merged / "group_manifest.json").read_text(encoding="utf-8"))
    assert manifest["merged_images"] == 2
    assert manifest["merged_groups"] == 2

    split = manager.split_by_manual_group(
        images_dir=images,
        labels_dir=labels,
        groups_dir=merged,
        out_dir=root / "manual_group_split",
        ratios="0.50,0.25,0.25",
        make_yolo=True,
    )
    assert (split / "lists" / "train.txt").is_file()
    assert (split / "attribute_coverage.csv").is_file()
    assert (split / "split_report.json").is_file()
    assert (split / "yolo" / "data.yaml").is_file()
    report = json.loads((split / "split_report.json").read_text(encoding="utf-8"))
    assert report["kind"] == "manual_group_dataset_split"
    assert report["manual_group_units"] == 2
    assert report["singleton_units"] == 2
