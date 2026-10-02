import json
from pathlib import Path

from PIL import Image

from yolo_data_manager.cli import main as cli_main
from yolo_data_manager.io.loader import load_yolo_dataset
from yolo_data_manager.scripting import build_task_argv
from yolo_data_manager.tools.filename_remap import (
    filename_digits_for_count,
    remap_yolo_dataset_filenames,
)


def _make_dataset(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "class.txt").write_text("object\n", encoding="utf-8")
    Image.new("RGB", (20, 20), "red").save(root / "images" / "a.jpg")
    Image.new("RGB", (20, 20), "blue").save(root / "images" / "b.png")
    (root / "labels" / "a.txt").write_text(
        "0 0.5 0.5 0.2 0.2 0.91\n",
        encoding="utf-8",
    )
    (root / "labels" / "b.txt").write_text(
        "0 0.4 0.4 0.3 0.3\n",
        encoding="utf-8",
    )
    (root / "train.txt").write_text("images/a.jpg\n", encoding="utf-8")
    (root / "val.txt").write_text("images/b.png\n", encoding="utf-8")
    return root


def test_filename_digits_uses_ten_times_image_count():
    assert filename_digits_for_count(8951) == 6
    assert filename_digits_for_count(0) == 1


def test_remap_yolo_dataset_filenames_copies_labels_and_writes_mapping(tmp_path):
    source = _make_dataset(tmp_path / "source")
    dataset = load_yolo_dataset(source, layout="auto", workers=1)
    output = tmp_path / "remapped"

    result = remap_yolo_dataset_filenames(
        dataset,
        output,
        digits=4,
        start=10,
        workers=1,
        progress=False,
    )

    assert result.images == 2
    assert result.labels == 2
    assert result.digits == 4
    assert (output / "images" / "0010.jpg").is_file()
    assert (output / "images" / "0011.png").is_file()
    assert (output / "labels" / "0010.txt").read_text(encoding="utf-8") == (
        "0 0.5 0.5 0.2 0.2 0.91\n"
    )
    assert (output / "train.txt").read_text(encoding="utf-8") == "images/0010.jpg\n"
    assert (output / "val.txt").read_text(encoding="utf-8") == "images/0011.png\n"

    mapping = json.loads((output / "filename_mapping.json").read_text(encoding="utf-8"))
    assert mapping["digits"] == 4
    assert mapping["image_mapping"] == {
        "images/a.jpg": "images/0010.jpg",
        "images/b.png": "images/0011.png",
    }
    assert mapping["label_mapping"] == {
        "labels/a.txt": "labels/0010.txt",
        "labels/b.txt": "labels/0011.txt",
    }


def test_filename_remap_cli_and_python_task_argv(tmp_path, capsys):
    source = _make_dataset(tmp_path / "source")
    output = tmp_path / "cli_remapped"
    argv = build_task_argv(
        "convert.filename_remap",
        root=source,
        out=output,
        digits=6,
        start=1,
    )
    assert argv[:2] == ["convert", "filename-remap"]
    assert "--digits" in argv
    assert "6" in argv

    assert cli_main(
        [
            "convert",
            "filename-remap",
            "--root",
            str(source),
            "--out",
            str(output),
            "--digits",
            "6",
            "--start",
            "1",
            "--workers",
            "1",
            "--no-progress",
        ]
    ) == 0
    assert (output / "images" / "000001.jpg").is_file()
    assert '"images": 2' in capsys.readouterr().out
