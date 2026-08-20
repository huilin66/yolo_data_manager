from pathlib import Path

from PIL import Image

from yolo_data_manager.cli import main as cli_main
from yolo_data_manager.io.loader import load_yolo_dataset
from yolo_data_manager.scripting import YoloManager, build_task_argv
from yolo_data_manager.tools.image_resize import resize_yolo_dataset


def _make_dataset(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    Image.new("RGB", (100, 50), (255, 0, 0)).save(root / "images" / "sample.jpg")
    (root / "labels" / "sample.txt").write_text(
        "0 0.5 0.5 0.2 0.4\n",
        encoding="utf-8",
    )
    (root / "class.txt").write_text("object\n", encoding="utf-8")
    return root


def test_resize_yolo_dataset_letterbox_transforms_boxes(tmp_path):
    source = _make_dataset(tmp_path / "source")
    dataset = load_yolo_dataset(source, layout="auto", workers=1)

    result = resize_yolo_dataset(
        dataset,
        tmp_path / "resized",
        width=200,
        height=200,
        keep_ratio=True,
        workers=1,
        progress=False,
    )

    assert result.images == 1
    assert result.annotations == 1
    assert result.letterboxed_images == 1
    with Image.open(tmp_path / "resized" / "images" / "sample.jpg") as image:
        assert image.size == (200, 200)

    resized = load_yolo_dataset(tmp_path / "resized", layout="auto", workers=1)
    box = resized.images[0].annotations[0].box
    assert box is not None
    assert round(box.cx, 6) == 0.5
    assert round(box.cy, 6) == 0.5
    assert round(box.width, 6) == 0.2
    assert round(box.height, 6) == 0.2


def test_resize_cli_stretches_images_and_manager_api_builds_task(tmp_path, capsys):
    source = _make_dataset(tmp_path / "source")
    output = tmp_path / "resized"
    argv = build_task_argv(
        "convert.resize",
        root=source,
        width=40,
        height=30,
        keep_ratio=False,
    )
    assert "--no-keep-ratio" in argv

    code = cli_main(
        [
            "convert",
            "resize",
            "--root",
            str(source),
            "--out",
            str(output),
            "--width",
            "40",
            "--height",
            "30",
            "--no-keep-ratio",
            "--workers",
            "1",
            "--no-progress",
        ]
    )

    assert code == 0
    with Image.open(output / "images" / "sample.jpg") as image:
        assert image.size == (40, 30)
    payload = capsys.readouterr().out
    assert '"images": 1' in payload

    manager_output = tmp_path / "manager_resized"
    manager = YoloManager(source, layout="auto", init_layout=False, init_check=False)
    assert manager.resize_images(
        out=manager_output,
        width=20,
        height=10,
        keep_ratio=False,
        workers=1,
        progress=False,
    ) == 0
    with Image.open(manager_output / "images" / "sample.jpg") as image:
        assert image.size == (20, 10)
