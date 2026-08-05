from pathlib import Path

from PIL import Image

from yolo_data_manager.io.loader import load_yolo_dataset
from yolo_data_manager.scripting import YoloManager, build_task_argv
from yolo_data_manager.vis.manual_box import draw_manual_box, find_dataset_image
import yolo_data_manager.vis.manual_box as manual_box_module


def test_manual_box_returns_coordinates_without_writing_label(tmp_path, monkeypatch):
    root = tmp_path / "yolo"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    image_path = root / "images" / "sample.jpg"
    label_path = root / "labels" / "sample.txt"
    Image.new("RGB", (100, 80), color="white").save(image_path)
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    original_label = label_path.read_text(encoding="utf-8")

    monkeypatch.setattr(
        manual_box_module,
        "_run_box_window",
        lambda *args, **kwargs: (10, 20, 50, 60),
    )
    result = draw_manual_box(image_path, class_id=3, precision=6)

    assert result is not None
    assert result.pixel_xyxy == (10, 20, 50, 60)
    assert result.yolo_xywhn == (0.3, 0.5, 0.4, 0.5)
    assert result.yolo_line == "3 0.300000 0.500000 0.400000 0.500000"
    assert label_path.read_text(encoding="utf-8") == original_label


def test_manual_box_finds_relative_dataset_image_and_builds_cli_argv(tmp_path):
    root = tmp_path / "yolo"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    Image.new("RGB", (20, 20), color="white").save(root / "images" / "sample.jpg")
    dataset = load_yolo_dataset(root, progress=False)

    image = find_dataset_image(dataset, "images/sample.jpg")
    assert image.path.name == "sample.jpg"

    argv = build_task_argv(
        "vis.manual_box",
        root=Path("dataset"),
        image="sample.jpg",
        class_id=3,
        out=Path("manual_box.json"),
    )
    assert argv[:2] == ["vis", "manual-box"]
    assert "--image" in argv
    assert "--class-id" in argv
    assert "--out" in argv


def test_yolo_manager_manual_box_delegates_without_write_options(tmp_path, monkeypatch):
    captured = {}

    def fake_run_task(command, **params):
        captured["command"] = command
        captured.update(params)
        return 0

    import yolo_data_manager.scripting as scripting

    monkeypatch.setattr(scripting, "run_task", fake_run_task)
    manager = YoloManager(
        tmp_path / "dataset",
        init_layout=False,
        init_check=False,
    )

    assert manager.vis_manual_box("images/sample.jpg", class_id=4) == 0
    assert captured["command"] == "vis.manual_box"
    assert captured["image"] == "images/sample.jpg"
    assert captured["class_id"] == 4
    assert "out" in captured
