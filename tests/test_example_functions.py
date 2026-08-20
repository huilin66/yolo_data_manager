from example.functions._manager import get_yolo_manager
from example.functions.data_resize import yolo_resize
from example.functions.data_split import yolo_split
from yolo_data_manager import YoloManager


def test_get_yolo_manager_reuses_existing_manager():
    manager = object.__new__(YoloManager)

    assert get_yolo_manager(manager, layout="flat") is manager


def test_example_function_does_not_initialize_existing_manager(monkeypatch):
    manager = object.__new__(YoloManager)
    manager.dataset_split = lambda **_kwargs: 7

    def fail_if_initialized(*_args, **_kwargs):
        raise AssertionError("an existing YoloManager must be reused")

    monkeypatch.setattr(YoloManager, "__init__", fail_if_initialized)

    assert yolo_split(manager) == 7


def test_example_split_passes_include_lists_to_manager():
    manager = object.__new__(YoloManager)
    captured = {}

    def fake_split(**kwargs):
        captured.update(kwargs)
        return 7

    manager.dataset_split = fake_split

    assert yolo_split(
        manager,
        train_include_list=["a.jpg"],
        val_include_list="val_include.txt",
    ) == 7
    assert captured["train_include_list"] == ["a.jpg"]
    assert captured["val_include_list"] == "val_include.txt"


def test_example_resize_reuses_existing_manager():
    manager = object.__new__(YoloManager)
    manager.resize_images = lambda **_kwargs: 9

    assert yolo_resize(manager, width=640) == 9


def test_yolo_manager_exposes_default_output_paths(tmp_path):
    manager = object.__new__(YoloManager)
    manager.root = str(tmp_path)

    assert manager.output_quality == tmp_path / "ydm_quality"
    assert manager.output_stats == tmp_path / "ydm_stats"
    assert manager.output_vis == tmp_path / "ydm_vis"
    assert manager.output_evaluation == tmp_path / "ydm_evaluation"
    assert manager.output_dataset == tmp_path / "ydm_dataset"
    assert manager.output_annotation == tmp_path / "ydm_annotation"
    assert manager.output_conversion == tmp_path / "ydm_conversion"
    assert manager.output_labels_backup == tmp_path / "labels_backup"
    assert manager.output_train == tmp_path / "train.txt"
    assert manager.output_val == tmp_path / "val.txt"
    assert manager.output_test == tmp_path / "test.txt"
    assert manager.output_dataset_yaml == tmp_path / "dataset.yaml"
