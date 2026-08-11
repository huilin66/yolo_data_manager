from example.functions._manager import get_yolo_manager
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
