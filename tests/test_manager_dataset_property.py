from pathlib import Path

from PIL import Image

from yolo_data_manager import YoloManager


def _make_dataset(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    Image.new("RGB", (20, 20), "white").save(root / "images" / "a.jpg")
    (root / "class.txt").write_text("old\nkeep\n", encoding="utf-8")
    (root / "labels" / "a.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n",
        encoding="utf-8",
    )
    return root


def test_manager_dataset_and_schema_properties_are_lazy(tmp_path):
    root = _make_dataset(tmp_path / "lazy")
    manager = YoloManager(
        root,
        layout="flat",
        task="detect",
        init_layout=False,
        init_check=False,
    )

    assert manager._dataset is None

    dataset = manager.load(progress=False)

    assert manager.dataset is dataset
    assert manager.classes is dataset.classes
    assert manager.classes.names == ["old", "keep"]
    assert manager.attributes is None
    assert len(dataset.images) == 1
    assert dataset.annotation_count() == 1


def test_manager_mutations_invalidate_lazy_dataset_cache(tmp_path):
    root = _make_dataset(tmp_path / "invalidate")
    manager = YoloManager(
        root,
        layout="flat",
        task="detect",
        init_layout=False,
        init_check=False,
    )

    before = manager.load(progress=False)
    assert before.classes.names == ["old", "keep"]

    manager.ann_update_from_map(
        {"rename": {"old": "new"}},
        progress=False,
    )

    assert manager._dataset is None
    after = manager.load(progress=False)
    assert after is not before
    assert manager.classes.names == ["new", "keep"]

