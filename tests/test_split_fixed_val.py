from pathlib import Path

from PIL import Image

from yolo_data_manager.dataset.split import split_dataset
from yolo_data_manager.io.loader import load_yolo_dataset


def _make_dataset(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "class.txt").write_text("object\n", encoding="utf-8")
    for index in range(8):
        name = f"image_{index:03d}.jpg"
        Image.new("RGB", (20, 20), "white").save(root / "images" / name)
        (root / "labels" / f"image_{index:03d}.txt").write_text(
            "0 0.5 0.5 0.4 0.4\n",
            encoding="utf-8",
        )
    return root


def test_val_source_keeps_previous_val_exactly(tmp_path):
    root = _make_dataset(tmp_path / "v2")
    previous_val = tmp_path / "v1_val.txt"
    previous_val.write_text(
        "images/image_001.jpg\nimages/image_003.jpg\n",
        encoding="utf-8",
    )
    dataset = load_yolo_dataset(root, layout="flat", progress=False)

    splits = split_dataset(
        dataset,
        train=0.8,
        val=0.2,
        test=0.0,
        seed=233,
        val_source=previous_val,
    )

    assert splits["val"] == ["image_001.jpg", "image_003.jpg"]
    assert splits["test"] == []
    assert len(splits["train"]) == 6
    assert set(splits["train"]) | set(splits["val"]) == {
        f"image_{index:03d}.jpg" for index in range(8)
    }


def test_val_source_splits_only_remaining_images_between_train_and_test(tmp_path):
    root = _make_dataset(tmp_path / "v2")
    previous_val = tmp_path / "v1_val.txt"
    previous_val.write_text("image_001.jpg\nimage_003.jpg\n", encoding="utf-8")
    dataset = load_yolo_dataset(root, layout="flat", progress=False)

    splits = split_dataset(
        dataset,
        train=0.7,
        val=0.1,
        test=0.2,
        seed=233,
        val_source=previous_val,
    )

    assert set(splits["val"]) == {"image_001.jpg", "image_003.jpg"}
    assert len(splits["train"]) == 5
    assert len(splits["test"]) == 1
    assert not set(splits["train"]) & set(splits["val"])
    assert not set(splits["test"]) & set(splits["val"])

