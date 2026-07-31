from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from yolo_data_manager.converters.tt100k import convert_tt100k


def _write_image(path: Path, size: tuple[int, int] = (100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (20, 30, 40)).save(path)


def test_convert_tt100k_writes_yolo_detection_dataset(tmp_path: Path) -> None:
    source = tmp_path / "tt100k"
    _write_image(source / "train" / "1.jpg")
    _write_image(source / "test" / "2.jpg")
    (source / "annotations_all.json").write_text(
        json.dumps(
            {
                "types": ["stop", "warning"],
                "imgs": {
                    "1": {
                        "path": "train/1.jpg",
                        "id": 1,
                        "objects": [
                            {
                                "category": "stop",
                                "bbox": {"xmin": 10, "ymin": 20, "xmax": 60, "ymax": 70},
                            }
                        ],
                    },
                    "2": {
                        "path": "test/2.jpg",
                        "id": 2,
                        "objects": [
                            {
                                "category": "warning",
                                "bbox": {"xmin": -2, "ymin": 0, "xmax": 100, "ymax": 80},
                            }
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "yolo"
    stats = convert_tt100k(source, output, progress=False)

    assert stats.images == 2
    assert stats.instances == 2
    assert stats.clipped_boxes == 1
    assert (output / "images" / "train" / "1.jpg").exists()
    assert (output / "images" / "test" / "2.jpg").exists()
    assert (output / "labels" / "train" / "1.txt").read_text(encoding="utf-8") == "0 0.350000 0.562500 0.500000 0.625000\n"
    assert (output / "labels" / "test" / "2.txt").read_text(encoding="utf-8") == "1 0.500000 0.500000 1.000000 1.000000\n"
    assert (output / "classes.txt").read_text(encoding="utf-8") == "stop\nwarning\n"
    assert "train: images/train" in (output / "dataset.yaml").read_text(encoding="utf-8")
    assert "val: images/test" in (output / "dataset.yaml").read_text(encoding="utf-8")
