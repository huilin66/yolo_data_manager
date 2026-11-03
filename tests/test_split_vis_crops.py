from importlib import import_module
from pathlib import Path


def _load_tool():
    return import_module("yolo_data_manager.tools.split_vis_crops")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"crop")


def test_source_split_reuses_source_txt_and_skips_unmatched(tmp_path):
    tool = _load_tool()
    crops = tmp_path / "crop"
    _touch(crops / "class_a" / "frame_001_1.jpg")
    _touch(crops / "class_a" / "frame_001_2.jpg")
    _touch(crops / "class_b" / "frame_002_1.jpg")
    _touch(crops / "class_b" / "not_a_crop.jpg")
    split_dir = tmp_path / "dataset"
    split_dir.mkdir()
    (split_dir / "train.txt").write_text("images/frame_001.jpg\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("images/frame_002.jpg\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("", encoding="utf-8")

    report = tool.split_vis_crops(
        crops,
        tmp_path / "crop_split",
        mode="source_split",
        split_dir=split_dir,
    )

    assert (tmp_path / "crop_split" / "train.txt").read_text(encoding="utf-8").splitlines() == [
        "class_a/frame_001_1.jpg",
        "class_a/frame_001_2.jpg",
    ]
    assert (tmp_path / "crop_split" / "val.txt").read_text(encoding="utf-8").splitlines() == [
        "class_b/frame_002_1.jpg",
    ]
    assert (tmp_path / "crop_split" / "test.txt").read_text(encoding="utf-8") == ""
    assert report["unmatched_crops"] == 1
    assert report["splits"]["train"]["source_stems"] == 1


def test_stem_random_keeps_one_source_stem_in_one_split(tmp_path):
    tool = _load_tool()
    crops = tmp_path / "crop"
    for name in ("frame_001_1.jpg", "frame_001_2.jpg", "frame_002_1.jpg", "frame_003_1.jpg"):
        _touch(crops / "class_a" / name)

    out = tmp_path / "crop_split"
    tool.split_vis_crops(
        crops,
        out,
        mode="stem_random",
        train=0.5,
        val=0.5,
        test=0.0,
        seed=233,
    )

    split_by_stem = {}
    for split in ("train", "val", "test"):
        for value in (out / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            stem = Path(value).stem.rsplit("_", 1)[0]
            split_by_stem.setdefault(stem, set()).add(split)
    assert split_by_stem
    assert all(len(splits) == 1 for splits in split_by_stem.values())


def test_crop_random_is_deterministic_and_assigns_each_file_independently(tmp_path):
    tool = _load_tool()
    crops = tmp_path / "crop"
    for index in range(6):
        _touch(crops / "class_a" / f"frame_{index:03d}_1.jpg")

    first = tmp_path / "first"
    second = tmp_path / "second"
    tool.split_vis_crops(crops, first, mode="crop_random", seed=7)
    tool.split_vis_crops(crops, second, mode="crop_random", seed=7)

    for split in ("train", "val", "test"):
        first_text = (first / f"{split}.txt").read_text(encoding="utf-8")
        second_text = (second / f"{split}.txt").read_text(encoding="utf-8")
        assert first_text == second_text
    all_entries = {
        value
        for split in ("train", "val", "test")
        for value in (first / f"{split}.txt").read_text(encoding="utf-8").splitlines()
    }
    assert len(all_entries) == 6
