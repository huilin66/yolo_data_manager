from pathlib import Path

from PIL import Image

from yolo_data_manager import (
    compute_multimodal_stats,
    load_multimodal_yolo_dataset,
    render_multimodal_dataset,
)


def test_multimodal_loader_associates_suffixes_and_parses_labels_once(tmp_path, monkeypatch):
    root = _make_multimodal_dataset(tmp_path / "multimodal")
    calls = []

    from yolo_data_manager.io.loader import parse_label_file as real_parse_label_file

    def tracked_parse_label_file(*args, **kwargs):
        calls.append(Path(args[0]))
        return real_parse_label_file(*args, **kwargs)

    monkeypatch.setattr("yolo_data_manager.io.multimodal.parse_label_file", tracked_parse_label_file)
    dataset = load_multimodal_yolo_dataset(
        root,
        image_dirs=["rgb", "infrared"],
        image_params={
            "rgb": {"suffix": "_V"},
            "infrared": {"suffix": "_T"},
        },
        labels_dir="labels",
        label_params={"suffix": "_gt"},
        task="detect",
    )

    assert list(dataset.scenes) == ["a", "b"]
    assert [scene.stem for scene in dataset.complete_scenes] == ["a"]
    assert dataset.annotation_count() == 1
    assert calls == [root / "labels" / "a_gt.txt"]
    assert dataset.scenes["a"].images["rgb"].path.name == "a_V.jpg"
    assert dataset.scenes["a"].images["infrared"].path.name == "a_T.png"
    assert dataset.alignment_report.summary()["warning:missing_modality"] == 1

    stats = compute_multimodal_stats(dataset)
    assert stats["scene_count"] == 1
    assert stats["annotation_stats"]["annotation_count"] == 1
    assert stats["modalities"]["rgb"]["stats"]["image_width"]["max"] == 100
    assert stats["modalities"]["infrared"]["stats"]["image_width"]["max"] == 40
    assert calls == [root / "labels" / "a_gt.txt"]


def test_multimodal_loader_defaults_to_identical_stems_and_visualizes_each_type(tmp_path):
    root = tmp_path / "default_names"
    for name in ("rgb", "depth", "labels"):
        (root / name).mkdir(parents=True)
    Image.new("RGB", (80, 60), color="white").save(root / "rgb" / "scene.jpg")
    Image.new("L", (20, 15), color=128).save(root / "depth" / "scene.png")
    (root / "labels" / "scene.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (root / "class.txt").write_text("object\n", encoding="utf-8")

    dataset = load_multimodal_yolo_dataset(
        root,
        image_dirs=["rgb", "depth"],
        class_file="class.txt",
        task="detect",
    )
    counts = render_multimodal_dataset(dataset, tmp_path / "rendered", show_txt_id=True, workers=1, progress=False)

    assert [scene.stem for scene in dataset.complete_scenes] == ["scene"]
    assert counts == {"rgb": 1, "depth": 1}
    assert (tmp_path / "rendered" / "rgb" / "scene.jpg").exists()
    assert (tmp_path / "rendered" / "depth" / "scene.png").exists()


def test_multimodal_loader_reports_duplicate_normalized_images(tmp_path):
    root = tmp_path / "duplicate"
    for name in ("rgb", "labels"):
        (root / name).mkdir(parents=True)
    Image.new("RGB", (80, 60), color="white").save(root / "rgb" / "scene_V.jpg")
    Image.new("RGB", (80, 60), color="black").save(root / "rgb" / "scene_V.png")
    (root / "labels" / "scene.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (root / "class.txt").write_text("object\n", encoding="utf-8")

    dataset = load_multimodal_yolo_dataset(
        root,
        image_dirs=["rgb"],
        image_params={"rgb": {"suffix": "_V"}},
        task="detect",
    )

    assert dataset.complete_scenes == []
    assert dataset.alignment_report.summary()["error:duplicate_scene_image"] == 1


def _make_multimodal_dataset(root: Path) -> Path:
    for name in ("rgb", "infrared", "labels"):
        (root / name).mkdir(parents=True)
    Image.new("RGB", (100, 80), color="white").save(root / "rgb" / "a_V.jpg")
    Image.new("RGB", (100, 80), color="white").save(root / "rgb" / "b_V.jpg")
    Image.new("L", (40, 30), color=120).save(root / "infrared" / "a_T.png")
    (root / "labels" / "a_gt.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    (root / "labels" / "b_gt.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    (root / "class.txt").write_text("object\n", encoding="utf-8")
    return root
