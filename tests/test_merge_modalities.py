import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from yolo_data_manager import YoloManager, MultiModalYoloManager, merge_modalities


def make_sources(tmp_path):
    sources = []
    for idx, stems in enumerate((('a', 'b'), ('a', 'c'))):
        root = tmp_path / f"source{idx}"
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        (root / "class.txt").write_text("thing\nother\n" if idx == 0 else "other\nthing\n", encoding="utf-8")
        for stem in stems:
            array = np.full((12, 16, 3), 100, dtype=np.uint8) if idx == 0 else np.full((6, 8), 3000, dtype=np.uint16)
            Image.fromarray(array).save(root / "images" / f"{stem}.png")
            (root / "labels" / f"{stem}.txt").write_text(f"{idx} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (root / "train.txt").write_text("images/a.png\n", encoding="utf-8")
        (root / ("val.txt" if idx == 0 else "test.txt")).write_text(f"images/{stems[1]}.png\n", encoding="utf-8")
        sources.append({"root": root})
    return sources


@pytest.mark.parametrize("unmatched,expected", [("keep", 3), ("drop", 1), ("zero_fill", 3)])
def test_merge_modalities_policies_and_splits(tmp_path, unmatched, expected):
    sources = make_sources(tmp_path)
    output = tmp_path / "merged"
    manager = YoloManager(sources[0]["root"], init_check=False, init_layout=False)
    report = manager.dataset_merge_modalities(sources, output, unmatched=unmatched, inherit_splits=True, workers=2, progress=False)
    assert report["scenes"] == expected
    assert report["image_dirs"] == ["image1", "image2"]
    assert (output / "labels/a.txt").read_text().splitlines() == ["0 0.5 0.5 0.2 0.2"] * 2
    assert (output / "train.txt").read_text().strip() == str(output / "image1/a.png")
    config = yaml.safe_load((output / "dataset.yaml").read_text())
    assert config["image_dirs"] == ["image1", "image2"]
    assert config["train"] == "train.txt"
    assert json.loads((output / "merge_report.json").read_text())["scenes"] == expected
    multimodal = MultiModalYoloManager(output, image_dirs=report["image_dirs"], progress=False).load()
    assert len(multimodal.scenes) == expected
    assert len(multimodal.complete_scenes) == (3 if unmatched == "zero_fill" else 1)
    if unmatched == "zero_fill":
        with Image.open(output / "image2/b.png") as image:
            assert image.size == (8, 6)
            assert np.asarray(image).dtype == np.uint16
            assert not np.asarray(image).any()
        with Image.open(output / "image1/c.png") as image:
            assert image.size == (16, 12)
            assert np.asarray(image).shape == (12, 16, 3)
            assert not np.asarray(image).any()
        assert report["zero_filled"] == 2
    elif unmatched == "keep":
        assert not (output / "image2/b.png").exists()
        assert not (output / "image1/c.png").exists()
    else:
        assert not (output / "labels/b.txt").exists()


def test_merge_modalities_first_labels_and_deduplication(tmp_path):
    sources = make_sources(tmp_path)
    sources[0]["name"] = "rgb"
    sources[1]["name"] = "thermal"
    output = tmp_path / "first_labels"
    result = merge_modalities(sources, output, merge_labels=False, progress=False)
    assert result["image_dirs"] == ["rgb", "thermal"]
    assert (output / "labels/c.txt").read_text() == ""
    assert result["primary_labels_missing"] == ["c"]
    assert len((output / "labels/a.txt").read_text().splitlines()) == 1
    result = merge_modalities(sources, tmp_path / "dedup", deduplicate=True, progress=False)
    assert result["deduplicated"] == 1
    assert len((tmp_path / "dedup/labels/a.txt").read_text().splitlines()) == 1


def test_merge_modalities_conflicts_and_dry_run(tmp_path):
    sources = make_sources(tmp_path)
    output = tmp_path / "merged"
    result = merge_modalities(sources, output, dry_run=True, progress=False)
    assert result["dry_run"]
    assert {path.name for path in output.iterdir()} == {"ydm_log"}
    (sources[1]["root"] / "train.txt").write_text("", encoding="utf-8")
    (sources[1]["root"] / "val.txt").write_text("images/a.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting train/val/test"):
        merge_modalities(sources, output, inherit_splits=True, progress=False)
    assert {path.name for path in output.iterdir()} == {"ydm_log"}


def test_merge_modalities_heterogeneous_zero_fill_requires_spec(tmp_path):
    sources = make_sources(tmp_path)
    Image.new("RGB", (20, 30)).save(sources[0]["root"] / "images/b.png")
    with pytest.raises(ValueError, match="uniform image specification"):
        merge_modalities(sources, tmp_path / "invalid", unmatched="zero_fill", progress=False)
    sources[0]["zero_fill"] = {"width": 16, "height": 12, "channels": 3, "dtype": "uint8"}
    merge_modalities(sources, tmp_path / "explicit", unmatched="zero_fill", progress=False)
    assert (tmp_path / "explicit/image1/c.png").is_file()


def test_merge_modalities_duplicate_stems_and_output_protection(tmp_path):
    sources = make_sources(tmp_path)
    Image.new("RGB", (16, 12)).save(sources[0]["root"] / "images/a.jpg")
    with pytest.raises(ValueError, match="duplicate image stem"):
        merge_modalities(sources, tmp_path / "duplicate", progress=False)
    with pytest.raises(ValueError, match="outside"):
        merge_modalities(sources, sources[0]["root"], progress=False)
    with pytest.raises(ValueError, match="folder name"):
        merge_modalities([{**sources[0], "name": "../bad"}, sources[1]], tmp_path / "bad", progress=False)


def test_merge_modalities_attribute_conflicts(tmp_path):
    sources = make_sources(tmp_path)
    for index, source in enumerate(sources):
        (source["root"] / "attribute.yaml").write_text(f"attributes:\n  risk: [low, {'high' if index == 0 else 'medium'}]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting attribute"):
        merge_modalities(sources, tmp_path / "conflict", progress=False)


def test_merge_modalities_reorders_attribute_values_and_has_no_implicit_split(tmp_path):
    sources = make_sources(tmp_path)
    for idx, source in enumerate(sources):
        root = source["root"]
        for path in (root / "train.txt", root / "val.txt", root / "test.txt"):
            path.unlink(missing_ok=True)
        fields = "  risk: [low, high]\n  wet: [dry, wet]\n" if idx == 0 else "  wet: [dry, wet]\n  risk: [low, high]\n"
        (root / "attribute.yaml").write_text("attributes:\n" + fields, encoding="utf-8")
        values = "1 0" if idx == 0 else "0 1"
        (root / "labels/a.txt").write_text(f"{idx} 2 {values} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    output = tmp_path / "attribute_merged"
    result = merge_modalities(sources, output, deduplicate=True, unmatched="drop", progress=False)
    assert (output / "labels/a.txt").read_text().strip() == "0 2 1 0 0.5 0.5 0.2 0.2"
    assert result["deduplicated"] == 1
    assert not (output / "train.txt").exists()
    assert "train" not in yaml.safe_load((output / "dataset.yaml").read_text())


def test_merge_modalities_empty_modality_can_use_explicit_zero_spec(tmp_path):
    sources = make_sources(tmp_path)
    for folder in ("images", "labels"):
        for path in (sources[1]["root"] / folder).iterdir():
            path.unlink()
    sources[1]["zero_fill"] = {"width": 8, "height": 6, "channels": 1, "dtype": "float32"}
    output = tmp_path / "empty_filled"
    result = merge_modalities(sources, output, unmatched="zero_fill", progress=False)
    assert result["zero_filled"] == 2
    with Image.open(output / "image2/a.tif") as image:
        assert np.asarray(image).dtype == np.float32
        assert not np.asarray(image).any()


def test_public_merge_logs_only_to_output_and_accepts_dry_run_logs(tmp_path):
    sources = make_sources(tmp_path)
    output = tmp_path / "public_merge"
    merge_modalities(sources, output, dry_run=True, progress=False)
    merge_modalities(sources, output, progress=False)
    logs = list((output / "ydm_log").glob("*.log"))
    assert len(logs) == 1
    text = logs[0].read_text(encoding="utf-8")
    assert text.count("START operation=dataset.merge_modalities") == 2
    assert text.count("END operation=dataset.merge_modalities status=ok") == 2
    assert all(not (source["root"] / "ydm_log").exists() for source in sources)
    with pytest.raises(FileExistsError):
        merge_modalities(sources, output, progress=False)


def test_manager_merge_delegate_does_not_use_its_root(tmp_path):
    sources = make_sources(tmp_path)
    unrelated = tmp_path / "unrelated"
    manager = YoloManager(unrelated, init_check=False, init_layout=False)
    output = tmp_path / "delegated"
    manager.dataset_merge_modalities(sources, output, progress=False)
    assert not unrelated.exists()
    assert (output / "ydm_log").is_dir()


def test_default_merge_ignores_source_split_conflicts(tmp_path):
    sources = make_sources(tmp_path)
    (sources[1]["root"] / "val.txt").write_text("images/a.png\n", encoding="utf-8")
    output = tmp_path / "no_split"
    report = merge_modalities(sources, output, progress=False)
    assert report["inherit_splits"] is False
    assert report["unassigned_scenes"] == 3
    for split in ("train", "val", "test"):
        assert not (output / f"{split}.txt").exists()
    for name in ("dataset.yaml", "multimodal.yaml"):
        config = yaml.safe_load((output / name).read_text())
        assert not any(split in config for split in ("train", "val", "test"))


@pytest.mark.parametrize("policy,images,scenes,annotations", [("keep", 4, 3, 4), ("drop", 2, 1, 2), ("zero_fill", 6, 3, 4)])
def test_merge_summary_counts_and_console_report(tmp_path, capsys, policy, images, scenes, annotations):
    sources = make_sources(tmp_path)
    output = tmp_path / "summary"
    result = merge_modalities(sources, output, unmatched=policy, progress=False)
    summary = result["summary"]
    assert [item["images"] for item in summary["inputs"]] == [2, 2]
    assert [item["annotations"] for item in summary["inputs"]] == [2, 2]
    assert [item["classes"] for item in summary["inputs"]] == [2, 2]
    assert summary["output"]["images"] == images
    assert summary["output"]["scenes"] == scenes
    assert summary["output"]["annotations"] == annotations
    assert summary["output"]["classes"] == 2
    assert summary["output"]["used_classes"] == 1
    assert sum(item["output_images"] for item in summary["modalities"]) == images
    text = capsys.readouterr().out
    assert text.count("Merge modalities summary") == 1
    assert "input:image1" in text and "input:image2" in text
    assert "Image changes by modality" in text
    assert "shared across modalities and counted once" in text
    saved = json.loads((output / "merge_report.json").read_text())
    assert saved["summary"] == summary
    assert "Merge modalities summary" in next((output / "ydm_log").glob("*.log")).read_text(encoding="utf-8")
