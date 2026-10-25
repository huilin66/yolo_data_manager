import json

from PIL import Image
import pytest

from yolo_data_manager import YoloManager


def setup_sources(tmp_path):
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    Image.new("RGB", (20, 20)).save(root / "images/a.jpg")
    (root / "class.txt").write_text("old\nnew\n", encoding="utf-8")
    (root / "labels/a.txt").write_text("0 0.1 0.1 0.1 0.1\n0 0.2 0.2 0.1 0.1\n", encoding="utf-8")
    (root / "val.txt").write_text("images/a.jpg\n", encoding="utf-8")
    sources = []
    for index in range(2):
        pred = tmp_path / f"pred{index}"
        crop = tmp_path / f"crop{index}"
        pred.mkdir()
        crop.mkdir()
        # pred1 differs across sources; pred2 is a duplicate across sources.
        x = 0.5 if index == 0 else 0.8
        (pred / "a.txt").write_text(f"1 {x} 0.5 0.1 0.1 0.9\n1 0.6 0.7 0.1 0.1 0.8\n", encoding="utf-8")
        (crop / "a_pred1_gtnone.jpg").touch()
        (crop / "a_pred2_gtnone.jpg").touch()
        (crop / f"a_prednone_gt{index + 1}.jpg").touch()
        sources.append({"pred_dir": pred, "crops_dir": {crop: "new"}})
    return root, sources


@pytest.mark.parametrize("export", [False, True])
def test_prediction_sources_one_session_and_cross_source_dedup(tmp_path, export):
    root, sources = setup_sources(tmp_path)
    original = (root / "labels/a.txt").read_text()
    output = tmp_path / "output" if export else root
    manager = YoloManager(root, layout="flat", init_check=False, init_layout=False)
    assert manager.ann_correct_from_error_crops(
        pred_sources=sources, out_data=output if export else None, progress=False,
    ) == 0
    lines = (output / "labels/a.txt").read_text().splitlines()
    assert len(lines) == 5
    assert all(line.split()[0] == "1" for line in lines)
    assert {float(line.split()[1]) for line in lines[2:]} == {0.5, 0.8, 0.6}
    assert all(len(line.split()) == 5 for line in lines)
    backups = list((output / "labels_backup").glob("*/backup_metadata.json"))
    assert len(backups) == 1
    assert (backups[0].parent / "labels/a.txt").read_text() == original
    metadata = json.loads(backups[0].read_text(encoding="utf-8"))
    assert len(metadata["result"]["pred_sources"]) == 2
    assert "ydm_pred_sources_" not in json.dumps(metadata)
    if export:
        assert (root / "labels/a.txt").read_text() == original
        assert not (root / "labels_backup").exists()
        assert (output / "val.txt").read_text().strip() == str((output / "images/a.jpg").resolve())


def test_prediction_sources_conflict_before_writing(tmp_path):
    root, sources = setup_sources(tmp_path)
    crop = next(iter(sources[1]["crops_dir"]))
    (crop / "a_prednone_gt1.jpg").touch()
    sources[1]["crops_dir"][crop] = "old"
    original = (root / "labels/a.txt").read_text()
    manager = YoloManager(root, layout="flat", init_check=False, init_layout=False)
    with pytest.raises(ValueError, match="conflicting corrections"):
        manager.ann_correct_from_error_crops(pred_sources=sources, progress=False)
    assert (root / "labels/a.txt").read_text() == original
    assert not (root / "labels_backup").exists()


def test_prediction_sources_dry_run_and_exclusive_arguments(tmp_path):
    root, sources = setup_sources(tmp_path)
    original = (root / "labels/a.txt").read_text()
    manager = YoloManager(root, layout="flat", init_check=False, init_layout=False)
    assert manager.ann_correct_from_error_crops(pred_sources=sources, dry_run=True, progress=False) == 0
    assert (root / "labels/a.txt").read_text() == original
    assert not (root / "labels_backup").exists()
    with pytest.raises(ValueError, match="cannot be combined"):
        manager.ann_correct_from_error_crops("crops", pred_sources=sources)
    with pytest.raises(ValueError, match="non-empty"):
        manager.ann_correct_from_error_crops(pred_sources=[], progress=False)


def test_prediction_sources_replacement_uses_bound_prediction_and_rejects_conflict(tmp_path):
    root, sources = setup_sources(tmp_path)
    for source in sources:
        crop = next(iter(source["crops_dir"]))
        for path in crop.iterdir():
            path.unlink()
    first_crop = next(iter(sources[0]["crops_dir"]))
    second_crop = next(iter(sources[1]["crops_dir"]))
    (first_crop / "a_pred1_gt1.jpg").touch()
    (second_crop / "a_pred1_gt1.jpg").touch()
    manager = YoloManager(root, layout="flat", init_check=False, init_layout=False)
    with pytest.raises(ValueError, match="conflicting corrections"):
        manager.ann_correct_from_error_crops(pred_sources=sources, replace_gt_from_pred=True, progress=False)
    assert not (root / "labels_backup").exists()
    (second_crop / "a_pred1_gt1.jpg").unlink()
    (second_crop / "a_pred1_gt2.jpg").touch()
    manager.ann_correct_from_error_crops(pred_sources=sources, replace_gt_from_pred=True, progress=False)
    lines = (root / "labels/a.txt").read_text().splitlines()
    assert [float(line.split()[1]) for line in lines] == [0.5, 0.8]


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("global_replace", [False, True])
def test_per_directory_replace_overrides_and_global_fallback(tmp_path, batch, global_replace):
    root, sources = setup_sources(tmp_path)
    crop_map = {}
    for index, rule in enumerate((["old", True], ["new", False], "old")):
        directory = tmp_path / f"rule{index}"
        directory.mkdir()
        (directory / f"a_pred{index + 1}_gt{index + 1}.jpg").touch()
        crop_map[directory] = rule
    (root / "labels/a.txt").write_text(
        "0 0.1 0.1 0.1 0.1\n0 0.2 0.2 0.1 0.1\n0 0.3 0.3 0.1 0.1\n",
        encoding="utf-8",
    )
    pred_dir = sources[0]["pred_dir"]
    (pred_dir / "a.txt").write_text(
        "1 0.5 0.5 0.1 0.1 0.9\n1 0.7 0.5 0.1 0.1 0.9\n1 0.9 0.5 0.1 0.1 0.9\n",
        encoding="utf-8",
    )
    manager = YoloManager(root, layout="flat", init_check=False, init_layout=False)
    kwargs = {"pred_sources": [{"pred_dir": pred_dir, "crops_dir": crop_map}]} if batch else {"crops_dir": crop_map, "pred_dir": pred_dir}
    manager.ann_correct_from_error_crops(**kwargs, replace_gt_from_pred=global_replace, progress=False)
    lines = [line.split() for line in (root / "labels/a.txt").read_text().splitlines()]
    assert [row[0] for row in lines] == ["1", "1", "1" if global_replace else "0"]
    assert [float(row[1]) for row in lines] == [0.5, 0.2, 0.9 if global_replace else 0.3]
    assert len(list((root / "labels_backup").glob("*/backup_metadata.json"))) == 1


@pytest.mark.parametrize("rule", [["new"], ["new", "false"], ["new", 0], ["new", True, False]])
def test_invalid_per_directory_replace_rule_rejected(tmp_path, rule):
    root, sources = setup_sources(tmp_path)
    directory = next(iter(sources[0]["crops_dir"]))
    manager = YoloManager(root, layout="flat", init_check=False, init_layout=False)
    with pytest.raises(ValueError, match=r"\[class, bool\]"):
        manager.ann_correct_from_error_crops({directory: rule}, progress=False)
    assert not (root / "labels_backup").exists()


def test_false_override_keeps_missing_prediction_gt_and_explicit_none_deletes(tmp_path):
    root, sources = setup_sources(tmp_path)
    keep = tmp_path / "keep_gt"
    delete = tmp_path / "delete_gt"
    keep.mkdir()
    delete.mkdir()
    (keep / "a_prednone_gt1.jpg").touch()
    (delete / "a_prednone_gt2.jpg").touch()
    output = tmp_path / "updated"
    manager = YoloManager(root, layout="flat", init_check=False, init_layout=False)
    manager.ann_correct_from_error_crops(
        pred_sources=[{"pred_dir": sources[0]["pred_dir"], "crops_dir": {keep: ["new", False], delete: [None, False]}}],
        replace_gt_from_pred=True, out_data=output, progress=False,
    )
    assert (output / "labels/a.txt").read_text().splitlines() == ["1 0.1 0.1 0.1 0.1"]
    assert len((root / "labels/a.txt").read_text().splitlines()) == 2


def test_mixed_modes_keep_original_gt_indices_after_replacement_dedup(tmp_path):
    root, sources = setup_sources(tmp_path)
    (root / "labels/a.txt").write_text(
        "0 0.1 0.1 0.1 0.1\n0 0.2 0.2 0.1 0.1\n0 0.3 0.3 0.1 0.1\n", encoding="utf-8",
    )
    pred_dir = sources[0]["pred_dir"]
    (pred_dir / "a.txt").write_text(
        "1 0.8 0.8 0.1 0.1 0.9\n1 0.8 0.8 0.1 0.1 0.8\n", encoding="utf-8",
    )
    replace = tmp_path / "replace"
    class_only = tmp_path / "class_only"
    replace.mkdir()
    class_only.mkdir()
    (replace / "a_pred1_gt1.jpg").touch()
    (replace / "a_pred2_gt2.jpg").touch()
    (class_only / "a_prednone_gt3.jpg").touch()
    manager = YoloManager(root, layout="flat", init_check=False, init_layout=False)
    manager.ann_correct_from_error_crops(
        {replace: ["new", True], class_only: ["new", False]}, pred_dir=pred_dir, progress=False,
    )
    assert (root / "labels/a.txt").read_text().splitlines() == [
        "1 0.8 0.8 0.1 0.1", "1 0.3 0.3 0.1 0.1",
    ]
