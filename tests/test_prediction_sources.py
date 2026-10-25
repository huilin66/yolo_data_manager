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
