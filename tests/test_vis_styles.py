from PIL import Image
import pytest

from yolo_data_manager.io.loader import load_yolo_dataset
from yolo_data_manager.scripting import build_task_argv
from yolo_data_manager.vis.renderer import crop_dataset, render_dataset


@pytest.mark.parametrize("style", ["pil", "cv2"])
def test_visualization_styles_render_and_crop_in_parallel(tmp_path, style):
    root = tmp_path / "数据集"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    Image.new("RGB", (100, 80), color="white").save(root / "images" / "sample.jpg")
    (root / "class.txt").write_text("object\n", encoding="utf-8")
    (root / "labels" / "sample.txt").write_text(
        "0 0.5 0.5 0.4 0.5\n",
        encoding="utf-8",
    )
    dataset = load_yolo_dataset(root, progress=False)

    rendered_dir = tmp_path / f"rendered-{style}"
    crop_dir = tmp_path / f"crops-{style}"
    render_dataset(dataset, rendered_dir, style=style, workers=2, progress=False)
    saved = crop_dataset(dataset, crop_dir, style=style, workers=2, progress=False)

    assert (rendered_dir / "sample.jpg").exists()
    assert (crop_dir / "object" / "sample_1.jpg").exists()
    assert saved == 1
    with Image.open(rendered_dir / "sample.jpg") as rendered:
        assert rendered.size == (100, 80)


def test_cv_alias_is_supported(tmp_path):
    root = tmp_path / "alias"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    Image.new("RGB", (20, 20), color="white").save(root / "images" / "a.jpg")
    (root / "class.txt").write_text("object\n", encoding="utf-8")
    (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    dataset = load_yolo_dataset(root, progress=False)

    render_dataset(dataset, tmp_path / "cv", style="cv", workers=1, progress=False)

    assert (tmp_path / "cv" / "a.jpg").exists()


@pytest.mark.parametrize("style", ["pil", "cv2"])
def test_attribute_separate_copies_drawn_images_and_filters_no(tmp_path, style):
    root = tmp_path / "attributes"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    Image.new("RGB", (100, 80), color="white").save(root / "images" / "yes.jpg")
    Image.new("RGB", (100, 80), color="white").save(root / "images" / "no.jpg")
    (root / "class.txt").write_text("object\n", encoding="utf-8")
    (root / "attribute.yaml").write_text(
        "attributes:\n  defect: [no, yes]\n",
        encoding="utf-8",
    )
    (root / "labels" / "yes.txt").write_text(
        "0 1 1 0.5 0.5 0.4 0.4\n",
        encoding="utf-8",
    )
    (root / "labels" / "no.txt").write_text(
        "0 1 0 0.5 0.5 0.4 0.4\n",
        encoding="utf-8",
    )
    dataset = load_yolo_dataset(root, progress=False)
    draw_dir = tmp_path / "ydm_vis" / "draw"

    render_dataset(
        dataset,
        draw_dir,
        style=style,
        show_attributes=True,
        filter_no_attributes=True,
        att_seperate=True,
        workers=2,
        progress=False,
    )

    separated_dir = tmp_path / "ydm_vis" / "att_seperate"
    assert (draw_dir / "yes.jpg").exists()
    assert (draw_dir / "no.jpg").exists()
    assert (separated_dir / "defect" / "yes" / "yes.jpg").exists()
    assert not (separated_dir / "defect" / "no").exists()
    assert (
        (separated_dir / "defect" / "yes" / "yes.jpg").read_bytes()
        == (draw_dir / "yes.jpg").read_bytes()
    )


def test_attribute_separate_argument_is_forwarded_to_cli():
    argv = build_task_argv("vis.draw", root="dataset", att_seperate=True)

    assert "--att-seperate" in argv
