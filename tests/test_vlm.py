from pathlib import Path

from PIL import Image
import pytest

from yolo_data_manager.core.models import (
    AttributeSchema,
    Box,
    ClassSchema,
    YoloAnnotation,
    YoloDataset,
    YoloImage,
)
from yolo_data_manager.vlm.assistant import AssistantError, run_assistant
from yolo_data_manager.vlm.auto_label import generate_yolo_labels
from yolo_data_manager.vlm.config import load_vlm_config
from yolo_data_manager.vlm.error_verify import apply_correction_plan, verify_error_crops
from yolo_data_manager.vlm.providers import VLMProvider, VLMResponse
from yolo_data_manager.vlm.schemas import parse_detection_response, parse_error_decision


class FakeVLM(VLMProvider):
    name = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, *, images=(), json_mode=True):
        self.prompts.append((prompt, tuple(images), json_mode))
        return VLMResponse(self.responses.pop(0))


def _dataset(tmp_path: Path) -> YoloDataset:
    image_path = tmp_path / "a.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), "white").save(image_path)
    label_path = tmp_path / "a.txt"
    label_path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    return YoloDataset(
        root=tmp_path,
        images=[
            YoloImage(
                image_path,
                label_path=label_path,
                width=20,
                height=20,
                annotations=[YoloAnnotation(0, Box(0.5, 0.5, 0.5, 0.5))],
            )
        ],
        classes=ClassSchema(["old", "new"]),
    )


def test_vlm_response_parsers_accept_fenced_json():
    fence = chr(96) * 3
    boxes = parse_detection_response(
        fence + 'json\n{"boxes":[{"class_name":"old","bbox":[0.5,0.5,0.2,0.4]}]}\n' + fence
    )
    assert boxes[0].bbox == (0.5, 0.5, 0.2, 0.4)
    decision = parse_error_decision('{"action":"correct","target_class":"new"}')
    assert decision.action == "correct_class"
    assert decision.target_class == "new"


def test_vlm_auto_label_writes_new_dataset(tmp_path):
    dataset = _dataset(tmp_path / "source")
    provider = FakeVLM(['{"boxes":[{"class_name":"new","bbox":[0.5,0.5,0.2,0.3],"confidence":0.9}]}'])
    summary = generate_yolo_labels(
        dataset,
        tmp_path / "out",
        provider,
        workers=1,
        progress=False,
    )
    assert summary.succeeded == 1
    assert (tmp_path / "out" / "images" / "a.jpg").exists()
    assert (tmp_path / "out" / "labels" / "a.txt").read_text(encoding="utf-8").startswith("1 ")
    assert (tmp_path / "out" / "vlm_results.json").exists()


def test_vlm_error_plan_can_be_applied_with_existing_correction_api(tmp_path):
    dataset = _dataset(tmp_path / "source")
    crop_dir = tmp_path / "errors" / "review" / "pred_gt" / "pred_new_gt_old" / "crops"
    crop_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(crop_dir / "a_pred1_gt1.jpg")
    provider = FakeVLM(['{"action":"correct_class","target_class":"new","confidence":0.99}'])

    plan = verify_error_crops(
        dataset,
        tmp_path / "errors",
        tmp_path / "plan",
        provider,
        mode="class",
        workers=1,
        progress=False,
    )
    assert plan["class_crop_map"]
    result = apply_correction_plan(
        dataset,
        plan,
        backup_dir=tmp_path / "backups",
    )
    assert result["class"]["report_rows"] == 1
    assert dataset.images[0].annotations[0].class_id == 1


def test_vlm_attribute_error_plan_can_be_applied(tmp_path):
    dataset = _dataset(tmp_path / "source")
    dataset.attributes = AttributeSchema({"defect": ["no", "yes"]})
    dataset.images[0].annotations[0].attributes = [0.0]
    crop_dir = tmp_path / "errors" / "review" / "attribute_error" / "attribute_defect" / "gt_no_pred_yes" / "crops"
    crop_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(crop_dir / "a_pred1_gt1_defect.jpg")
    provider = FakeVLM(['{"action":"correct_attribute","attribute_name":"defect","attribute_value":"yes","confidence":0.99}'])

    plan = verify_error_crops(
        dataset,
        tmp_path / "errors",
        tmp_path / "plan",
        provider,
        mode="attribute",
        workers=1,
        progress=False,
    )
    result = apply_correction_plan(dataset, plan, dry_run=False, backup_dir=tmp_path / "backups")
    assert result["attribute"]["report_rows"] == 1
    assert dataset.images[0].annotations[0].attributes == [1.0]


def test_assistant_requires_confirmation_for_write_methods():
    class Manager:
        def ann_update_from_map(self, class_map=None, **kwargs):
            return 0

    provider = FakeVLM([
        '{"method":"ann_update_from_map","arguments":{"class_map":{"merge":{}}},"explanation":"update classes"}'
    ])
    manager = Manager()
    with pytest.raises(AssistantError):
        run_assistant(manager, "update the classes", provider, execute=True)
    result = run_assistant(
        manager,
        "update the classes",
        FakeVLM([
            '{"method":"ann_update_from_map","arguments":{"class_map":{"merge":{}}}}'
        ]),
        execute=True,
        confirm=True,
    )
    assert result["executed"] is True


def test_vlm_dotenv_values_take_priority_over_process_environment(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "VLM_PROVIDER=qwen\nVLM_MODEL=model-from-dotenv\nVLM_TIMEOUT=45\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VLM_MODEL", "model-from-process")
    monkeypatch.setenv("VLM_TIMEOUT", "90")

    config = load_vlm_config(dotenv, overrides={"VLM_TIMEOUT": "12"})

    assert config.model == "model-from-dotenv"
    assert config.timeout == 12.0
