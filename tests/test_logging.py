from datetime import datetime
import re

from PIL import Image

from yolo_data_manager.cli import main as cli_main
from yolo_data_manager.logging_utils import log_file
from yolo_data_manager.scripting import YoloManager


def _write_dataset(root, *, with_attributes: bool = False) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    Image.new("RGB", (20, 20), color="white").save(root / "images" / "a.jpg")
    (root / "class.txt").write_text("object\n", encoding="utf-8")
    if with_attributes:
        (root / "attribute.yaml").write_text(
            "attributes:\n  defect: [no, yes]\n",
            encoding="utf-8",
        )
        label = "0 1 0 0.5 0.5 0.2 0.2\n"
    else:
        label = "0 0.5 0.5 0.2 0.2\n"
    (root / "labels" / "a.txt").write_text(label, encoding="utf-8")


def test_cli_writes_daily_log_and_timestamped_console(tmp_path, capsys):
    root = tmp_path / "logged_dataset"
    _write_dataset(root)

    assert cli_main(
        ["check", "--root", str(root), "--layout", "flat", "--no-progress"]
    ) == 0

    log_path = log_file(root)
    assert log_path == root / "ydm_log" / f"{datetime.now():%Y-%m-%d}.log"
    log_text = log_path.read_text(encoding="utf-8")
    assert "START operation=check" in log_text
    assert "END operation=check status=ok" in log_text

    captured = capsys.readouterr()
    assert re.search(
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}[+-]\d{4}\] "
        r"\[INFO\] START operation=check",
        captured.err,
    )


def test_direct_manager_operation_uses_same_daily_log(tmp_path):
    root = tmp_path / "direct_logged_dataset"
    _write_dataset(root, with_attributes=True)
    manager = YoloManager(
        root,
        layout="flat",
        task="detect",
        init_layout=False,
        init_check=False,
    )

    assert manager.output_log == root / "ydm_log"
    assert manager.ann_att_update_from_map(
        {"update": {"defect": {"no": "yes"}}},
        progress=False,
    ) == 0

    log_text = log_file(root).read_text(encoding="utf-8")
    assert "START operation=ann_att_update_from_map" in log_text
    assert "END operation=ann_att_update_from_map status=ok" in log_text
