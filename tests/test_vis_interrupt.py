from pathlib import Path

import pytest

from yolo_data_manager.core.models import YoloDataset
import yolo_data_manager.vis.renderer as renderer


@pytest.mark.parametrize(
    ("operation", "expected_message"),
    [
        (renderer.render_dataset, "vis draw cancelled"),
        (renderer.crop_dataset, "vis crop cancelled"),
    ],
)
def test_parallel_visualization_cancels_pending_work_on_keyboard_interrupt(
    tmp_path, monkeypatch, capsys, operation, expected_message
):
    executors = []

    class InterruptibleExecutor:
        def __init__(self, **_kwargs):
            self.shutdown_calls = []
            executors.append(self)

        def submit(self, *_args, **_kwargs):
            return object()

        def shutdown(self, *, wait, cancel_futures):
            self.shutdown_calls.append((wait, cancel_futures))

    def interrupt(_futures):
        raise KeyboardInterrupt

    monkeypatch.setattr(renderer, "ThreadPoolExecutor", InterruptibleExecutor)
    monkeypatch.setattr(renderer, "as_completed", interrupt)

    with pytest.raises(KeyboardInterrupt):
        operation(YoloDataset(root=Path(tmp_path), images=[]), tmp_path / "out", workers=2, progress=False)

    assert executors[0].shutdown_calls == [(False, True)]
    assert expected_message in capsys.readouterr().err
