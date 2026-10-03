from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from PIL import Image

from yolo_data_manager.web.server import create_app


def _make_dataset(root):
    images = root / "images"
    labels = root / "labels"
    images.mkdir()
    labels.mkdir()
    (root / "data.yaml").write_text(
        "path: .\nnames:\n  0: car\n  1: bus\n",
        encoding="utf-8",
    )
    for index, label in enumerate(("0 0.5 0.5 0.4 0.3\n", "1 0.4 0.5 0.2 0.2\n"), start=1):
        Image.new("RGB", (320, 200), (40 * index, 80, 140)).save(images / f"{index:06d}.jpg")
        (labels / f"{index:06d}.txt").write_text(label, encoding="utf-8")


def test_web_dataset_load_overview_and_preview(tmp_path):
    _make_dataset(tmp_path)
    client = TestClient(create_app())

    response = client.post("/api/dataset/load", json={"root": str(tmp_path), "task": "detect"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["counts"]["images"] == 2
    assert payload["counts"]["boxes"] == 2
    assert [item["name"] for item in payload["classes"]] == ["car", "bus"]

    images = client.get("/api/dataset/images?limit=10")
    assert images.status_code == 200
    assert images.json()["total"] == 2

    preview = client.get("/api/dataset/images/0/preview")
    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith("image/jpeg")

    assert client.post("/api/dataset/unload").status_code == 200
