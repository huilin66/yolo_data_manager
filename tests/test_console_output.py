from yolo_data_manager.cli import _compact_console_payload


def test_compact_console_payload_limits_long_lists():
    payload = {
        "changed": 12,
        "invalid_crops": [f"crop_{index}.jpg" for index in range(25)],
    }

    compacted = _compact_console_payload(payload)

    assert compacted["changed"] == 12
    assert len(compacted["invalid_crops"]) == 21
    assert compacted["invalid_crops"][-1] == (
        "... (5 more items omitted; see the edit report)"
    )
    assert len(payload["invalid_crops"]) == 25
