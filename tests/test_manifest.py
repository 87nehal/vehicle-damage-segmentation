import json
from pathlib import Path
from unittest.mock import patch

import pytest

from vehicle_damage.manifest import Sample, load_manifest, validate_manifest


def test_commercial_gate_fails_closed():
    row = json.dumps({
        "image": "a.jpg", "mask": "a.png", "split": "train", "group_id": "v1",
        "source": "unknown", "license_id": "unknown", "commercial_use": False,
    })
    with patch.object(Path, "read_text", return_value=row):
        with pytest.raises(ValueError, match="commercial_use"):
            load_manifest("manifest.jsonl")


def test_group_leakage_is_detected():
    rows = [
        Sample(Path("a.jpg"), Path("a.png"), "train", "vehicle-1", "owned", "Proprietary", True),
        Sample(Path("b.jpg"), Path("b.png"), "test", "vehicle-1", "owned", "Proprietary", True),
    ]
    report = validate_manifest(rows, 3, check_files=False)
    assert not report["valid"]
    assert "group leakage" in report["errors"][0]


def test_damage_unsupervised_requires_exterior_mask():
    rows = [
        Sample(
            Path("a.jpg"), Path("a.png"), "train", "vehicle-1",
            "owned", "Proprietary", True, damage_supervised=False,
        )
    ]
    report = validate_manifest(rows, 3, check_files=False)
    assert not report["valid"]
    assert report["damage_unsupervised_images"] == 1
    assert "requires an exterior_mask" in report["errors"][0]
