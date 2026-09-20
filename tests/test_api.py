import io
import json

from fastapi.testclient import TestClient
from PIL import Image

import numpy as np

from vehicle_damage.api import _components, create_app, selected_artifacts


class _FakePredictor:
    def health(self):
        return {
            "ok": True,
            "model_loaded": True,
            "weights_exist": True,
            "cuda": True,
            "production_approved": False,
            "triage_threshold": 0.45,
            "segmentation_threshold": 0.55,
        }

    def predict_bytes(self, payload):
        assert payload
        return {
            "decision": "manual_review_required",
            "production_approved": False,
            "triage_alert": True,
            "high_confidence_alert": False,
        }


def test_selected_frontend_artifacts_are_hash_verified():
    artifacts = selected_artifacts()
    assert artifacts["checkpoint"].name == "best.pt"
    assert artifacts["development_profile"].suffix == ".json"
    assert artifacts["selection"]["production_approved"] is False


def test_damage_api_health_upload_and_media_type_contract():
    app = create_app(lambda: _FakePredictor())
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["damage"]["model_loaded"] is True

        rejected = client.post(
            "/api/damage/predict",
            files={"file": ("car.txt", b"not an image", "text/plain")},
        )
        assert rejected.status_code == 415

        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), "gray").save(buffer, format="PNG")
        response = client.post(
            "/api/damage/predict",
            files={"file": ("car.png", buffer.getvalue(), "image/png")},
        )
        assert response.status_code == 200
        assert response.json()["decision"] == "manual_review_required"


def test_selected_artifact_integrity_failure_is_explicit(tmp_path):
    selected = selected_artifacts()["selection"]
    tampered = json.loads(json.dumps(selected))
    tampered["artifacts"]["checkpoint"]["sha256"] = "0" * 64
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    try:
        selected_artifacts(path)
    except ValueError as exc:
        assert "pinned integrity check" in str(exc)
    else:
        raise AssertionError("a mismatched selected checkpoint hash must fail closed")


def test_component_table_groups_connected_mixed_class_regions():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    mask[4:6, 4:6] = 2
    rows = _components(
        mask,
        np.full(mask.shape, 0.8, dtype=np.float32),
        ["background", "dent", "scratch"],
        kind="segmentation",
        minimum_pixels=16,
    )
    assert len(rows) == 1
    assert rows[0]["class_name"] == "dent"
    assert rows[0]["area_pixels"] == 36
