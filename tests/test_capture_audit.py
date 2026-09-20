import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from vehicle_damage.capture_audit import audit_capture_directory


ROOT = Path(__file__).resolve().parents[1]


def _noise_image(path, size, seed):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(40, 220, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)
    return pixels


def test_capture_audit_routes_quality_and_duplicate_cases(tmp_path):
    images = tmp_path / "captures"
    images.mkdir()
    _noise_image(images / "a-good.png", (640, 640), 1)
    glare = _noise_image(images / "glare.png", (640, 640), 2)
    glare[100:210, 100:210] = 255
    Image.fromarray(glare, mode="RGB").save(images / "glare.png")
    _noise_image(images / "small.png", (160, 120), 3)
    (images / "broken.jpg").write_bytes(b"not an image")
    (images / "z-good-copy.png").write_bytes((images / "a-good.png").read_bytes())

    report = audit_capture_directory(images)
    records = {row["relative_path"]: row for row in report["records"]}
    assert report["training_eligible"] is False
    assert report["commercial_use_approved"] is False
    assert report["summary"] == {
        "images": 5,
        "quality_pass": 1,
        "manual_quality_review": 2,
        "recapture_required": 2,
        "duplicate_content": 1,
    }
    assert records["a-good.png"]["disposition"] == "quality_pass"
    assert records["z-good-copy.png"]["duplicate_of"] == "a-good.png"
    assert "duplicate_content" in records["z-good-copy.png"]["review_reasons"]
    assert records["glare.png"]["disposition"] == "manual_quality_review"
    assert "possible_specular_glare" in records["glare.png"]["review_reasons"]
    assert "low_resolution" in records["small.png"]["recapture_reasons"]
    assert records["broken.jpg"]["recapture_reasons"] == ["unreadable_image"]


def test_capture_audit_cli_writes_json_and_csv(tmp_path):
    images = tmp_path / "captures"
    images.mkdir()
    _noise_image(images / "car.png", (640, 640), 4)
    output = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_capture_quality.py"),
            "--images-dir",
            str(images),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["images"] == 1
    with output.with_suffix(".csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["relative_path"] == "car.png"
