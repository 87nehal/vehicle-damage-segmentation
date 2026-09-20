import base64
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import requests
from PIL import Image

from vehicle_damage.manifest import load_manifest
from vehicle_damage.open_images_download import download_open_images_originals
from vehicle_damage.open_images_intake import (
    CC_BY_2_URL,
    load_license_approved_candidates,
    prepare_open_images_candidates,
)


ROOT = Path(__file__).resolve().parents[1]


def _assert_embedded_javascript_parses(path):
    node = shutil.which("node")
    if node is None:
        return
    html = path.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.DOTALL)
    assert scripts
    subprocess.run(
        [node, "--check", "-"],
        input="\n".join(scripts),
        text=True,
        check=True,
        capture_output=True,
    )


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metadata_row(image_id, image_bytes, **updates):
    row = {
        "ImageID": image_id,
        "Subset": "train",
        "OriginalURL": f"https://images.example/{image_id}.jpg",
        "OriginalLandingURL": f"https://source.example/photo/{image_id}",
        "License": CC_BY_2_URL,
        "AuthorProfileURL": "https://source.example/photographer",
        "Author": "Example Photographer",
        "Title": f"Car {image_id}",
        "OriginalSize": str(len(image_bytes)),
        "OriginalMD5": base64.b64encode(
            hashlib.md5(image_bytes, usedforsecurity=False).digest()
        ).decode("ascii"),
        "Thumbnail300KURL": "https://thumbnail.example/image.jpg",
        "Rotation": "0",
    }
    row.update(updates)
    return row


def _box_row(image_id, **updates):
    row = {
        "ImageID": image_id,
        "Source": "xclick",
        "LabelName": "/m/car",
        "Confidence": "1",
        "XMin": "0.1",
        "XMax": "0.9",
        "YMin": "0.2",
        "YMax": "0.8",
        "IsOccluded": "0",
        "IsTruncated": "0",
        "IsGroupOf": "0",
        "IsDepiction": "0",
        "IsInside": "0",
    }
    row.update(updates)
    return row


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield self.payload[:7]
        yield self.payload[7:]

    def close(self):
        self.closed = True


class _RateLimitedResponse(_FakeResponse):
    def raise_for_status(self):
        response = requests.Response()
        response.status_code = 429
        raise requests.HTTPError("rate limited", response=response)


def test_original_downloader_verifies_bytes_and_keeps_them_quarantined(tmp_path):
    source_image = tmp_path / "source.jpg"
    Image.new("RGB", (20, 14), "silver").save(source_image)
    payload = source_image.read_bytes()
    classes = tmp_path / "classes.csv"
    _write_csv(
        classes,
        ["LabelName", "DisplayName"],
        [{"LabelName": "/m/car", "DisplayName": "Car"}],
    )
    boxes = tmp_path / "boxes.csv"
    _write_csv(boxes, list(_box_row("download-me")), [_box_row("download-me")])
    metadata = tmp_path / "metadata.csv"
    row = _metadata_row("download-me", payload)
    _write_csv(metadata, list(row), [row])
    response = _FakeResponse(payload)

    report = download_open_images_originals(
        metadata=metadata,
        boxes=boxes,
        class_descriptions=classes,
        output_dir=tmp_path / "downloaded",
        limit=1,
        max_attempts=1,
        request_get=lambda *args, **kwargs: response,
    )
    assert report["training_eligible"] is False
    assert report["commercial_use_approved"] is False
    assert report["selection"]["verified_originals"] == 1
    assert (tmp_path / "downloaded" / "download-me.jpg").read_bytes() == payload
    assert response.closed
    assert not list((tmp_path / "downloaded").glob("*.part"))


def test_original_downloader_stops_after_consecutive_rate_limits(tmp_path):
    source_image = tmp_path / "source.jpg"
    Image.new("RGB", (20, 14), "silver").save(source_image)
    payload = source_image.read_bytes()
    classes = tmp_path / "classes.csv"
    _write_csv(
        classes,
        ["LabelName", "DisplayName"],
        [{"LabelName": "/m/car", "DisplayName": "Car"}],
    )
    image_ids = [f"rate-limited-{index}" for index in range(5)]
    boxes = tmp_path / "boxes.csv"
    _write_csv(boxes, list(_box_row(image_ids[0])), [_box_row(value) for value in image_ids])
    metadata = tmp_path / "metadata.csv"
    rows = [_metadata_row(value, payload) for value in image_ids]
    _write_csv(metadata, list(rows[0]), rows)

    report = download_open_images_originals(
        metadata=metadata,
        boxes=boxes,
        class_descriptions=classes,
        output_dir=tmp_path / "downloaded",
        limit=1,
        max_attempts=5,
        max_consecutive_rate_limits=3,
        request_get=lambda *args, **kwargs: _RateLimitedResponse(payload),
    )
    assert report["selection"]["attempted"] == 3
    assert report["selection"]["failures"] == {"http_429": 3}
    assert report["selection"]["stopped_reason"] == (
        "consecutive_http_429_circuit_breaker"
    )


def test_open_images_intake_is_hash_verified_and_never_training_approved(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    image_bytes = {}
    for image_id, color in (
        ("valid", "gray"), ("wrong-license", "white"), ("bad-md5", "black")
    ):
        path = images / f"{image_id}.jpg"
        Image.new("RGB", (32, 24), color).save(path)
        image_bytes[image_id] = path.read_bytes()

    classes = tmp_path / "classes.csv"
    _write_csv(
        classes,
        ["LabelName", "DisplayName"],
        [{"LabelName": "/m/car", "DisplayName": "Car"}],
    )
    boxes = tmp_path / "boxes.csv"
    _write_csv(
        boxes,
        list(_box_row("valid")),
        [
            _box_row("valid"),
            _box_row("wrong-license"),
            _box_row("bad-md5"),
            _box_row("machine-box", Source="activemil"),
            _box_row("depiction", IsDepiction="1"),
        ],
    )
    metadata = tmp_path / "metadata.csv"
    rows = [
        _metadata_row("valid", image_bytes["valid"], Rotation="90"),
        _metadata_row(
            "wrong-license",
            image_bytes["wrong-license"],
            License="https://creativecommons.org/licenses/by-nc/2.0/",
        ),
        _metadata_row(
            "bad-md5", image_bytes["bad-md5"], OriginalMD5="AAAAAAAAAAAAAAAAAAAAAA=="
        ),
    ]
    _write_csv(metadata, list(rows[0]), rows)

    output = tmp_path / "review"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_open_images_candidates.py"),
            "--metadata", str(metadata),
            "--boxes", str(boxes),
            "--class-descriptions", str(classes),
            "--images-dir", str(images),
            "--output-dir", str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    candidates = [
        json.loads(line)
        for line in (output / "CANDIDATES.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["image_id"] == "valid"
    assert candidate["image_sha256"] == hashlib.sha256(image_bytes["valid"]).hexdigest()
    assert candidate["commercial_use_approved"] is False
    assert candidate["training_eligible"] is False
    assert candidate["damage_label_status"] == "unreviewed"
    assert candidate["rotation_degrees_counterclockwise"] == 90
    assert candidate["intake_status"] == "quarantined_pending_per_image_license_review"
    assert "mask" not in candidate
    assert "commercial_use" not in candidate

    provenance = json.loads((output / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert provenance["status"] == "quarantined_review_candidates_not_training_data"
    assert provenance["training_eligible"] is False
    assert provenance["selection"]["rejections"]["license_not_exact_cc_by_2"] == 1
    assert provenance["selection"]["rejections"]["original_md5_mismatch"] == 1
    assert (output / "ATTRIBUTION.csv").is_file()
    assert (output / "LICENSE_REVIEW_TEMPLATE.csv").is_file()
    assert (output / "license-review.html").is_file()
    license_page = (output / "license-review.html").read_text(encoding="utf-8")
    assert "transform:rotate(-90deg)" in license_page
    assert "Download completed CSV" in license_page
    assert "Download draft CSV" in license_page
    assert "draft-license-review.csv" in license_page
    assert "Draft CSVs are backups only" in license_page
    assert "Download state backup" in license_page
    assert "license-review-state.json" in license_page
    assert "open-images-license-review-state-v1" in license_page
    assert provenance["inventory"]["sha256"] in license_page
    assert "localStorage" in license_page
    assert "Confirm commercial ML training rights" in license_page
    assert "completed-license-review.csv" in license_page
    _assert_embedded_javascript_parses(output / "license-review.html")

    with (output / "LICENSE_REVIEW_TEMPLATE.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        license_rows = list(csv.DictReader(handle))
        license_fields = list(license_rows[0])

    draft_license = tmp_path / "draft-license-review.csv"
    draft_rows = [dict(row, reviewer_id="rights-reviewer-a") for row in license_rows]
    _write_csv(draft_license, license_fields, draft_rows)
    try:
        load_license_approved_candidates(output / "CANDIDATES.jsonl", draft_license)
    except ValueError as exc:
        assert "source_page_checked_at_utc" in str(exc)
    else:
        raise AssertionError("an incomplete draft review must fail closed")

    license_rows[0].update(
        reviewer_id="rights-reviewer-a",
        source_page_checked_at_utc="2026-09-19T12:00:00Z",
        decision="approve",
        observed_license_url=CC_BY_2_URL,
        observed_author="Example Photographer",
        commercial_ml_training_rights_confirmed="yes",
    )
    completed_license = tmp_path / "license-review.csv"
    _write_csv(completed_license, license_fields, license_rows)
    damage_batch = tmp_path / "damage-review"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_open_images_damage_review_batch.py"),
            "--candidates", str(output / "CANDIDATES.jsonl"),
            "--license-review", str(completed_license),
            "--output-dir", str(damage_batch),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    damage_provenance = json.loads(
        (damage_batch / "PROVENANCE.json").read_text(encoding="utf-8")
    )
    assert damage_provenance["training_eligible"] is False
    assert damage_provenance["samples"] == 1
    assert "transform:rotate(-90deg)" in (damage_batch / "review.html").read_text(
        encoding="utf-8"
    )
    _assert_embedded_javascript_parses(damage_batch / "review.html")

    with (damage_batch / "review-template.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        damage_rows = list(csv.DictReader(handle))
        damage_fields = list(damage_rows[0])
    visual_reviews = []
    for reviewer in ("vehicle-assessor-a", "vehicle-assessor-b"):
        review = dict(damage_rows[0])
        review.update(
            reviewer_id=reviewer,
            decision="clean",
            nuisance_tags="glare;panel_gap",
            distance="full_car",
            angle="oblique",
            lighting="daylight",
            cleanliness="clean",
        )
        path = tmp_path / f"{reviewer}.csv"
        _write_csv(path, damage_fields, [review])
        visual_reviews.append(path)

    clean_manifest = tmp_path / "clean" / "manifest.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "import_open_images_clean_reviews.py"),
            "--candidates", str(output / "CANDIDATES.jsonl"),
            "--license-review", str(completed_license),
            "--review-a", str(visual_reviews[0]),
            "--review-b", str(visual_reviews[1]),
            "--output-manifest", str(clean_manifest),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    promoted = load_manifest(clean_manifest)
    assert len(promoted) == 1
    assert promoted[0].commercial_use
    assert promoted[0].damage_supervised
    assert {
        "per_image_license_reviewed",
        "reviewed_clean",
        "double_reviewed",
        "orientation_normalized",
    } <= set(promoted[0].tags)
    with Image.open(promoted[0].image) as normalized:
        assert normalized.size == (24, 32)
    report = json.loads(
        Path(str(clean_manifest) + ".reviews.json").read_text(encoding="utf-8")
    )
    assert report["training_eligible"] is True
    assert report["clean_images_imported"] == 1
    assert len(report["orientation_normalized_images"]) == 1
    assert report["orientation_normalized_images"][0][
        "derived_image_sha256"
    ] == hashlib.sha256(promoted[0].image.read_bytes()).hexdigest()

    inventory_path = output / "CANDIDATES.jsonl"
    inventory_path.write_text(
        inventory_path.read_text(encoding="utf-8").replace(
            "Example Photographer", "Tampered Attribution"
        ),
        encoding="utf-8",
    )
    try:
        load_license_approved_candidates(inventory_path, completed_license)
    except ValueError as exc:
        assert "does not match its fail-closed provenance" in str(exc)
    else:
        raise AssertionError("tampered candidate inventory must fail closed")


def test_open_images_intake_fails_when_required_attribution_is_missing(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    image = images / "missing-author.jpg"
    Image.new("RGB", (16, 12), "gray").save(image)
    classes = tmp_path / "classes.csv"
    _write_csv(
        classes,
        ["LabelName", "DisplayName"],
        [{"LabelName": "/m/car", "DisplayName": "Car"}],
    )
    boxes = tmp_path / "boxes.csv"
    _write_csv(boxes, list(_box_row("missing-author")), [_box_row("missing-author")])
    metadata = tmp_path / "metadata.csv"
    row = _metadata_row("missing-author", image.read_bytes(), Author="")
    _write_csv(metadata, list(row), [row])

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_open_images_candidates.py"),
            "--metadata", str(metadata),
            "--boxes", str(boxes),
            "--class-descriptions", str(classes),
            "--images-dir", str(images),
            "--output-dir", str(tmp_path / "out"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "no candidates passed the fail-closed intake checks" in result.stderr


def test_open_images_intake_rejects_duplicate_metadata_rows(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    image = images / "duplicate.jpg"
    Image.new("RGB", (16, 12), "gray").save(image)
    classes = tmp_path / "classes.csv"
    _write_csv(
        classes,
        ["LabelName", "DisplayName"],
        [{"LabelName": "/m/car", "DisplayName": "Car"}],
    )
    boxes = tmp_path / "boxes.csv"
    _write_csv(boxes, list(_box_row("duplicate")), [_box_row("duplicate")])
    metadata = tmp_path / "metadata.csv"
    row = _metadata_row("duplicate", image.read_bytes())
    _write_csv(metadata, list(row), [row, row])

    try:
        prepare_open_images_candidates(
            metadata=metadata,
            boxes=boxes,
            class_descriptions=classes,
            images_dir=images,
            output_dir=tmp_path / "out",
        )
    except ValueError as exc:
        assert "no candidates passed the fail-closed intake checks" in str(exc)
    else:
        raise AssertionError("duplicate per-image metadata must fail closed")
