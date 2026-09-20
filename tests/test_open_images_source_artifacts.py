import base64
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_staged_open_images_sources_and_candidates_remain_hash_bound_and_quarantined():
    source_root = ROOT / "data" / "open_images_v7_source"
    provenance = json.loads(
        (source_root / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8")
    )
    assert provenance["training_eligible"] is False
    assert provenance["commercial_use_approved"] is False
    for artifact in provenance["official_files"]:
        path = source_root / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]

    originals = provenance["quarantined_originals"]
    assert originals["verified_images"] == 109
    assert originals["acquisition_status"].endswith("reconfirmed_2026-09-19")
    for artifact in originals["acquisition_reports"]:
        report_path = source_root / artifact["path"]
        assert report_path.stat().st_size == artifact["bytes"]
        assert _sha256(report_path) == artifact["sha256"]
        download_report = json.loads(report_path.read_text(encoding="utf-8"))
        assert download_report["training_eligible"] is False
        assert download_report["commercial_use_approved"] is False

    latest_report = json.loads(
        (source_root / originals["acquisition_reports"][-1]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert latest_report["selection"]["attempted"] == 3
    assert latest_report["selection"]["failures"] == {"http_429": 3}
    assert latest_report["selection"]["stopped_reason"] == (
        "consecutive_http_429_circuit_breaker"
    )

    batch = provenance["license_review_batch"]
    batch_root = source_root / batch["directory"]
    paths = {
        "inventory_sha256": batch_root / "CANDIDATES.jsonl",
        "provenance_sha256": batch_root / "PROVENANCE.json",
        "attribution_sha256": batch_root / "ATTRIBUTION.csv",
        "review_template_sha256": batch_root / "LICENSE_REVIEW_TEMPLATE.csv",
        "review_html_sha256": batch_root / "license-review.html",
    }
    for digest_field, path in paths.items():
        assert _sha256(path) == batch[digest_field]
    rows = [
        json.loads(line)
        for line in paths["inventory_sha256"].read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == batch["candidate_count"] == 109
    for row in rows:
        assert row["training_eligible"] is False
        assert row["commercial_use_approved"] is False
        assert row["damage_label_status"] == "unreviewed"
        image = Path(row["image"])
        payload = image.read_bytes()
        assert len(payload) == row["original_size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == row["image_sha256"]
        assert hashlib.md5(payload, usedforsecurity=False).hexdigest() == row["image_md5_hex"]
        assert base64.b64decode(row["official_original_md5"], validate=True) == hashlib.md5(
            payload, usedforsecurity=False
        ).digest()

    quality = batch["capture_quality_audit"]
    quality_json = batch_root / "CAPTURE_QUALITY.json"
    quality_csv = batch_root / "CAPTURE_QUALITY.csv"
    assert quality_json.stat().st_size == quality["json_bytes"]
    assert quality_csv.stat().st_size == quality["csv_bytes"]
    assert _sha256(quality_json) == quality["json_sha256"]
    assert _sha256(quality_csv) == quality["csv_sha256"]
    quality_report = json.loads(quality_json.read_text(encoding="utf-8"))
    assert quality_report["training_eligible"] is False
    assert quality_report["commercial_use_approved"] is False
    assert quality_report["summary"] == {
        key: quality[key]
        for key in (
            "images",
            "quality_pass",
            "manual_quality_review",
            "recapture_required",
            "duplicate_content",
        )
    }
    inventory_hashes = {row["image_sha256"] for row in rows}
    assert {row["sha256"] for row in quality_report["records"]} == inventory_hashes
