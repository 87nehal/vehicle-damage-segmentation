from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .inference import assess_quality, capture_quality_disposition


SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)
CSV_FIELDS = (
    "relative_path",
    "sha256",
    "bytes",
    "width",
    "height",
    "disposition",
    "review_reasons",
    "recapture_reasons",
    "duplicate_of",
    "dark_fraction",
    "clipped_highlight_fraction",
    "specular_highlight_fraction",
    "sharpness",
)


def _failed_record(path: Path, root: Path, reason: str) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": None,
        "bytes": None,
        "width": None,
        "height": None,
        "disposition": "recapture_required",
        "review_reasons": [reason],
        "recapture_reasons": [reason],
        "duplicate_of": None,
        "dark_fraction": None,
        "clipped_highlight_fraction": None,
        "specular_highlight_fraction": None,
        "sharpness": None,
    }


def audit_capture_directory(images_dir: str | Path) -> dict[str, object]:
    """Screen captured images without granting license, label, or training approval."""
    root = Path(images_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"missing capture directory: {root}")
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"no supported images found in capture directory: {root}")

    records: list[dict[str, object]] = []
    first_path_by_hash: dict[str, str] = {}
    for path in paths:
        if path.is_symlink():
            records.append(_failed_record(path, root, "symbolic_link_not_allowed"))
            continue
        relative = path.relative_to(root).as_posix()
        try:
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            with Image.open(path) as source:
                oriented = ImageOps.exif_transpose(source).convert("RGB")
                oriented.load()
        except (OSError, ValueError, UnidentifiedImageError):
            records.append(_failed_record(path, root, "unreadable_image"))
            continue

        quality = assess_quality(oriented)
        disposition, recapture_reasons = capture_quality_disposition(quality)
        review_reasons = list(quality.review_reasons)
        duplicate_of = first_path_by_hash.get(digest)
        if duplicate_of is None:
            first_path_by_hash[digest] = relative
        else:
            review_reasons.append("duplicate_content")
            if disposition == "quality_pass":
                disposition = "manual_quality_review"
        records.append(
            {
                "relative_path": relative,
                "sha256": digest,
                "bytes": len(payload),
                "width": quality.width,
                "height": quality.height,
                "disposition": disposition,
                "review_reasons": review_reasons,
                "recapture_reasons": list(recapture_reasons),
                "duplicate_of": duplicate_of,
                "dark_fraction": quality.dark_fraction,
                "clipped_highlight_fraction": quality.clipped_highlight_fraction,
                "specular_highlight_fraction": quality.specular_highlight_fraction,
                "sharpness": quality.sharpness,
            }
        )

    counts = Counter(str(row["disposition"]) for row in records)
    return {
        "status": "capture_quality_screening_not_training_approval",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "images_dir": str(root),
        "training_eligible": False,
        "commercial_use_approved": False,
        "damage_labels_present": False,
        "warning": (
            "Quality screening does not establish consent, commercial rights, damage "
            "labels, reviewer independence, or release-test eligibility."
        ),
        "summary": {
            "images": len(records),
            "quality_pass": counts["quality_pass"],
            "manual_quality_review": counts["manual_quality_review"],
            "recapture_required": counts["recapture_required"],
            "duplicate_content": sum(
                row["duplicate_of"] is not None for row in records
            ),
        },
        "records": records,
    }


def write_capture_audit(
    report: dict[str, object],
    output_json: str | Path,
    output_csv: str | Path | None = None,
) -> tuple[Path, Path]:
    json_path = Path(output_json).resolve()
    csv_path = (
        Path(output_csv).resolve()
        if output_csv is not None
        else json_path.with_suffix(".csv")
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for source in report["records"]:
            row = dict(source)
            row["review_reasons"] = ";".join(row["review_reasons"])
            row["recapture_reasons"] = ";".join(row["recapture_reasons"])
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})
    return json_path, csv_path
