from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import requests

from .open_images_intake import (
    CC_BY_2_URL,
    METADATA_COLUMNS,
    _decode_official_md5,
    _is_http_url,
    _load_car_boxes,
    _load_car_label_ids,
    _require_columns,
    _sha256,
    _valid_rotation,
)


def _eligible_metadata_rows(
    metadata: Path, eligible_image_ids: set[str], seed: int
) -> list[dict[str, str]]:
    rows = []
    with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader, METADATA_COLUMNS, metadata)
        for row in reader:
            image_id = row["ImageID"].strip()
            if image_id not in eligible_image_ids or row["License"].strip() != CC_BY_2_URL:
                continue
            if not all(
                row[field].strip()
                for field in (
                    "OriginalURL", "OriginalLandingURL", "AuthorProfileURL", "Author",
                    "Title", "OriginalSize", "OriginalMD5", "Subset",
                )
            ):
                continue
            if not all(
                _is_http_url(row[field])
                for field in (
                    "OriginalURL", "OriginalLandingURL", "AuthorProfileURL", "License"
                )
            ):
                continue
            if _decode_official_md5(row["OriginalMD5"]) is None:
                continue
            if _valid_rotation(row["Rotation"]) is None:
                continue
            try:
                size = int(row["OriginalSize"])
            except ValueError:
                continue
            if size <= 0:
                continue
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f'{seed}:{row["ImageID"]}'.encode("utf-8")).hexdigest(),
            row["ImageID"],
        ),
    )


def download_open_images_originals(
    *,
    metadata: str | Path,
    boxes: str | Path,
    class_descriptions: str | Path,
    output_dir: str | Path,
    limit: int,
    max_attempts: int = 200,
    max_image_bytes: int = 20_000_000,
    seed: int = 101,
    timeout_seconds: float = 20.0,
    start_offset: int = 0,
    report_name: str = "DOWNLOAD_REPORT.json",
    delay_seconds: float = 0.0,
    max_consecutive_rate_limits: int = 3,
    request_get: Callable[..., object] = requests.get,
) -> dict[str, object]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if max_attempts < limit:
        raise ValueError("max_attempts must be at least limit")
    if max_image_bytes <= 0 or timeout_seconds <= 0 or delay_seconds < 0:
        raise ValueError("download limits must be positive")
    if start_offset < 0:
        raise ValueError("start_offset must be non-negative")
    if max_consecutive_rate_limits <= 0:
        raise ValueError("max_consecutive_rate_limits must be positive")
    if Path(report_name).name != report_name or not report_name.endswith(".json"):
        raise ValueError("report_name must be a JSON filename without directories")
    metadata_path = Path(metadata).resolve()
    boxes_path = Path(boxes).resolve()
    classes_path = Path(class_descriptions).resolve()
    for path in (metadata_path, boxes_path, classes_path):
        if not path.is_file():
            raise ValueError(f"missing input file: {path}")
    car_labels = _load_car_label_ids(classes_path)
    car_boxes = _load_car_boxes(boxes_path, car_labels)
    rows = _eligible_metadata_rows(metadata_path, set(car_boxes), seed)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    failures: Counter[str] = Counter()
    rejected = []
    downloaded = []
    attempted = 0
    consecutive_rate_limits = 0
    stopped_reason = None
    for row in rows[start_offset:]:
        if len(downloaded) >= limit or attempted >= max_attempts:
            break
        attempted += 1
        if attempted > 1 and delay_seconds:
            time.sleep(delay_seconds)
        image_id = row["ImageID"].strip()
        expected_size = int(row["OriginalSize"])
        expected_md5 = _decode_official_md5(row["OriginalMD5"])
        if expected_size > max_image_bytes:
            failures["declared_size_exceeds_limit"] += 1
            rejected.append({"image_id": image_id, "reason": "declared_size_exceeds_limit"})
            continue
        target = output / f"{image_id}.jpg"
        if target.exists():
            existing = target.read_bytes()
            if len(existing) == expected_size and hashlib.md5(
                existing, usedforsecurity=False
            ).digest() == expected_md5:
                downloaded.append({"image_id": image_id, "status": "reused_verified"})
            else:
                failures["existing_file_mismatch"] += 1
                rejected.append({"image_id": image_id, "reason": "existing_file_mismatch"})
            continue
        part = output / f"{image_id}.jpg.part"
        if part.exists():
            part.unlink()
        response = None
        try:
            response = request_get(
                row["OriginalURL"].strip(),
                stream=True,
                timeout=(timeout_seconds, timeout_seconds),
                headers={"User-Agent": "vehicle-damage-dataset-intake/0.1"},
            )
            response.raise_for_status()
            received = 0
            digest = hashlib.md5(usedforsecurity=False)
            with part.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > max_image_bytes or received > expected_size:
                        raise ValueError("download exceeds declared size")
                    digest.update(chunk)
                    handle.write(chunk)
            if received != expected_size:
                raise ValueError("download size mismatch")
            if digest.digest() != expected_md5:
                raise ValueError("download MD5 mismatch")
            part.replace(target)
            downloaded.append({"image_id": image_id, "status": "downloaded_verified"})
            consecutive_rate_limits = 0
        except requests.RequestException as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            reason = (
                f"http_{status_code}"
                if status_code is not None
                else type(exc).__name__.replace("Error", "").casefold()
            )
            failures[reason] += 1
            rejected.append({"image_id": image_id, "reason": reason})
            if status_code == 429:
                consecutive_rate_limits += 1
                if consecutive_rate_limits >= max_consecutive_rate_limits:
                    stopped_reason = "consecutive_http_429_circuit_breaker"
                    break
            else:
                consecutive_rate_limits = 0
        except (OSError, ValueError):
            failures["content_verification_failed"] += 1
            rejected.append({"image_id": image_id, "reason": "content_verification_failed"})
            consecutive_rate_limits = 0
        finally:
            if response is not None:
                response.close()
            if part.exists():
                part.unlink()
    report: dict[str, object] = {
        "status": "original_files_downloaded_but_not_license_or_damage_approved",
        "training_eligible": False,
        "commercial_use_approved": False,
        "inputs": {
            "metadata": {"path": str(metadata_path), "sha256": _sha256(metadata_path)},
            "boxes": {"path": str(boxes_path), "sha256": _sha256(boxes_path)},
            "class_descriptions": {
                "path": str(classes_path), "sha256": _sha256(classes_path),
            },
        },
        "selection": {
            "seed": seed,
            "start_offset": start_offset,
            "requested": limit,
            "max_attempts": max_attempts,
            "delay_seconds": delay_seconds,
            "max_consecutive_rate_limits": max_consecutive_rate_limits,
            "eligible_metadata_rows": len(rows),
            "attempted": attempted,
            "verified_originals": len(downloaded),
            "failures": dict(sorted(failures.items())),
            "stopped_reason": stopped_reason,
        },
        "files": downloaded,
        "rejected_files": rejected,
    }
    (output / report_name).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
