"""Prepare COCO car photos as full-scene clean-negative candidates.

This is an experimental source expansion. COCO has car bounding boxes but no
vehicle-damage labels, so damage masks are intentionally empty and the rows are
tagged as candidates rather than production ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def prepare(
    annotations: Path,
    images_dir: Path,
    output_dir: Path,
    *,
    limit: int,
    validation_count: int,
    calibration_count: int,
    test_count: int,
) -> dict[str, int]:
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    image_rows = {int(row["id"]): row for row in payload["images"]}
    boxes: dict[int, list[list[float]]] = {}
    for annotation in payload["annotations"]:
        if int(annotation["category_id"]) != 3 or not annotation.get("bbox"):
            continue
        x, y, width, height = (float(value) for value in annotation["bbox"])
        if width > 1 and height > 1:
            boxes.setdefault(int(annotation["image_id"]), []).append(
                [x, y, width, height]
            )

    selected = [image_rows[key] for key in sorted(boxes)]
    if limit > 0:
        selected = selected[:limit]
    required = validation_count + calibration_count + test_count
    if len(selected) <= required:
        raise ValueError("limit must leave at least one training image")

    image_root = images_dir.resolve()
    output_dir = output_dir.resolve()
    mask_dir = output_dir / "masks"
    exterior_dir = output_dir / "exterior_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    exterior_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for index, image_row in enumerate(selected):
        image_path = image_root / str(image_row["file_name"])
        if not image_path.is_file():
            continue
        with Image.open(image_path) as source:
            width, height = source.size
        damage = np.zeros((height, width), dtype=np.uint8)
        exterior = np.zeros((height, width), dtype=np.uint8)
        for x, y, box_width, box_height in boxes[int(image_row["id"])]:
            left = max(0, min(width, int(round(x))))
            top = max(0, min(height, int(round(y))))
            right = max(left, min(width, int(round(x + box_width))))
            bottom = max(top, min(height, int(round(y + box_height))))
            exterior[top:bottom, left:right] = 255
        mask_name = f"{int(image_row['id']):012d}.png"
        Image.fromarray(damage, mode="L").save(mask_dir / mask_name)
        Image.fromarray(exterior, mode="L").save(exterior_dir / mask_name)
        if index < len(selected) - test_count - calibration_count - validation_count:
            split = "train"
        elif index < len(selected) - test_count - calibration_count:
            split = "validation"
        elif index < len(selected) - test_count:
            split = "calibration"
        else:
            split = "test"
        rows.append(
            {
                "image": str(image_path),
                "mask": str((mask_dir / mask_name).resolve()),
                "exterior_mask": str((exterior_dir / mask_name).resolve()),
                "damage_supervised": True,
                "split": split,
                "group_id": f"coco_val_{int(image_row['id']):012d}",
                "source": "COCO 2017 validation car candidates",
                "source_url": "https://cocodataset.org/#download",
                "license_id": "COCO-CC-BY-4.0-candidate",
                "commercial_use": True,
                "tags": ["coco", "car_bbox", "clean_negative_candidate"],
            }
        )

    output_manifest = output_dir / "manifest.jsonl"
    output_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "selected": len(selected),
        "written": len(rows),
        "train": sum(row["split"] == "train" for row in rows),
        "validation": sum(row["split"] == "validation" for row in rows),
        "calibration": sum(row["split"] == "calibration" for row in rows),
        "test": sum(row["split"] == "test" for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--validation-count", type=int, default=150)
    parser.add_argument("--calibration-count", type=int, default=150)
    parser.add_argument("--test-count", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
