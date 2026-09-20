"""Import independently corrected CVAT masks into the training manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from vehicle_damage.annotation_import import decode_cvat_class_mask
from vehicle_damage.manifest import load_manifest, validate_manifest


NUM_CLASSES = 6
PATH_FIELDS = ("image", "mask", "exterior_mask", "hard_negative_mask")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _absolute_manifest_rows(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        for field in PATH_FIELDS:
            value = row.get(field)
            if value:
                candidate = Path(value)
                row[field] = str(
                    (candidate if candidate.is_absolute() else path.parent / candidate)
                    .resolve()
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import human-corrected CVAT masks into a training manifest"
    )
    parser.add_argument("--batch-manifest", required=True)
    parser.add_argument("--corrected-annotations", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--annotator-id", required=True)
    parser.add_argument("--reviewer-id", required=True)
    args = parser.parse_args()
    annotator_id = args.annotator_id.strip()
    reviewer_id = args.reviewer_id.strip()
    if not annotator_id or not reviewer_id:
        parser.error("annotator and reviewer IDs must be non-empty")
    if annotator_id == reviewer_id:
        parser.error("annotator and reviewer must be different people")

    batch_path = Path(args.batch_manifest)
    corrected_path = Path(args.corrected_annotations)
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    if batch.get("status") != "model_preannotations_require_human_correction":
        raise ValueError("batch manifest is not a model-preannotation artifact")
    source_manifest = Path(batch["source_manifest"])
    if _sha256(source_manifest) != batch["source_manifest_sha256"]:
        raise ValueError("source manifest changed after annotation export")
    rows = _absolute_manifest_rows(source_manifest)
    rows_by_hash = {}
    for row in rows:
        digest = _sha256(Path(row["image"]))
        if digest in rows_by_hash:
            raise ValueError("source manifest contains duplicate image hashes")
        rows_by_hash[digest] = row

    output_manifest = Path(args.output_manifest)
    mask_dir = output_manifest.parent / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    imported = []
    unchanged = 0
    preannotations_path = Path(batch["annotations_zip"])
    with zipfile.ZipFile(corrected_path) as corrected, zipfile.ZipFile(
        preannotations_path
    ) as preannotations:
        for sample in batch["samples"]:
            digest = sample["source_image_sha256"]
            row = rows_by_hash.get(digest)
            if row is None:
                raise ValueError(f"batch image is missing from source manifest: {digest}")
            if row.get("damage_supervised", True) is not False:
                raise ValueError("dense import can replace only damage-unsupervised rows")
            if _sha256(Path(row["image"])) != digest:
                raise ValueError("source image changed after annotation export")
            stem = Path(sample["archive_name"]).stem
            member = f"SegmentationClass/{stem}.png"
            try:
                corrected_bytes = corrected.read(member)
            except KeyError as error:
                raise ValueError(f"corrected archive is missing {member}") from error
            with Image.open(io.BytesIO(corrected_bytes)) as image:
                mask = decode_cvat_class_mask(np.asarray(image), NUM_CLASSES)
            with Image.open(row["image"]) as source_image:
                expected_shape = (source_image.height, source_image.width)
            if mask.shape != expected_shape:
                raise ValueError(f"corrected mask size mismatch for {stem}")
            positive_pixels = int((mask > 0).sum())
            if positive_pixels == 0:
                raise ValueError(
                    f"corrected damage mask is empty for {stem}; clean decisions "
                    "must use the independent double-review workflow"
                )
            with Image.open(io.BytesIO(preannotations.read(member))) as image:
                original = decode_cvat_class_mask(np.asarray(image), NUM_CLASSES)
            unchanged += int(np.array_equal(mask, original))
            mask_path = mask_dir / f"{stem}.png"
            Image.fromarray(mask, mode="L").save(mask_path, format="PNG", optimize=True)
            tags = set(str(value) for value in row.get("tags", []))
            tags.difference_update({"damage_unverified", "exterior_only"})
            tags.update(
                {
                    "dense_annotated",
                    "model_assisted_annotation",
                    "independently_reviewed_mask",
                }
            )
            row["mask"] = str(mask_path.resolve())
            row["damage_supervised"] = True
            row["tags"] = sorted(tags)
            imported.append(
                {
                    "group_id": row["group_id"],
                    "image_sha256": digest,
                    "mask": str(mask_path.resolve()),
                    "mask_sha256": _sha256(mask_path),
                    "positive_pixels": positive_pixels,
                    "class_ids": sorted(int(value) for value in np.unique(mask) if value),
                    "unchanged_from_preannotation": bool(np.array_equal(mask, original)),
                }
            )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    validation = validate_manifest(
        load_manifest(output_manifest, require_commercial=True), NUM_CLASSES
    )
    if not validation["valid"]:
        raise ValueError(f"imported manifest failed validation: {validation['errors']}")
    report = {
        "status": "human_corrected_dense_annotations_imported",
        "batch_manifest": str(batch_path.resolve()),
        "batch_manifest_sha256": _sha256(batch_path),
        "corrected_annotations": str(corrected_path.resolve()),
        "corrected_annotations_sha256": _sha256(corrected_path),
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": batch["source_manifest_sha256"],
        "output_manifest": str(output_manifest.resolve()),
        "output_manifest_sha256": _sha256(output_manifest),
        "annotator_id": annotator_id,
        "reviewer_id": reviewer_id,
        "imported_masks": len(imported),
        "unchanged_from_preannotation": unchanged,
        "validation": validation,
        "samples": imported,
    }
    report_path = Path(str(output_manifest) + ".dense-import.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
