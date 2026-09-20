import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from vehicle_damage.annotation_export import (
    PASCAL_COLORS,
    safe_annotation_name,
    write_cvat_segmentation_batch,
)
from vehicle_damage.annotation_import import decode_cvat_class_mask
from vehicle_damage.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[1]
CLASSES = (
    "background",
    "dent",
    "scratch",
    "crack_or_breakage",
    "paint_damage",
    "deformation_or_detachment",
)


def test_decode_cvat_class_mask_accepts_indexed_and_rgb_masks():
    indexed = np.array([[0, 1, 5]], dtype=np.uint8)
    assert np.array_equal(decode_cvat_class_mask(indexed, 6), indexed)

    rgb = np.asarray(
        [[PASCAL_COLORS[0], PASCAL_COLORS[2], PASCAL_COLORS[4]]],
        dtype=np.uint8,
    )
    assert np.array_equal(
        decode_cvat_class_mask(rgb, 6),
        np.array([[0, 2, 4]], dtype=np.uint8),
    )


def test_decode_cvat_class_mask_rejects_unknown_values():
    with pytest.raises(ValueError, match="invalid class"):
        decode_cvat_class_mask(np.array([[6]], dtype=np.uint8), 6)
    with pytest.raises(ValueError, match="unknown RGB"):
        decode_cvat_class_mask(np.array([[[1, 2, 3]]], dtype=np.uint8), 6)


def test_corrected_cvat_masks_replace_only_unsupervised_manifest_rows(tmp_path):
    image = tmp_path / "car.jpg"
    unknown = tmp_path / "unknown.png"
    exterior = tmp_path / "exterior.png"
    Image.new("RGB", (12, 8), "gray").save(image)
    Image.new("L", (12, 8), 0).save(unknown)
    Image.new("L", (12, 8), 255).save(exterior)
    source_manifest = tmp_path / "manifest.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "image": image.name,
                "mask": unknown.name,
                "exterior_mask": exterior.name,
                "split": "train",
                "group_id": "vehicle-1",
                "source": "cc0-test",
                "license_id": "CC0-1.0",
                "commercial_use": True,
                "damage_supervised": False,
                "tags": ["damage_unverified", "exterior_only"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    archive_name = safe_annotation_name("vehicle-1", digest, image.suffix)
    class_mask = np.zeros((8, 12), dtype=np.uint8)
    class_mask[2:6, 4:9] = 1
    batch_dir = tmp_path / "batch"
    batch_manifest = batch_dir / "BATCH_MANIFEST.json"
    annotations = batch_dir / "preannotations.zip"
    write_cvat_segmentation_batch(
        [
            {
                "group_id": "vehicle-1",
                "source_image": str(image),
                "archive_name": archive_name,
                "class_mask": class_mask,
            }
        ],
        classes=CLASSES,
        images_zip=batch_dir / "images.zip",
        annotations_zip=annotations,
        batch_manifest=batch_manifest,
        provenance={
            "source_manifest": str(source_manifest.resolve()),
            "source_manifest_sha256": hashlib.sha256(
                source_manifest.read_bytes()
            ).hexdigest(),
        },
    )
    output_manifest = tmp_path / "reviewed" / "manifest.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "import_cvat_annotation_batch.py"),
            "--batch-manifest",
            str(batch_manifest),
            "--corrected-annotations",
            str(annotations),
            "--output-manifest",
            str(output_manifest),
            "--annotator-id",
            "dense-annotator",
            "--reviewer-id",
            "independent-reviewer",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    imported = load_manifest(output_manifest)
    assert imported[0].damage_supervised
    assert {"dense_annotated", "independently_reviewed_mask"} <= set(
        imported[0].tags
    )
    assert int(np.asarray(Image.open(imported[0].mask)).sum()) == 20
    report = json.loads(
        Path(str(output_manifest) + ".dense-import.json").read_text(encoding="utf-8")
    )
    assert report["imported_masks"] == 1
    assert report["unchanged_from_preannotation"] == 1
