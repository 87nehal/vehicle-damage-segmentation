import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from vehicle_damage.annotation_export import (
    instance_mask,
    safe_annotation_name,
    write_cvat_segmentation_batch,
)


CLASSES = (
    "background",
    "dent",
    "scratch",
    "crack_or_breakage",
    "paint_damage",
    "deformation_or_detachment",
)


def test_instance_mask_separates_classes_and_components():
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    mask[3:5, 0:2] = 1
    mask[0:2, 4:6] = 2
    objects = instance_mask(mask)
    assert set(np.unique(objects)) == {0, 1, 2, 3}
    assert objects[0, 0] != objects[3, 0]
    assert objects[0, 0] != objects[0, 4]


def test_cvat_segmentation_batch_contains_images_masks_and_provenance(tmp_path):
    image = tmp_path / "car.jpg"
    Image.new("RGB", (7, 5), "gray").save(image)
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[1:4, 2:5] = 1
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    name = safe_annotation_name("vehicle / one", digest, image.suffix)
    images_zip = tmp_path / "images.zip"
    annotations_zip = tmp_path / "annotations.zip"
    manifest_path = tmp_path / "BATCH_MANIFEST.json"
    rendered = write_cvat_segmentation_batch(
        [
            {
                "group_id": "vehicle / one",
                "source_image": str(image),
                "archive_name": name,
                "class_mask": mask,
            }
        ],
        classes=CLASSES,
        images_zip=images_zip,
        annotations_zip=annotations_zip,
        batch_manifest=manifest_path,
        provenance={"checkpoint_sha256": "abc"},
    )
    assert rendered["status"] == "model_preannotations_require_human_correction"
    assert rendered["samples"][0]["preannotation_positive_pixels"] == 9
    with zipfile.ZipFile(images_zip) as archive:
        assert archive.namelist() == [name]
    stem = Path(name).stem
    with zipfile.ZipFile(annotations_zip) as archive:
        assert "labelmap.txt" in archive.namelist()
        assert f"SegmentationClass/{stem}.png" in archive.namelist()
        assert f"SegmentationObject/{stem}.png" in archive.namelist()
        assert archive.read("ImageSets/Segmentation/default.txt").decode().strip() == stem
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["classes"] == list(CLASSES)
    assert "not ground truth" in loaded["warning"].lower()
