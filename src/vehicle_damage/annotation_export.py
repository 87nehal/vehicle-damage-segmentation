from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


PASCAL_COLORS = (
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
)


def safe_annotation_name(group_id: str, image_sha256: str, suffix: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", group_id).strip("._")
    if not slug:
        slug = "vehicle"
    normalized_suffix = suffix.lower() if suffix else ".jpg"
    return f"{slug}_{image_sha256[:12]}{normalized_suffix}"


def instance_mask(class_mask: np.ndarray) -> np.ndarray:
    """Assign one 8-connected object id per class-specific component."""
    mask = np.asarray(class_mask)
    if mask.ndim != 2:
        raise ValueError("class mask must be a 2D indexed array")
    result = np.zeros(mask.shape, dtype=np.uint8)
    next_id = 1
    structure = np.ones((3, 3), dtype=np.uint8)
    for class_id in sorted(int(value) for value in np.unique(mask) if value > 0):
        labels, count = ndimage.label(mask == class_id, structure=structure)
        if next_id + count > 255:
            raise ValueError("class mask has too many components for an 8-bit object mask")
        for component_id in range(1, count + 1):
            result[labels == component_id] = next_id
            next_id += 1
    return result


def _png_bytes(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").save(
        buffer, format="PNG", optimize=True
    )
    return buffer.getvalue()


def write_cvat_segmentation_batch(
    records: list[dict],
    *,
    classes: list[str] | tuple[str, ...],
    images_zip: Path,
    annotations_zip: Path,
    batch_manifest: Path,
    provenance: dict,
) -> dict:
    """Write CVAT image and Segmentation Mask 1.1 archives plus provenance."""
    if not records:
        raise ValueError("annotation batch requires at least one record")
    if len(classes) != len(PASCAL_COLORS):
        raise ValueError("classes must match the six-entry vehicle-damage palette")
    names = [str(record["archive_name"]) for record in records]
    if len(set(names)) != len(names):
        raise ValueError("archive image names must be unique")

    for path in (images_zip, annotations_zip, batch_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    labelmap = "".join(
        f"{name}:{red},{green},{blue}::\n"
        for name, (red, green, blue) in zip(classes, PASCAL_COLORS)
    )
    stems = [Path(name).stem for name in names]

    rendered_records = []
    with zipfile.ZipFile(images_zip, "w", compression=zipfile.ZIP_DEFLATED) as images:
        with zipfile.ZipFile(
            annotations_zip, "w", compression=zipfile.ZIP_DEFLATED
        ) as annotations:
            annotations.writestr("labelmap.txt", labelmap)
            annotations.writestr(
                "ImageSets/Segmentation/default.txt", "".join(f"{stem}\n" for stem in stems)
            )
            for record in records:
                source = Path(record["source_image"])
                archive_name = str(record["archive_name"])
                class_mask = np.asarray(record["class_mask"], dtype=np.uint8)
                with Image.open(source) as image:
                    if class_mask.shape != (image.height, image.width):
                        raise ValueError(
                            f"mask size does not match source image: {source}"
                        )
                image_bytes = source.read_bytes()
                class_bytes = _png_bytes(class_mask)
                object_bytes = _png_bytes(instance_mask(class_mask))
                stem = Path(archive_name).stem
                images.writestr(archive_name, image_bytes)
                annotations.writestr(
                    f"SegmentationClass/{stem}.png", class_bytes
                )
                annotations.writestr(
                    f"SegmentationObject/{stem}.png", object_bytes
                )
                rendered_records.append(
                    {
                        key: value
                        for key, value in record.items()
                        if key != "class_mask"
                    }
                    | {
                        "source_image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                        "preannotation_mask_sha256": hashlib.sha256(
                            class_bytes
                        ).hexdigest(),
                        "preannotation_positive_pixels": int((class_mask > 0).sum()),
                    }
                )

    manifest = {
        **provenance,
        "status": "model_preannotations_require_human_correction",
        "warning": (
            "These masks are model suggestions, not ground truth. Import into CVAT "
            "as Segmentation Mask 1.1, correct every image, and independently review "
            "the result before training."
        ),
        "classes": list(classes),
        "palette_rgb": [list(color) for color in PASCAL_COLORS],
        "images_zip": str(images_zip.resolve()),
        "annotations_zip": str(annotations_zip.resolve()),
        "samples": rendered_records,
    }
    batch_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
