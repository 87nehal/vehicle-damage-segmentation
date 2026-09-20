"""Prepare the CC0 car-part subset as exterior-only supervision.

These images have car-part polygons but no exhaustive damage review. The
generated rows therefore set ``damage_supervised`` to false: they train the
vehicle-exterior head and must never be counted as clean damage negatives.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw

from prepare_hitl_cc0 import (
    DATASET_URL,
    LICENSE_ID,
    RemoteZip,
    _meta_classes,
    _paired_image,
    _resize_pair,
)


PART_CLASSES = {
    "windshield", "back-windshield", "front-window", "back-window",
    "front-door", "back-door", "front-wheel", "back-wheel", "front-bumper",
    "back-bumper", "headlight", "tail-light", "hood", "trunk",
    "license-plate", "mirror", "roof", "grille", "rocker-panel",
    "quarter-panel", "fender",
}


def _find_parts_root(remote: RemoteZip) -> tuple[str, dict]:
    diagnostics = []
    for name in sorted(x for x in remote.entries if x.endswith("/meta.json")):
        meta = json.loads(remote.read(name))
        classes = _meta_classes(meta)
        overlap = classes & PART_CLASSES
        diagnostics.append((name, sorted(classes), sorted(overlap)))
        if len(overlap) >= 10:
            return name.rsplit("/", 1)[0], meta
    raise RuntimeError(f"could not identify car-part subset: {diagnostics}")


def _draw_exterior(annotation: dict, size: tuple[int, int]) -> Image.Image:
    """Draw the union of annotated visible exterior parts."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    unknown: set[str] = set()
    for obj in annotation.get("objects", []):
        title = str(obj.get("classTitle", "")).strip().lower()
        if title not in PART_CLASSES:
            unknown.add(title)
            continue
        points = obj.get("points", {}).get("exterior", [])
        if len(points) >= 3:
            draw.polygon([(float(x), float(y)) for x, y in points], fill=255)
    if unknown:
        raise ValueError(f"unmapped car-part labels: {sorted(unknown)}")
    return mask


def prepare(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    images_dir = output / "images"
    masks_dir = output / "masks"
    exterior_dir = output / "exterior_masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    exterior_dir.mkdir(parents=True, exist_ok=True)

    remote = RemoteZip(DATASET_URL)
    root, meta = _find_parts_root(remote)
    annotations = sorted(
        name for name in remote.entries
        if name.startswith(f"{root}/File1/ann/") and name.endswith(".json")
    )
    available = len(annotations)
    rng = random.Random(args.seed)
    rng.shuffle(annotations)
    if args.limit:
        annotations = annotations[: args.limit]
    print(json.dumps({
        "parts_root": root,
        "available": available,
        "selected": len(annotations),
        "classes": sorted(_meta_classes(meta)),
    }))

    def convert(index: int, annotation_name: str) -> tuple[int, dict]:
        image_name = _paired_image(remote.entries, root, annotation_name)
        annotation = json.loads(remote.read(annotation_name))
        image = Image.open(io.BytesIO(remote.read(image_name))).convert("RGB")
        exterior = _draw_exterior(annotation, image.size)
        image, exterior = _resize_pair(image, exterior, args.max_side)
        damage = Image.new("L", image.size, 0)
        stem = f"hitl_parts_{index:05d}"
        image_path = images_dir / f"{stem}.jpg"
        mask_path = masks_dir / f"{stem}.png"
        exterior_path = exterior_dir / f"{stem}.png"
        image.save(image_path, format="JPEG", quality=92, optimize=True)
        damage.save(mask_path, format="PNG", optimize=True)
        exterior.save(exterior_path, format="PNG", optimize=True)
        return index, {
            "image": image_path.relative_to(output).as_posix(),
            "mask": mask_path.relative_to(output).as_posix(),
            "exterior_mask": exterior_path.relative_to(output).as_posix(),
            "damage_supervised": False,
            "split": "train",
            "group_id": stem,
            "source": "Humans in the Loop car parts and car damages",
            "source_url": "https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/",
            "license_id": LICENSE_ID,
            "commercial_use": True,
            "tags": ["public_cc0", "exterior_only", "damage_unverified"],
        }

    rows_by_index: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(convert, index, annotation_name): annotation_name
            for index, annotation_name in enumerate(annotations)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            index, row = future.result()
            rows_by_index[index] = row
            print(json.dumps({
                "prepared": completed,
                "total": len(annotations),
                "image": futures[future],
            }))

    rows = [rows_by_index[index] for index in range(len(annotations))]
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (output / "PROVENANCE.json").write_text(
        json.dumps({
            "dataset": "Car Parts and Car Damages",
            "publisher": "Humans in the Loop",
            "license": "CC0 1.0",
            "source_url": "https://humansintheloop.org/resources/datasets/car-parts-and-car-damages-dataset/",
            "download_url": DATASET_URL,
            "archive_bytes": remote.length,
            "parts_root": root,
            "prepared_images": len(rows),
            "damage_labels": "not reviewed; excluded from damage loss",
        }, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/hitl_parts_cc0")
    parser.add_argument("--limit", type=int, default=0, help="0 fetches every car-part image")
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--workers", type=int, default=8)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
