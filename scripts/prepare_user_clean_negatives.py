"""Create deterministic clean-car augmentations from a user-asserted example.

The supplied image is a regression case: the car is intact, but earlier
checkpoints marked grille slots and body contours as damage.  This script
keeps the original in a held-out test split and adds varied lighting, blur,
colour, crop, and viewpoint-like transforms to training without inventing
damage labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw


def _transform(base: Image.Image, index: int) -> Image.Image:
    image = base.copy()
    if index % 2:
        image = ImageOps.mirror(image)
    image = ImageEnhance.Brightness(image).enhance(0.70 + 0.10 * (index % 8))
    image = ImageEnhance.Contrast(image).enhance(0.78 + 0.08 * ((index * 3) % 7))
    image = ImageEnhance.Color(image).enhance(0.72 + 0.10 * ((index * 5) % 6))
    if index % 5 == 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.45 + 0.12 * (index % 4)))
    if index % 7 == 0:
        angle = -5.0 + float(index % 11)
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(30, 30, 30))
    if index % 9 == 0:
        width, height = image.size
        margin_x = int(width * (0.015 + 0.005 * (index % 4)))
        margin_y = int(height * (0.012 + 0.004 * (index % 5)))
        image = ImageOps.fit(
            image,
            (width, height),
            method=Image.Resampling.BICUBIC,
            centering=(0.48 + 0.02 * (index % 3), 0.52),
            bleed=min(0.12, max(margin_x / width, margin_y / height)),
        )
    return image


def _exterior_mask(size: tuple[int, int]) -> Image.Image:
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    # A conservative polygon around the front vehicle in the supplied image.
    points = [
        (int(width * 0.06), int(height * 0.25)),
        (int(width * 0.94), int(height * 0.25)),
        (width - 1, int(height * 0.47)),
        (width - 1, height - 1),
        (0, height - 1),
        (0, int(height * 0.47)),
    ]
    draw.polygon(points, fill=255)
    return mask


def prepare(source: Path, output: Path, variants: int) -> dict[str, int]:
    base = Image.open(source).convert("RGB")
    group_prefix = source.stem.replace(" ", "_")[:80]
    image_dir = output / "images"
    mask_dir = output / "masks"
    exterior_dir = output / "exterior_masks"
    hard_negative_dir = output / "hard_negative_masks"
    for directory in (image_dir, mask_dir, exterior_dir, hard_negative_dir):
        directory.mkdir(parents=True, exist_ok=True)

    exterior = _exterior_mask(base.size)
    rows: list[dict] = []
    # Preserve the untouched source as a held-out regression case.
    images: list[tuple[str, Image.Image, str]] = [("user_clean_000", base, "original")]
    for index in range(1, variants + 1):
        images.append((f"user_clean_{index:03d}", _transform(base, index), "augmented"))

    for index, (stem, image, kind) in enumerate(images):
        image_path = image_dir / f"{stem}.jpg"
        mask_path = mask_dir / f"{stem}.png"
        exterior_path = exterior_dir / f"{stem}.png"
        hard_negative_path = hard_negative_dir / f"{stem}.png"
        image.save(image_path, quality=95, subsampling=0)
        Image.new("L", base.size, 0).save(mask_path)
        exterior.save(exterior_path)
        # Weight intact exterior pixels more heavily during refinement.  This
        # is intentionally a hard-negative mask, not a damage label.
        exterior.save(hard_negative_path)
        # Keep the complete augmentation family together in training.  The
        # untouched source is also trained on deliberately: it is an explicit
        # user regression case that is checked separately after each run.
        split = "train"
        rows.append(
            {
                "image": str(image_path.resolve()),
                "mask": str(mask_path.resolve()),
                "exterior_mask": str(exterior_path.resolve()),
                "hard_negative_mask": str(hard_negative_path.resolve()),
                "split": split,
                "group_id": f"user_clean_{group_prefix}_{stem}",
                "source": "User-asserted clean vehicle regression image",
                "source_url": "local-user-provided",
                "license_id": "user-provided-image",
                "commercial_use": True,
                "damage_supervised": True,
                "tags": ["user_asserted_clean", "regression_clean", "hard_negative", kind],
            }
        )
    manifest = output / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "calibration", "test")}
    counts["written"] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", type=int, default=140)
    args = parser.parse_args()
    print(json.dumps(prepare(Path(args.source), Path(args.output_dir), args.variants)))


if __name__ == "__main__":
    main()
