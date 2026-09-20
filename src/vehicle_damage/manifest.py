from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    split: str
    group_id: str
    source: str
    license_id: str
    commercial_use: bool
    tags: tuple[str, ...] = ()
    exterior_mask: Path | None = None
    hard_negative_mask: Path | None = None
    damage_supervised: bool = True


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_manifest(path: str | Path, *, require_commercial: bool = True) -> list[Sample]:
    manifest = Path(path)
    root = manifest.parent
    samples: list[Sample] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"image", "mask", "split", "group_id", "source", "license_id", "commercial_use"}
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{manifest}:{line_number}: missing {sorted(missing)}")
        if require_commercial and row["commercial_use"] is not True:
            raise ValueError(
                f"{manifest}:{line_number}: commercial_use must be explicitly true; "
                "unknown or research-only data is rejected"
            )
        if not str(row["license_id"]).strip():
            raise ValueError(f"{manifest}:{line_number}: license_id cannot be empty")
        damage_supervised = row.get("damage_supervised", True)
        if not isinstance(damage_supervised, bool):
            raise ValueError(f"{manifest}:{line_number}: damage_supervised must be boolean")
        samples.append(
            Sample(
                image=_resolve(root, row["image"]),  # type: ignore[arg-type]
                mask=_resolve(root, row["mask"]),  # type: ignore[arg-type]
                split=str(row["split"]),
                group_id=str(row["group_id"]),
                source=str(row["source"]),
                license_id=str(row["license_id"]),
                commercial_use=bool(row["commercial_use"]),
                tags=tuple(str(x) for x in row.get("tags", [])),
                exterior_mask=_resolve(root, row.get("exterior_mask")),
                hard_negative_mask=_resolve(root, row.get("hard_negative_mask")),
                damage_supervised=damage_supervised,
            )
        )
    if not samples:
        raise ValueError(f"manifest is empty: {manifest}")
    return samples


def validate_manifest(samples: Iterable[Sample], num_classes: int, *, check_files: bool = True) -> dict:
    rows = list(samples)
    errors: list[str] = []
    split_groups: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    positive = 0
    negative = 0
    damage_unsupervised = 0
    exterior_supervised = 0
    hard_negative_supervised = 0
    hard_negative_pixels = 0
    hashes: dict[str, tuple[str, str]] = {}
    perceptual_hashes: list[tuple[int, str, str]] = []
    class_pixel_counts = [0] * num_classes
    class_image_counts = [0] * num_classes

    for sample in rows:
        split_counts[sample.split] = split_counts.get(sample.split, 0) + 1
        split_groups.setdefault(sample.split, set()).add(sample.group_id)
        for tag in sample.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if not sample.damage_supervised:
            damage_unsupervised += 1
            if sample.exterior_mask is None:
                errors.append(
                    f"damage-unsupervised sample requires an exterior_mask: {sample.image}"
                )
        if sample.exterior_mask is not None:
            exterior_supervised += 1
        for path in (sample.image, sample.mask, sample.exterior_mask, sample.hard_negative_mask):
            if path is not None and check_files and not path.is_file():
                errors.append(f"missing file: {path}")
        if not check_files or not sample.image.is_file() or not sample.mask.is_file():
            continue
        with Image.open(sample.mask) as mask_image:
            mask = np.asarray(mask_image)
        if mask.ndim != 2:
            errors.append(f"mask must be single-channel indexed PNG: {sample.mask}")
        elif mask.size:
            labels = set(np.unique(mask).tolist())
            bad = sorted(x for x in labels if x < 0 or x >= num_classes)
            if bad:
                errors.append(f"out-of-range mask ids {bad}: {sample.mask}")
            if sample.damage_supervised:
                if any(x > 0 for x in labels):
                    positive += 1
                else:
                    negative += 1
                for class_id in range(num_classes):
                    count = int((mask == class_id).sum())
                    class_pixel_counts[class_id] += count
                    class_image_counts[class_id] += int(count > 0)
        with Image.open(sample.image) as image:
            gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
            bits = (gray[:, 1:] > gray[:, :-1]).reshape(-1)
            dhash = sum(int(value) << index for index, value in enumerate(bits))
            if image.size != (mask.shape[1], mask.shape[0]):
                errors.append(f"image/mask size mismatch: {sample.image} / {sample.mask}")
            expected_size = image.size
        for name, optional_path in (
            ("exterior", sample.exterior_mask),
            ("hard-negative", sample.hard_negative_mask),
        ):
            if optional_path is None or not optional_path.is_file():
                continue
            with Image.open(optional_path) as optional_image:
                optional = np.asarray(optional_image.convert("L"))
                optional_size = optional_image.size
            if optional_size != expected_size:
                errors.append(
                    f"{name} mask size mismatch: {optional_path} / {sample.image}"
                )
            if name == "hard-negative":
                hard_negative_supervised += 1
                hard_negative_pixels += int((optional > 0).sum())
                if optional.shape == mask.shape and ((optional > 0) & (mask > 0)).any():
                    errors.append(f"hard-negative mask overlaps damage: {optional_path}")
        for other_hash, other_split, other_path in perceptual_hashes:
            if other_split != sample.split and (dhash ^ other_hash).bit_count() <= 2:
                errors.append(f"near-duplicate may cross splits: {other_path} and {sample.image}")
                break
        perceptual_hashes.append((dhash, sample.split, str(sample.image)))
        digest = hashlib.sha256(sample.image.read_bytes()).hexdigest()
        if digest in hashes:
            other_split, other_path = hashes[digest]
            if other_split != sample.split:
                errors.append(f"exact duplicate crosses splits: {other_path} and {sample.image}")
        else:
            hashes[digest] = (sample.split, str(sample.image))

    splits = sorted(split_groups)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            overlap = split_groups[left] & split_groups[right]
            if overlap:
                errors.append(
                    f"group leakage between {left} and {right}: "
                    + ", ".join(sorted(overlap)[:10])
                )
    return {
        "valid": not errors,
        "errors": errors,
        "samples": len(rows),
        "positive_images": positive,
        "negative_images": negative,
        "damage_unsupervised_images": damage_unsupervised,
        "exterior_supervised_images": exterior_supervised,
        "hard_negative_supervised_images": hard_negative_supervised,
        "hard_negative_pixels": hard_negative_pixels,
        "split_counts": split_counts,
        "tag_counts": tag_counts,
        "class_pixel_counts": class_pixel_counts,
        "class_image_counts": class_image_counts,
    }
