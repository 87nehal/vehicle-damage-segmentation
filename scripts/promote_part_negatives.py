"""Build a clean-negative fine-tuning manifest from annotated car-part views.

The HITL part subset contains empty damage masks plus an exterior mask. It was
previously kept damage-unsupervised because the original project had not
verified that every crop was damage-free. This opt-in promotion is intended
for an explicitly experimental run: it records the source rows, assigns a
small clean validation/calibration holdout, and never changes the source
manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _dhash(path: Path) -> int:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    bits = (gray[:, 1:] > gray[:, :-1]).reshape(-1)
    return sum(int(value) << index for index, value in enumerate(bits))

def _rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build(
    base_manifest: Path,
    parts_manifest: Path,
    output_manifest: Path,
    *,
    validation_count: int,
    calibration_count: int,
) -> dict[str, int]:
    raw_base = _rows(base_manifest)
    # combine_manifests.py has already merged exact auxiliary rows that match
    # damage-training images and removed auxiliary copies of held-out images.
    # Promote only the remaining unique auxiliary rows; re-adding all 998
    # source rows would leak duplicate damage images across splits.
    base = [row for row in raw_base if row.get("damage_supervised", True) is True]
    parts = [row for row in raw_base if row.get("damage_supervised", True) is False]
    if validation_count < 0 or calibration_count < 0:
        raise ValueError("holdout counts must be non-negative")
    if validation_count + calibration_count >= len(parts):
        raise ValueError("clean holdouts must leave part images for training")

    # Keep perceptual near-duplicates in one split. The manifest validator uses
    # the same dHash distance, so this makes the generated holdouts auditable.
    fingerprints = [_dhash((base_manifest.parent / row["image"]).resolve()) for row in parts]
    base_fingerprints = [
        (_dhash((base_manifest.parent / row["image"]).resolve()), row["split"])
        for row in base
    ]
    split_names = [
        "validation" if index < validation_count else
        "calibration" if index < validation_count + calibration_count else "train"
        for index in range(len(parts))
    ]
    for index, fingerprint in enumerate(fingerprints):
        if any(
            split != "train" and (fingerprint ^ other).bit_count() <= 2
            for other, split in base_fingerprints
        ):
            split_names[index] = "train"
    for left in range(len(parts)):
        for right in range(left + 1, len(parts)):
            if (fingerprints[left] ^ fingerprints[right]).bit_count() > 2:
                continue
            if split_names[left] == split_names[right]:
                continue
            # Keep the pair together. If the earlier row is already in train,
            # move it to the later row's holdout; otherwise move the later row
            # to the earlier row's split.
            if split_names[left] == "train":
                split_names[left] = split_names[right]
            else:
                split_names[right] = split_names[left]

    promoted: list[dict[str, object]] = []
    for index, source in enumerate(parts):
        row = dict(source)
        row["damage_supervised"] = True
        row["group_id"] = f"clean_part_{index:04d}"
        row["tags"] = sorted(
            set(str(tag) for tag in row.get("tags", []))
            | {"clean_negative_experimental", "part_annotation"}
        )
        row["split"] = split_names[index]
        promoted.append(row)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in [*base, *promoted]),
        encoding="utf-8",
    )
    return {
        "base_rows": len(base),
        "promoted_rows": len(promoted),
        "promoted_train": sum(row["split"] == "train" for row in promoted),
        "promoted_validation": sum(
            row["split"] == "validation" for row in promoted
        ),
        "promoted_calibration": sum(
            row["split"] == "calibration" for row in promoted
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--parts-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--calibration-count", type=int, default=100)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.base_manifest,
                args.parts_manifest,
                args.output_manifest,
                validation_count=args.validation_count,
                calibration_count=args.calibration_count,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
