"""Combine manifests while preserving their path semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


PATH_FIELDS = ("image", "mask", "exterior_mask", "hard_negative_mask")


def _image_path(row: dict, output: Path) -> Path:
    return (output.parent / row["image"]).resolve()


def _fingerprints(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    bits = (gray[:, 1:] > gray[:, :-1]).reshape(-1)
    dhash = sum(int(value) << index for index, value in enumerate(bits))
    return digest, dhash


def _merge_auxiliary(rows: list[dict], output: Path) -> tuple[list[dict], dict[str, int]]:
    """Merge auxiliary labels without allowing held-out image leakage."""
    supervised = [row for row in rows if row.get("damage_supervised", True) is True]
    auxiliary = [row for row in rows if row.get("damage_supervised", True) is False]
    digest_to_supervised: dict[str, dict] = {}
    held_out_hashes: list[int] = []
    for row in supervised:
        digest, dhash = _fingerprints(_image_path(row, output))
        digest_to_supervised[digest] = row
        if row["split"] != "train":
            held_out_hashes.append(dhash)

    kept_auxiliary: list[dict] = []
    auxiliary_digests: set[str] = set()
    stats = {
        "merged_into_damage_train": 0,
        "dropped_exact_held_out": 0,
        "dropped_near_held_out": 0,
        "dropped_duplicate_auxiliary": 0,
        "kept_auxiliary": 0,
    }
    for row in auxiliary:
        digest, dhash = _fingerprints(_image_path(row, output))
        match = digest_to_supervised.get(digest)
        if match is not None:
            if match["split"] == "train":
                if match.get("exterior_mask") is None:
                    match["exterior_mask"] = row["exterior_mask"]
                match["tags"] = sorted(set(match.get("tags", [])) | {"exterior_supervised"})
                stats["merged_into_damage_train"] += 1
            else:
                stats["dropped_exact_held_out"] += 1
            continue
        if any((dhash ^ held_out).bit_count() <= 2 for held_out in held_out_hashes):
            stats["dropped_near_held_out"] += 1
            continue
        if digest in auxiliary_digests:
            stats["dropped_duplicate_auxiliary"] += 1
            continue
        auxiliary_digests.add(digest)
        kept_auxiliary.append(row)
    stats["kept_auxiliary"] = len(kept_auxiliary)
    return supervised + kept_auxiliary, stats


def combine(inputs: list[Path], output: Path, *, merge_auxiliary: bool = False) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    seen: set[Path] = set()
    for manifest in inputs:
        manifest = manifest.resolve()
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in PATH_FIELDS:
                value = row.get(field)
                if value is None:
                    continue
                source_path = Path(value)
                if not source_path.is_absolute():
                    source_path = (manifest.parent / source_path).resolve()
                # A manifest may combine sources stored on different Windows
                # volumes (for example a local training set on C: and COCO on
                # D:).  ``os.path.relpath`` raises in that case; preserve the
                # absolute path so the loader can still open the asset.
                try:
                    row[field] = Path(os.path.relpath(source_path, output.parent)).as_posix()
                except ValueError:
                    row[field] = str(source_path)
            image = (output.parent / row["image"]).resolve()
            if image in seen:
                raise ValueError(f"duplicate image at {manifest}:{line_number}: {image}")
            seen.add(image)
            rows.append(row)
    stats: dict[str, int] = {}
    if merge_auxiliary:
        rows, stats = _merge_auxiliary(rows, output)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({
        "output": str(output), "rows": len(rows), "inputs": len(inputs), **stats,
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--merge-auxiliary", action="store_true",
        help="merge exact exterior labels into training rows and protect held-out splits",
    )
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    combine(
        [Path(x) for x in args.inputs], Path(args.output),
        merge_auxiliary=args.merge_auxiliary,
    )


if __name__ == "__main__":
    main()
