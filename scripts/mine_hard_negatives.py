"""Mine high-confidence false-positive pixels from supervised training rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vehicle_damage.calibration import Calibration
from vehicle_damage.hard_negative import combine_hard_negative_masks, select_hard_negative_mask
from vehicle_damage.inference import predict_multiscale_probabilities
from vehicle_damage.manifest import load_manifest
from vehicle_damage.model import load_checkpoint


PATH_FIELDS = ("image", "mask", "exterior_mask", "hard_negative_mask")


def _relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def mine(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    output = Path(args.output_manifest).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mask_dir = output.parent / "hard_negative_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = load_manifest(manifest_path, require_commercial=True)
    if len(raw_rows) != len(samples):
        raise RuntimeError("manifest parsing produced inconsistent row counts")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    calibration = Calibration.load(args.calibration)
    exterior_floor = calibration.exterior_floor if args.exterior_floor is None else args.exterior_floor
    tile_size = calibration.tile_size if args.tile_size is None else args.tile_size
    overlap = calibration.overlap if args.overlap is None else args.overlap
    horizontal_flip_tta = calibration.horizontal_flip_tta
    if exterior_floor != calibration.exterior_floor:
        raise ValueError(
            "exterior floor must match the frozen calibration; "
            f"expected {calibration.exterior_floor}, got {exterior_floor}"
        )
    if tile_size != calibration.tile_size or overlap != calibration.overlap:
        raise ValueError("tile size and overlap must match the frozen calibration")
    if args.no_tta and horizontal_flip_tta:
        raise ValueError("--no-tta conflicts with a threshold calibrated using flip TTA")

    output_rows: list[dict] = []
    mined_images = 0
    mined_pixels = 0
    newly_mined_pixels = 0
    retained_existing_pixels = 0
    eligible_pixels = 0
    selected_samples = [
        sample for sample in samples
        if sample.split == args.split and sample.damage_supervised
    ]
    total = len(selected_samples)
    completed = 0
    for row, sample in zip(raw_rows, samples):
        normalized = dict(row)
        resolved_paths = {
            "image": sample.image,
            "mask": sample.mask,
            "exterior_mask": sample.exterior_mask,
            "hard_negative_mask": sample.hard_negative_mask,
        }
        for field in PATH_FIELDS:
            path = resolved_paths[field]
            if path is not None:
                normalized[field] = _relative(path, output.parent)

        if sample.split == args.split and sample.damage_supervised:
            with Image.open(sample.image) as source:
                image = source.convert("RGB")
            with Image.open(sample.mask) as source:
                target = np.asarray(source.convert("L"))
            damage, exterior = predict_multiscale_probabilities(
                model,
                image,
                device=device,
                scale_factors=calibration.triage_scales,
                tile_size=tile_size,
                overlap=overlap,
                horizontal_flip_tta=horizontal_flip_tta,
                mixed_precision=calibration.mixed_precision,
            )
            score = (
                (1.0 - damage[0])
                * (exterior_floor + (1.0 - exterior_floor) * exterior)
            ).numpy()
            hard = select_hard_negative_mask(
                score,
                target,
                minimum_score=args.minimum_score,
                maximum_fraction=args.maximum_fraction,
                guard_radius=args.guard_radius,
            )
            new_count = int((hard > 0).sum())
            existing_count = 0
            if args.merge_existing and sample.hard_negative_mask is not None:
                with Image.open(sample.hard_negative_mask) as source:
                    existing = np.asarray(source.convert("L"))
                if existing.shape != target.shape:
                    raise ValueError(
                        f"hard-negative mask size mismatch: {sample.hard_negative_mask}"
                    )
                existing_count = int((existing > 0).sum())
                hard = combine_hard_negative_masks(hard, existing)
            hard_path = mask_dir / f"{sample.group_id}.png"
            Image.fromarray(hard, mode="L").save(hard_path, format="PNG", optimize=True)
            normalized["hard_negative_mask"] = _relative(hard_path, output.parent)
            normalized["tags"] = sorted(set(normalized.get("tags", [])) | {"mined_hard_negative"})
            count = int((hard > 0).sum())
            mined_images += int(count > 0)
            mined_pixels += count
            newly_mined_pixels += new_count
            retained_existing_pixels += existing_count
            eligible_pixels += int((target == 0).sum())
            completed += 1
            print(json.dumps({
                "mined": completed,
                "total": total,
                "image": str(sample.image),
                "hard_negative_pixels": count,
            }), flush=True)
        output_rows.append(normalized)

    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")

    checkpoint_path = Path(args.checkpoint).resolve()
    provenance = {
        "source_manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "calibration": str(Path(args.calibration).resolve()),
        "split": args.split,
        "minimum_score": args.minimum_score,
        "maximum_fraction": args.maximum_fraction,
        "guard_radius": args.guard_radius,
        "exterior_floor": exterior_floor,
        "tile_size": tile_size,
        "overlap": overlap,
        "horizontal_flip_tta": horizontal_flip_tta,
        "mixed_precision": calibration.mixed_precision,
        "triage_scales": calibration.triage_scales,
        "mined_images": mined_images,
        "mined_pixels": mined_pixels,
        "newly_selected_pixels_before_merge": newly_mined_pixels,
        "existing_pixels_before_merge": retained_existing_pixels,
        "merge_existing": args.merge_existing,
        # This denominator is all annotated background before the polygon guard
        # band is removed.  Name it precisely so the provenance does not imply
        # that every counted pixel was eligible for mining.
        "background_pixels_before_guard": eligible_pixels,
        "mined_fraction_of_background": (
            mined_pixels / eligible_pixels if eligible_pixels else 0.0
        ),
    }
    (output.parent / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--minimum-score", type=float, required=True)
    parser.add_argument("--maximum-fraction", type=float, default=0.10)
    parser.add_argument("--guard-radius", type=int, default=5)
    parser.add_argument("--exterior-floor", type=float)
    parser.add_argument("--device")
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="union newly mined pixels with any existing hard-negative mask",
    )
    mine(parser.parse_args())


if __name__ == "__main__":
    main()
