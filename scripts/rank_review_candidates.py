"""Rank commercially permitted but damage-unverified images for human review.

The output is prioritization metadata, not a clean/damaged pseudo-label. Images
must still pass the repository's independent double-review workflow before
they can supervise damage absence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vehicle_damage.calibration import Calibration
from vehicle_damage.inference import (
    assess_quality,
    predict_profile_probabilities,
    probabilities_to_mask,
)
from vehicle_damage.manifest import load_manifest
from vehicle_damage.model import load_checkpoint
from vehicle_damage.review_priority import (
    rank_review_records,
    summarize_review_candidate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank damage-unverified images for independent human review"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="damage_unverified")
    parser.add_argument("--split")
    parser.add_argument("--uncertainty-margin", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    if not 0 < args.uncertainty_margin <= 1:
        parser.error("--uncertainty-margin must be in (0, 1]")

    checkpoint_path = Path(args.checkpoint)
    calibration_path = Path(args.calibration)
    manifest_path = Path(args.manifest)
    calibration = Calibration.load(calibration_path)
    model, metadata = load_checkpoint(str(checkpoint_path), args.device)
    model.eval()

    samples = [
        sample
        for sample in load_manifest(manifest_path, require_commercial=True)
        if args.tag in sample.tags
        and (args.split is None or sample.split == args.split)
    ]
    if args.max_samples:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("no manifest samples match the requested review selection")

    records: list[dict] = []
    with torch.inference_mode():
        for index, sample in enumerate(samples, 1):
            with Image.open(sample.image) as opened:
                image = opened.convert("RGB")
            (triage_damage, triage_exterior), (
                mask_damage,
                mask_exterior,
            ) = predict_profile_probabilities(
                model,
                image,
                device=args.device,
                triage_scales=calibration.triage_scales,
                segmentation_scales=calibration.effective_segmentation_scales,
                tile_size=calibration.tile_size,
                overlap=calibration.overlap,
                horizontal_flip_tta=calibration.horizontal_flip_tta,
                mixed_precision=calibration.mixed_precision,
            )
            triage_score = (
                (1.0 - triage_damage[0])
                * (
                    calibration.exterior_floor
                    + (1.0 - calibration.exterior_floor) * triage_exterior
                )
            ).numpy()
            precise_mask, _ = probabilities_to_mask(
                mask_damage,
                mask_exterior,
                calibration.effective_segmentation_threshold,
                calibration.exterior_floor,
                calibration.effective_segmentation_minimum_component_pixels,
                calibration.type_probability_multipliers,
            )
            quality = assess_quality(image)
            summary = summarize_review_candidate(
                triage_score,
                precise_mask,
                triage_threshold=calibration.any_damage_threshold,
                uncertainty_margin=args.uncertainty_margin,
                quality_review_reasons=quality.review_reasons,
            )
            records.append(
                {
                    "image_sha256": _sha256(sample.image),
                    "group_id": sample.group_id,
                    "image": str(sample.image.resolve()),
                    "damage_supervised": sample.damage_supervised,
                    "quality_review_reasons": list(quality.review_reasons),
                    **summary,
                }
            )
            if index % 25 == 0 or index == len(samples):
                print(f"scored {index}/{len(samples)}", flush=True)

    ranked = rank_review_records(records)
    report = {
        "status": "review_priority_only_not_ground_truth",
        "warning": (
            "No record is labeled clean or damaged. Independent double review "
            "and adjudication remain required before damage supervision."
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": metadata.get("epoch"),
        "calibration": str(calibration_path.resolve()),
        "calibration_sha256": _sha256(calibration_path),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "selection_tag": args.tag,
        "split": args.split,
        "uncertainty_margin": args.uncertainty_margin,
        "ranking_method": (
            "3*sqrt(precise_mask_fraction) + sqrt(triage_positive_fraction) + "
            "0.25*triage_q99 + 0.15*near_threshold_fraction + "
            "0.05*min(quality_reason_count,4)"
        ),
        "samples": ranked,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "samples": len(ranked)}, indent=2))


if __name__ == "__main__":
    main()
