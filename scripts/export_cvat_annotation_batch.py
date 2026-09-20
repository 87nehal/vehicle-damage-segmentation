"""Export selected model predictions as editable CVAT segmentation prelabels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from vehicle_damage.annotation_export import (
    safe_annotation_name,
    write_cvat_segmentation_batch,
)
from vehicle_damage.calibration import Calibration
from vehicle_damage.inference import predict_profile_probabilities, probabilities_to_mask
from vehicle_damage.manifest import load_manifest
from vehicle_damage.model import load_checkpoint


CLASSES = (
    "background",
    "dent",
    "scratch",
    "crack_or_breakage",
    "paint_damage",
    "deformation_or_detachment",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export editable CVAT Segmentation Mask 1.1 preannotations"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ranking", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="damage_unverified")
    parser.add_argument("--split", default="train")
    parser.add_argument("--ranking-order", choices=("highest", "lowest"), default="lowest")
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    checkpoint_path = Path(args.checkpoint)
    calibration_path = Path(args.calibration)
    manifest_path = Path(args.manifest)
    ranking_path = Path(args.ranking)
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    if ranking.get("status") != "review_priority_only_not_ground_truth":
        raise ValueError("ranking is not a review-priority-only artifact")
    if ranking.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("ranking source manifest hash does not match --manifest")
    if ranking.get("selection_tag") != args.tag:
        raise ValueError("ranking selection tag does not match --tag")

    ranked = {
        str(row["image_sha256"]): row for row in ranking.get("samples", [])
    }
    if len(ranked) != len(ranking.get("samples", [])):
        raise ValueError("ranking image hashes must be unique")
    candidates = []
    for sample in load_manifest(manifest_path, require_commercial=True):
        if args.tag not in sample.tags or sample.split != args.split:
            continue
        if sample.damage_supervised:
            raise ValueError("annotation candidates must remain damage-unsupervised")
        digest = _sha256(sample.image)
        row = ranked.get(digest)
        if row is None:
            raise ValueError(f"ranking is missing selected image: {sample.image}")
        candidates.append((int(row["rank"]), digest, sample, row))
    candidates.sort(
        key=lambda item: item[0],
        reverse=args.ranking_order == "lowest",
    )
    candidates = candidates[: args.limit]
    if not candidates:
        raise ValueError("no matching annotation candidates")

    calibration = Calibration.load(calibration_path)
    model, metadata = load_checkpoint(str(checkpoint_path), args.device)
    model.eval()
    records = []
    with torch.inference_mode():
        for index, (rank, digest, sample, ranking_row) in enumerate(candidates, 1):
            with Image.open(sample.image) as opened:
                image = opened.convert("RGB")
            (_, _), (mask_damage, mask_exterior) = predict_profile_probabilities(
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
            class_mask, _ = probabilities_to_mask(
                mask_damage,
                mask_exterior,
                calibration.effective_segmentation_threshold,
                calibration.exterior_floor,
                calibration.effective_segmentation_minimum_component_pixels,
                calibration.type_probability_multipliers,
            )
            records.append(
                {
                    "group_id": sample.group_id,
                    "source_image": str(sample.image.resolve()),
                    "archive_name": safe_annotation_name(
                        sample.group_id, digest, sample.image.suffix
                    ),
                    "source_split": sample.split,
                    "source_name": sample.source,
                    "source_license_id": sample.license_id,
                    "source_commercial_use": sample.commercial_use,
                    "source_tags": list(sample.tags),
                    "source_exterior_mask": (
                        None
                        if sample.exterior_mask is None
                        else str(sample.exterior_mask.resolve())
                    ),
                    "ranking_rank": rank,
                    "ranking_priority_score": ranking_row["priority_score"],
                    "class_mask": class_mask,
                }
            )
            print(f"preannotated {index}/{len(candidates)}", flush=True)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = write_cvat_segmentation_batch(
        records,
        classes=CLASSES,
        images_zip=output / "images.zip",
        annotations_zip=output / "preannotations-segmentation-mask-1.1.zip",
        batch_manifest=output / "BATCH_MANIFEST.json",
        provenance={
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_epoch": metadata.get("epoch"),
            "calibration": str(calibration_path.resolve()),
            "calibration_sha256": _sha256(calibration_path),
            "source_manifest": str(manifest_path.resolve()),
            "source_manifest_sha256": _sha256(manifest_path),
            "ranking": str(ranking_path.resolve()),
            "ranking_sha256": _sha256(ranking_path),
            "ranking_order": args.ranking_order,
            "selection_tag": args.tag,
            "split": args.split,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output.resolve()),
                "samples": len(manifest["samples"]),
                "images_zip": str((output / "images.zip").resolve()),
                "preannotations_zip": str(
                    (output / "preannotations-segmentation-mask-1.1.zip").resolve()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
