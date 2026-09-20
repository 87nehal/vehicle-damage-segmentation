"""Diagnose independent damage-type thresholds on a development split.

This does not create release evidence.  It measures whether rare type scores
contain usable signal that is hidden by the mutually exclusive argmax mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vehicle_damage.calibration import Calibration
from vehicle_damage.inference import predict_multiscale_probabilities
from vehicle_damage.manifest import load_manifest
from vehicle_damage.model import load_checkpoint


def _case_threshold(scores: list[float], target_recall: float) -> float:
    descending = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    required = int(np.ceil(target_recall * descending.size))
    return float(descending[required - 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--minimum-coverage", type=float, default=0.05)
    parser.add_argument("--negative-pixels-per-image", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device")
    args = parser.parse_args()
    if not 0 < args.target_recall <= 1 or not 0 < args.minimum_coverage <= 1:
        parser.error("recall and coverage must be in (0, 1]")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    calibration = Calibration.load(args.calibration)
    classes = checkpoint["classes"]
    samples = [
        sample
        for sample in load_manifest(args.manifest, require_commercial=True)
        if sample.split == args.split and sample.damage_supervised
    ]
    if not samples:
        raise ValueError(f"no supervised samples in split {args.split!r}")

    rng = np.random.default_rng(args.seed)
    case_scores: list[list[float]] = [[] for _ in classes]
    positive_scores: list[list[np.ndarray]] = [[] for _ in classes]
    negative_scores: list[list[np.ndarray]] = [[] for _ in classes]
    absent_image_maxima: list[list[float]] = [[] for _ in classes]

    for index, sample in enumerate(samples, 1):
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
        with Image.open(sample.mask) as source:
            target = np.asarray(source.convert("L"))
        damage, exterior = predict_multiscale_probabilities(
            model,
            image,
            device=device,
            scale_factors=calibration.effective_segmentation_scales,
            tile_size=calibration.tile_size,
            overlap=calibration.overlap,
            horizontal_flip_tta=calibration.horizontal_flip_tta,
            mixed_precision=calibration.mixed_precision,
        )
        exterior_factor = (
            calibration.exterior_floor
            + (1.0 - calibration.exterior_floor) * exterior.numpy()
        )
        probabilities = damage.numpy()
        for class_id in range(1, len(classes)):
            score = probabilities[class_id] * exterior_factor
            positive = score[target == class_id]
            if positive.size:
                case_scores[class_id].append(
                    float(
                        np.quantile(
                            positive,
                            1.0 - args.minimum_coverage,
                            method="lower",
                        )
                    )
                )
                positive_scores[class_id].append(positive)
            else:
                absent_image_maxima[class_id].append(float(score.max()))
            negative = score[target != class_id]
            if negative.size > args.negative_pixels_per_image:
                negative = rng.choice(
                    negative, size=args.negative_pixels_per_image, replace=False
                )
            negative_scores[class_id].append(negative)
        print(
            json.dumps(
                {"scored": index, "total": len(samples), "image": str(sample.image)}
            ),
            flush=True,
        )

    results: dict[str, dict] = {}
    for class_id, name in enumerate(classes[1:], 1):
        cases = case_scores[class_id]
        if not cases:
            results[name] = {"cases": 0, "threshold": None}
            continue
        threshold = _case_threshold(cases, args.target_recall)
        positives = np.concatenate(positive_scores[class_id])
        negatives = np.concatenate(negative_scores[class_id])
        absent = np.asarray(absent_image_maxima[class_id])
        results[name] = {
            "cases": len(cases),
            "threshold": threshold,
            "case_recall": float((np.asarray(cases) >= threshold).mean()),
            "positive_pixel_recall": float((positives >= threshold).mean()),
            "sampled_negative_pixel_fpr": float((negatives >= threshold).mean()),
            "absent_images": int(absent.size),
            "absent_image_false_alert_rate": (
                float((absent >= threshold).mean()) if absent.size else None
            ),
        }
    report = {
        "status": "development_diagnostic_only",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "calibration": str(Path(args.calibration).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "split": args.split,
        "target_case_recall": args.target_recall,
        "minimum_class_coverage": args.minimum_coverage,
        "inference_settings": {
            "tile_size": calibration.tile_size,
            "overlap": calibration.overlap,
            "horizontal_flip_tta": calibration.horizontal_flip_tta,
            "mixed_precision": calibration.mixed_precision,
            "segmentation_scales": calibration.effective_segmentation_scales,
            "exterior_floor": calibration.exterior_floor,
        },
        "classes": results,
    }
    rendered = json.dumps(report, indent=2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
