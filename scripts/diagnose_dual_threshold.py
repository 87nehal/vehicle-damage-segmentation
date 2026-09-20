from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vehicle_damage.calibration import Calibration
from vehicle_damage.evaluation import component_counts
from vehicle_damage.inference import (
    predict_multiscale_probabilities,
    probabilities_to_mask,
)
from vehicle_damage.manifest import load_manifest
from vehicle_damage.model import load_checkpoint


def _f_beta(precision: float | None, recall: float | None, beta: float) -> float:
    if precision is None or recall is None or precision + recall == 0:
        return 0.0
    beta_squared = beta * beta
    return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure a precise mask threshold separately from recall-first triage"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="calibration")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80],
    )
    parser.add_argument("--minimum-component-pixels", type=int, default=0)
    parser.add_argument("--region-minimum-pixels", type=int, default=16)
    parser.add_argument("--region-iou", type=float, default=0.10)
    parser.add_argument("--device")
    args = parser.parse_args()
    if any(not 0 < value < 1 for value in args.thresholds):
        parser.error("all --thresholds must be in (0, 1)")

    calibration = Calibration.load(args.calibration)
    thresholds = sorted(set([calibration.any_damage_threshold, *args.thresholds]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = load_checkpoint(args.checkpoint, device)
    samples = [
        sample
        for sample in load_manifest(args.manifest, require_commercial=True)
        if sample.split == args.split and sample.damage_supervised
    ]
    if not samples:
        raise ValueError(f"no damage-supervised samples in split {args.split!r}")

    counters = {
        threshold: {
            "case_hits": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "region_tp": 0,
            "region_fp": 0,
            "region_fn": 0,
        }
        for threshold in thresholds
    }
    coverage = calibration.minimum_damage_coverage or 0.05
    for index, sample in enumerate(samples, 1):
        image = Image.open(sample.image).convert("RGB")
        target = np.asarray(Image.open(sample.mask).convert("L"))
        target_damage = target > 0
        target_pixels = int(target_damage.sum())
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
        for threshold in thresholds:
            prediction, _ = probabilities_to_mask(
                damage,
                exterior,
                threshold,
                calibration.exterior_floor,
                args.minimum_component_pixels,
                calibration.type_probability_multipliers,
            )
            predicted_damage = prediction > 0
            values = counters[threshold]
            intersection = int((predicted_damage & target_damage).sum())
            values["case_hits"] += int(
                bool(target_pixels and intersection / target_pixels >= coverage)
            )
            values["tp"] += intersection
            values["fp"] += int((predicted_damage & ~target_damage).sum())
            values["fn"] += int((~predicted_damage & target_damage).sum())
            values["tn"] += int((~predicted_damage & ~target_damage).sum())
            region_tp, region_fp, region_fn, _ = component_counts(
                predicted_damage,
                target_damage,
                minimum_pixels=args.region_minimum_pixels,
                minimum_iou=args.region_iou,
            )
            values["region_tp"] += region_tp
            values["region_fp"] += region_fp
            values["region_fn"] += region_fn
        print(json.dumps({"processed": index, "total": len(samples)}))

    profiles = []
    for threshold in thresholds:
        values = counters[threshold]
        pixel_precision = values["tp"] / (values["tp"] + values["fp"])
        pixel_recall = values["tp"] / (values["tp"] + values["fn"])
        region_precision = values["region_tp"] / (
            values["region_tp"] + values["region_fp"]
        ) if values["region_tp"] + values["region_fp"] else None
        region_recall = values["region_tp"] / (
            values["region_tp"] + values["region_fn"]
        ) if values["region_tp"] + values["region_fn"] else None
        profiles.append(
            {
                "segmentation_threshold": threshold,
                "case_hits_if_used_alone": values["case_hits"],
                "cases": len(samples),
                "case_recall_if_used_alone": values["case_hits"] / len(samples),
                "pixel_precision": pixel_precision,
                "pixel_recall": pixel_recall,
                "pixel_f2": _f_beta(pixel_precision, pixel_recall, 2.0),
                "background_false_positive_rate": values["fp"] / (
                    values["fp"] + values["tn"]
                ),
                "region_true_positive": values["region_tp"],
                "region_false_positive": values["region_fp"],
                "region_false_negative": values["region_fn"],
                "region_precision": region_precision,
                "region_recall": region_recall,
                "region_f2": _f_beta(region_precision, region_recall, 2.0),
            }
        )
    best_region = max(profiles, key=lambda profile: profile["region_f2"])
    best_pixel = max(profiles, key=lambda profile: profile["pixel_f2"])
    report = {
        "status": "development_diagnostic_only",
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint)),
        "calibration": str(Path(args.calibration)),
        "triage_threshold": calibration.any_damage_threshold,
        "triage_target_case_recall": calibration.target_recall,
        "minimum_damage_coverage": coverage,
        "minimum_component_pixels": args.minimum_component_pixels,
        "segmentation_scales": calibration.effective_segmentation_scales,
        "type_probability_multipliers": calibration.type_probability_multipliers,
        "region_minimum_pixels": args.region_minimum_pixels,
        "region_iou": args.region_iou,
        "profiles": profiles,
        "best_region_f2_threshold": best_region["segmentation_threshold"],
        "best_pixel_f2_threshold": best_pixel["segmentation_threshold"],
        "selection_warning": (
            "A distinct segmentation threshold requires a low-threshold triage alert "
            "and manual review path; it must not replace recall-first case detection."
        ),
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
