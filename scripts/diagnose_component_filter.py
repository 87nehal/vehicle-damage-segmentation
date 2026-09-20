from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vehicle_damage.calibration import Calibration
from vehicle_damage.evaluation import component_counts
from vehicle_damage.inference import (
    filter_small_damage_components,
    predict_multiscale_probabilities,
    probabilities_to_mask,
)
from vehicle_damage.manifest import load_manifest
from vehicle_damage.model import load_checkpoint
from vehicle_damage.robustness import (
    STRESS_CONDITIONS,
    apply_stress,
    stress_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure component-area filtering on a calibration split"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-calibration")
    parser.add_argument("--split", default="calibration")
    parser.add_argument(
        "--stress-condition",
        choices=STRESS_CONDITIONS,
        default="clean",
        help="optional deterministic development stress transform",
    )
    parser.add_argument(
        "--prediction-role",
        choices=("triage", "segmentation"),
        default="triage",
        help="score the recall-first triage mask or the high-confidence mask",
    )
    parser.add_argument(
        "--minimum-pixels",
        type=int,
        nargs="+",
        default=[0, 16, 64, 256, 1024, 4096],
    )
    parser.add_argument("--region-minimum-pixels", type=int, default=16)
    parser.add_argument("--region-iou", type=float, default=0.10)
    parser.add_argument("--device")
    args = parser.parse_args()
    if any(value < 0 for value in args.minimum_pixels):
        parser.error("--minimum-pixels values must be non-negative")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    calibration = Calibration.load(args.calibration)
    model, _ = load_checkpoint(args.checkpoint, device)
    samples = [
        sample
        for sample in load_manifest(args.manifest, require_commercial=True)
        if sample.split == args.split and sample.damage_supervised
    ]
    if not samples:
        raise ValueError(f"no damage-supervised samples in split {args.split!r}")

    sizes = sorted(set([0, *args.minimum_pixels]))
    scales = (
        calibration.triage_scales
        if args.prediction_role == "triage"
        else calibration.effective_segmentation_scales
    )
    threshold = (
        calibration.any_damage_threshold
        if args.prediction_role == "triage"
        else calibration.effective_segmentation_threshold
    )
    counters = {
        size: {
            "case_hits": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "region_tp": 0,
            "region_fp": 0,
            "region_fn": 0,
        }
        for size in sizes
    }
    coverage = calibration.minimum_damage_coverage or 0.05
    for index, sample in enumerate(samples, 1):
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
        with Image.open(sample.mask) as source:
            target_image = source.convert("L")
        if args.stress_condition != "clean":
            identity = hashlib.sha256(Path(sample.image).read_bytes()).hexdigest()
            image, target_image = apply_stress(
                image,
                target_image,
                args.stress_condition,
                seed=stress_seed(identity, args.stress_condition),
            )
        target = np.asarray(target_image)
        damage, exterior = predict_multiscale_probabilities(
            model,
            image,
            device=device,
            scale_factors=scales,
            tile_size=calibration.tile_size,
            overlap=calibration.overlap,
            horizontal_flip_tta=calibration.horizontal_flip_tta,
            mixed_precision=calibration.mixed_precision,
        )
        prediction, _ = probabilities_to_mask(
            damage,
            exterior,
            threshold,
            calibration.exterior_floor,
            0,
            calibration.type_probability_multipliers,
        )
        target_damage = target > 0
        target_pixels = int(target_damage.sum())
        for size in sizes:
            filtered = filter_small_damage_components(prediction, size)
            predicted_damage = filtered > 0
            values = counters[size]
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
    for size in sizes:
        values = counters[size]
        case_recall = values["case_hits"] / len(samples)
        pixel_precision = values["tp"] / (values["tp"] + values["fp"])
        pixel_recall = values["tp"] / (values["tp"] + values["fn"])
        background_fpr = values["fp"] / (values["fp"] + values["tn"])
        region_precision = values["region_tp"] / (
            values["region_tp"] + values["region_fp"]
        ) if values["region_tp"] + values["region_fp"] else None
        region_recall = values["region_tp"] / (
            values["region_tp"] + values["region_fn"]
        ) if values["region_tp"] + values["region_fn"] else None
        profiles.append(
            {
                "minimum_component_pixels": size,
                "case_hits": values["case_hits"],
                "cases": len(samples),
                "case_recall": case_recall,
                "pixel_precision": pixel_precision,
                "pixel_recall": pixel_recall,
                "background_false_positive_rate": background_fpr,
                "region_true_positive": values["region_tp"],
                "region_false_positive": values["region_fp"],
                "region_false_negative": values["region_fn"],
                "region_precision": region_precision,
                "region_recall": region_recall,
            }
        )
    baseline = profiles[0]
    if args.prediction_role == "triage":
        eligible = [
            profile
            for profile in profiles
            if profile["case_recall"] >= calibration.target_recall
        ]
        selection_rule = (
            "lowest background pixel FPR among profiles meeting calibrated "
            "target case recall"
        )
    else:
        eligible = [
            profile
            for profile in profiles
            if profile["case_hits"] >= baseline["case_hits"]
            and profile["region_true_positive"] >= baseline["region_true_positive"]
            and profile["pixel_recall"] >= 0.999 * baseline["pixel_recall"]
        ]
        selection_rule = (
            "lowest background pixel FPR while preserving baseline case hits and "
            "matched regions and at least 99.9% of baseline pixel recall"
        )
    selected = (
        min(
            eligible,
            key=lambda profile: (
                profile["background_false_positive_rate"],
                profile["minimum_component_pixels"],
            ),
        )
        if eligible
        else None
    )
    report = {
        "status": "development_diagnostic_only",
        "split": args.split,
        "stress_condition": args.stress_condition,
        "checkpoint": str(Path(args.checkpoint)),
        "calibration": str(Path(args.calibration)),
        "prediction_role": args.prediction_role,
        "threshold": threshold,
        "inference_scales": scales,
        "mixed_precision": calibration.mixed_precision,
        "type_probability_multipliers": calibration.type_probability_multipliers,
        "target_case_recall": calibration.target_recall,
        "minimum_damage_coverage": coverage,
        "region_minimum_pixels": args.region_minimum_pixels,
        "region_iou": args.region_iou,
        "profiles": profiles,
        "selected_minimum_component_pixels": (
            selected["minimum_component_pixels"] if selected else None
        ),
        "selection_rule": selection_rule,
        "output_calibration": args.output_calibration,
    }
    if args.output_calibration and selected:
        if args.prediction_role == "triage":
            promoted = replace(
                calibration,
                measured_recall=selected["case_recall"],
                negative_pixel_fpr=selected["background_false_positive_rate"],
                minimum_component_pixels=selected["minimum_component_pixels"],
            )
        else:
            promoted = replace(
                calibration,
                segmentation_minimum_component_pixels=(
                    selected["minimum_component_pixels"]
                ),
                segmentation_component_filter_basis=(
                    f"{args.split} split: {selection_rule}"
                ),
            )
        promoted.save(args.output_calibration)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
