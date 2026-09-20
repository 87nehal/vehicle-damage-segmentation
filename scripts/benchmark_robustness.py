"""Measure degradation under deterministic, synthetic nuisance conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vehicle_damage.calibration import Calibration
from vehicle_damage.evaluation import CaseEvaluator
from vehicle_damage.inference import (
    assess_quality,
    predict_profile_probabilities,
    probabilities_to_mask,
)
from vehicle_damage.manifest import load_manifest
from vehicle_damage.model import load_checkpoint
from vehicle_damage.robustness import (
    STRESS_CONDITIONS,
    STRESS_DESCRIPTIONS,
    STRESS_VERSION,
    apply_stress,
    stress_seed,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _delta(value: float | None, baseline: float | None) -> float | None:
    return None if value is None or baseline is None else value - baseline


def _summary_delta(report: dict, baseline: dict) -> dict:
    pixel, base_pixel = report["pixel_metrics"], baseline["pixel_metrics"]
    region, base_region = report["region_metrics"], baseline["region_metrics"]
    return {
        "case_recall_delta": _delta(report["case_recall"], baseline["case_recall"]),
        "pixel_precision_delta": _delta(pixel["precision"], base_pixel["precision"]),
        "pixel_recall_delta": _delta(pixel["recall"], base_pixel["recall"]),
        "background_false_positive_rate_delta": _delta(
            pixel["false_positive_rate"], base_pixel["false_positive_rate"]
        ),
        "region_precision_delta": _delta(region["precision"], base_region["precision"]),
        "region_recall_delta": _delta(region["recall"], base_region["recall"]),
    }


def _new_sweep_state() -> dict:
    return {"groups": {}, "false_positive_pixels": 0, "true_negative_pixels": 0}


def _update_sweep_state(
    state: dict,
    prediction: np.ndarray,
    target: np.ndarray,
    group_id: str,
    minimum_coverage: float,
) -> tuple[bool, float]:
    actual = bool((target > 0).any())
    predicted = bool((prediction > 0).any())
    target_pixels = int((target > 0).sum())
    covered = int(((prediction > 0) & (target > 0)).sum())
    coverage = covered / target_pixels if target_pixels else 0.0
    group = state["groups"].setdefault(
        group_id, {"actual_positive": False, "hit": False, "predicted_positive": False}
    )
    group["actual_positive"] |= actual
    group["hit"] |= bool(actual and coverage >= minimum_coverage)
    group["predicted_positive"] |= predicted
    background = target == 0
    state["false_positive_pixels"] += int(((prediction > 0) & background).sum())
    state["true_negative_pixels"] += int(((prediction == 0) & background).sum())
    return bool(actual and coverage >= minimum_coverage), coverage


def _finish_sweep_state(state: dict, threshold: float) -> dict:
    groups = state["groups"]
    cases = sum(value["actual_positive"] for value in groups.values())
    hits = sum(value["actual_positive"] and value["hit"] for value in groups.values())
    clean = len(groups) - cases
    clean_alerts = sum(
        not value["actual_positive"] and value["predicted_positive"]
        for value in groups.values()
    )
    fp = state["false_positive_pixels"]
    tn = state["true_negative_pixels"]
    return {
        "threshold": threshold,
        "damage_cases": cases,
        "detected_cases": hits,
        "case_recall": hits / cases if cases else None,
        "clean_groups": clean,
        "clean_alerts": clean_alerts,
        "clean_false_alert_rate": clean_alerts / clean if clean else None,
        "background_false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "missed_group_ids": sorted(
            group_id
            for group_id, value in groups.items()
            if value["actual_positive"] and not value["hit"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a frozen model/profile over deterministic synthetic stress conditions"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-component-pixels", type=int, default=16)
    parser.add_argument("--region-iou", type=float, default=0.10)
    parser.add_argument(
        "--inference-scales",
        type=float,
        nargs="+",
        default=None,
        help="diagnostic input scales; values other than 1.0 are not covered by the profile",
    )
    parser.add_argument(
        "--triage-thresholds",
        type=float,
        nargs="*",
        default=None,
        help="additional lower thresholds for diagnostic recall/FPR sweeps",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=STRESS_CONDITIONS,
        default=list(STRESS_CONDITIONS),
    )
    args = parser.parse_args()

    conditions = list(dict.fromkeys(args.conditions))
    if "clean" not in conditions:
        parser.error("--conditions must include clean for degradation comparisons")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.inference_scales is not None and any(
        not np.isfinite(value) or value <= 0 for value in args.inference_scales
    ):
        parser.error("--inference-scales must contain positive finite values")
    if args.inference_scales is not None and len(
        set(args.inference_scales)
    ) != len(args.inference_scales):
        parser.error("--inference-scales must be unique")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    calibration = Calibration.load(args.calibration)
    if calibration.mixed_precision and device.type != "cuda":
        raise ValueError("the frozen profile requires CUDA mixed-precision inference")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if args.inference_scales is None:
        triage_scales = calibration.triage_scales
        segmentation_scales = calibration.effective_segmentation_scales
        scoring_path_modified = False
    else:
        triage_scales = tuple(args.inference_scales)
        segmentation_scales = tuple(args.inference_scales)
        scoring_path_modified = (
            triage_scales != calibration.triage_scales
            or segmentation_scales != calibration.effective_segmentation_scales
        )
    samples = [
        sample
        for sample in load_manifest(args.manifest, require_commercial=True)
        if sample.split == args.split and sample.damage_supervised
    ]
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError(f"no damage-supervised samples found for split {args.split!r}")
    coverage = calibration.minimum_damage_coverage or 0.05
    requested_thresholds = (
        args.triage_thresholds
        if args.triage_thresholds is not None
        else [
            max(0.0, calibration.any_damage_threshold - 0.025),
            max(0.0, calibration.any_damage_threshold - 0.05),
        ]
    )
    triage_thresholds = sorted(
        {calibration.any_damage_threshold, *requested_thresholds}, reverse=True
    )
    if any(
        threshold < 0 or threshold > calibration.any_damage_threshold
        for threshold in triage_thresholds
    ):
        parser.error(
            "diagnostic triage thresholds must be in [0, frozen triage threshold]"
        )
    evaluators = {
        condition: CaseEvaluator(
            checkpoint["classes"],
            minimum_damage_coverage=coverage,
            minimum_component_pixels=args.min_component_pixels,
            minimum_region_iou=args.region_iou,
        )
        for condition in conditions
    }
    quality_review = {condition: 0 for condition in conditions}
    sweep_states = {
        condition: {
            threshold: _new_sweep_state() for threshold in triage_thresholds
        }
        for condition in conditions
    }
    missed_views = {condition: [] for condition in conditions}

    for index, sample in enumerate(samples, 1):
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
        with Image.open(sample.mask) as source:
            target = source.convert("L")
        sample_identity = _sha256(sample.image)
        for condition in conditions:
            stressed_image, stressed_target = apply_stress(
                image,
                target,
                condition,
                seed=stress_seed(sample_identity, condition),
            )
            target_array = np.asarray(stressed_target)
            (triage_damage, triage_exterior), (mask_damage, mask_exterior) = (
                predict_profile_probabilities(
                    model,
                    stressed_image,
                    device=device,
                    triage_scales=triage_scales,
                    segmentation_scales=segmentation_scales,
                    tile_size=calibration.tile_size,
                    overlap=calibration.overlap,
                    horizontal_flip_tta=calibration.horizontal_flip_tta,
                    mixed_precision=calibration.mixed_precision,
                )
            )
            triage_predictions = {}
            for threshold in triage_thresholds:
                triage_predictions[threshold], _ = probabilities_to_mask(
                    triage_damage,
                    triage_exterior,
                    threshold,
                    calibration.exterior_floor,
                    calibration.minimum_component_pixels,
                    calibration.type_probability_multipliers,
                )
                hit, view_coverage = _update_sweep_state(
                    sweep_states[condition][threshold],
                    triage_predictions[threshold],
                    target_array,
                    sample.group_id,
                    coverage,
                )
                if threshold == calibration.any_damage_threshold and not hit and (
                    target_array > 0
                ).any():
                    missed_views[condition].append(
                        {
                            "group_id": sample.group_id,
                            "image": str(sample.image),
                            "target_pixels": int((target_array > 0).sum()),
                            "covered_fraction": view_coverage,
                        }
                    )
            triage = triage_predictions[calibration.any_damage_threshold]
            precise, _ = probabilities_to_mask(
                mask_damage,
                mask_exterior,
                calibration.effective_segmentation_threshold,
                calibration.exterior_floor,
                calibration.effective_segmentation_minimum_component_pixels,
                calibration.type_probability_multipliers,
            )
            evaluators[condition].update(
                precise,
                target_array,
                (*sample.tags, f"stress:{condition}"),
                sample.group_id,
                case_prediction=triage,
            )
            quality_review[condition] += int(
                bool(assess_quality(stressed_image).review_reasons)
            )
        if index == 1 or index % 10 == 0 or index == len(samples):
            print(json.dumps({"evaluated": index, "total": len(samples)}), flush=True)

    reports = {condition: evaluators[condition].report() for condition in conditions}
    baseline = reports["clean"]
    for condition, report in reports.items():
        report["quality_review_images"] = quality_review[condition]
        report["quality_review_rate"] = quality_review[condition] / len(samples)
        report["delta_from_clean"] = _summary_delta(report, baseline)
        report["triage_threshold_sweep"] = [
            _finish_sweep_state(sweep_states[condition][threshold], threshold)
            for threshold in triage_thresholds
        ]
        report["missed_positive_views_at_frozen_threshold"] = missed_views[condition]
    non_clean = [name for name in conditions if name != "clean"]
    report = {
        "report_type": "synthetic_development_robustness_diagnostic",
        "stress_version": STRESS_VERSION,
        "stress_descriptions": STRESS_DESCRIPTIONS,
        "production_evidence": False,
        "profile_scoring_path_modified": scoring_path_modified,
        "warning": (
            "Synthetic corruptions on a consumed development split are diagnostic only; "
            "they do not replace independently captured and reviewed deployment data."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _sha256(Path(args.checkpoint)),
        "calibration": str(Path(args.calibration).resolve()),
        "calibration_sha256": _sha256(Path(args.calibration)),
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": _sha256(Path(args.manifest)),
        "split": args.split,
        "samples": len(samples),
        "groups": baseline["groups_evaluated"],
        "conditions": conditions,
        "inference_settings": {
            "device": str(device),
            "tile_size": calibration.tile_size,
            "overlap": calibration.overlap,
            "horizontal_flip_tta": calibration.horizontal_flip_tta,
            "mixed_precision": calibration.mixed_precision,
            "triage_scales": triage_scales,
            "segmentation_scales": segmentation_scales,
            "triage_threshold": calibration.any_damage_threshold,
            "segmentation_threshold": calibration.effective_segmentation_threshold,
            "minimum_component_pixels": calibration.minimum_component_pixels,
            "triage_minimum_component_pixels": calibration.minimum_component_pixels,
            "segmentation_minimum_component_pixels": (
                calibration.effective_segmentation_minimum_component_pixels
            ),
            "segmentation_component_filter_basis": (
                calibration.segmentation_component_filter_basis
            ),
            "type_probability_multipliers": calibration.type_probability_multipliers,
            "type_probability_multiplier_basis": (
                calibration.type_probability_multiplier_basis
            ),
        },
        "worst_condition_case_recall": (
            min(non_clean, key=lambda name: reports[name]["case_recall"])
            if non_clean else "clean"
        ),
        "worst_condition_background_fpr": (
            max(
                non_clean,
                key=lambda name: reports[name]["pixel_metrics"]["false_positive_rate"],
            )
            if non_clean else "clean"
        ),
        "results": reports,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "worst_condition_case_recall": report["worst_condition_case_recall"],
        "worst_condition_background_fpr": report["worst_condition_background_fpr"],
    }, indent=2))


if __name__ == "__main__":
    main()
