from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from .calibration import Calibration, calibrate_case_threshold
from .evaluation import CaseEvaluator
from .inference import (
    assess_quality,
    predict_multiscale_probabilities,
    predict_profile_probabilities,
    probabilities_to_mask,
    route_inference_decision,
)
from .manifest import load_manifest, validate_manifest
from .model import load_checkpoint
from .release import audit_release
from .training import train


def _frozen_inference_settings(
    calibration: Calibration, args: argparse.Namespace
) -> tuple[int, int, bool, bool]:
    """Use the calibrated scoring path and reject threshold-invalidating overrides."""
    if args.tile_size is not None and args.tile_size != calibration.tile_size:
        raise ValueError(
            f"tile size {args.tile_size} conflicts with calibrated value {calibration.tile_size}"
        )
    if args.overlap is not None and args.overlap != calibration.overlap:
        raise ValueError(
            f"overlap {args.overlap} conflicts with calibrated value {calibration.overlap}"
        )
    if args.no_tta and calibration.horizontal_flip_tta:
        raise ValueError("--no-tta conflicts with a threshold calibrated using flip TTA")
    return (
        calibration.tile_size,
        calibration.overlap,
        calibration.horizontal_flip_tta,
        calibration.mixed_precision,
    )


def _validate(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    samples = load_manifest(args.manifest, require_commercial=not args.allow_noncommercial)
    report = validate_manifest(samples, len(config["classes"]), check_files=not args.skip_files)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(2)


def _infer(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    calibration = Calibration.load(args.calibration)
    tile_size, overlap, horizontal_flip_tta, mixed_precision = _frozen_inference_settings(
        calibration, args
    )
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    image = Image.open(args.image).convert("RGB")
    quality = assess_quality(image)
    (triage_damage, triage_exterior), (mask_damage, mask_exterior) = (
        predict_profile_probabilities(
            model,
            image,
            device=device,
            triage_scales=calibration.triage_scales,
            segmentation_scales=calibration.effective_segmentation_scales,
            tile_size=tile_size,
            overlap=overlap,
            horizontal_flip_tta=horizontal_flip_tta,
            mixed_precision=mixed_precision,
        )
    )
    triage_mask, uncertainty = probabilities_to_mask(
        triage_damage,
        triage_exterior,
        calibration.any_damage_threshold,
        calibration.exterior_floor,
        calibration.minimum_component_pixels,
        calibration.type_probability_multipliers,
    )
    segmentation_threshold = calibration.effective_segmentation_threshold
    if (
        segmentation_threshold == calibration.any_damage_threshold
        and calibration.effective_segmentation_scales == calibration.triage_scales
        and calibration.effective_segmentation_minimum_component_pixels
        == calibration.minimum_component_pixels
    ):
        mask = triage_mask
    else:
        mask, _ = probabilities_to_mask(
            mask_damage,
            mask_exterior,
            segmentation_threshold,
            calibration.exterior_floor,
            calibration.effective_segmentation_minimum_component_pixels,
            calibration.type_probability_multipliers,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output)
    review_only_mask = np.where((triage_mask > 0) & (mask == 0), triage_mask, 0).astype(
        np.uint8
    )
    review_output: Path | None = None
    disagreement_output: Path | None = None
    if calibration.segmentation_threshold is not None:
        suffix = output.suffix or ".png"
        review_output = output.with_name(f"{output.stem}.review{suffix}")
        Image.fromarray(review_only_mask, mode="L").save(review_output)
        disagreement_mask = np.where(
            triage_mask != mask,
            np.where(mask > 0, mask, triage_mask),
            0,
        ).astype(np.uint8)
        disagreement_output = output.with_name(
            f"{output.stem}.disagreement{suffix}"
        )
        Image.fromarray(disagreement_mask, mode="L").save(disagreement_output)
    near_threshold = float((uncertainty < args.uncertainty_margin).mean())
    review_only_fraction = float((review_only_mask > 0).mean())
    decision = route_inference_decision(
        quality,
        triage_mask,
        mask,
        near_threshold_fraction=near_threshold,
        max_uncertain_fraction=args.max_uncertain_fraction,
    )
    report = {
        "image": str(args.image),
        "mask": str(output),
        "review_mask": str(review_output) if review_output is not None else None,
        "disagreement_mask": (
            str(disagreement_output) if disagreement_output is not None else None
        ),
        "classes": checkpoint["classes"],
        "quality": asdict(quality),
        "inference_settings": {
            "tile_size": tile_size,
            "overlap": overlap,
            "horizontal_flip_tta": horizontal_flip_tta,
            "mixed_precision": mixed_precision,
            "triage_scales": calibration.triage_scales,
            "segmentation_scales": calibration.effective_segmentation_scales,
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
            "triage_threshold": calibration.any_damage_threshold,
            "segmentation_threshold": segmentation_threshold,
            "segmentation_threshold_basis": calibration.segmentation_threshold_basis,
        },
        "triage_alert": bool(np.any(triage_mask)),
        "high_confidence_alert": bool(np.any(mask)),
        "review_only_fraction": review_only_fraction,
        **asdict(decision),
    }
    Path(str(output) + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _calibrate(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = load_checkpoint(args.checkpoint, device)
    samples = [
        x for x in load_manifest(args.manifest, require_commercial=True)
        if x.split == args.split and x.damage_supervised
    ]
    if not samples:
        raise ValueError(f"no samples found for split {args.split!r}")
    rng = np.random.default_rng(args.seed)
    negative_scores: list[np.ndarray] = []
    case_scores: list[float] = []
    positive_pixel_count = 0
    for index, sample in enumerate(samples, 1):
        image = Image.open(sample.image).convert("RGB")
        target = np.asarray(Image.open(sample.mask).convert("L")) > 0
        if target.shape != (image.height, image.width):
            raise ValueError(f"image/mask size mismatch: {sample.image} / {sample.mask}")
        damage, exterior = predict_multiscale_probabilities(
            model,
            image,
            device=device,
            scale_factors=args.inference_scales,
            tile_size=args.tile_size,
            overlap=args.overlap,
            horizontal_flip_tta=not args.no_tta,
            mixed_precision=args.mixed_precision,
        )
        score = (
            (1.0 - damage[0])
            * (args.exterior_floor + (1.0 - args.exterior_floor) * exterior)
        ).numpy()
        if target.any():
            positive_pixel_count += int(target.sum())
            case_scores.append(
                float(np.quantile(score[target], 1.0 - args.min_damage_coverage, method="lower"))
            )
        values = score[~target]
        if values.size > args.pixels_per_image:
            values = rng.choice(values, size=args.pixels_per_image, replace=False)
        if values.size:
            negative_scores.append(values)
        print(json.dumps({"calibrated": index, "total": len(samples), "image": str(sample.image)}))
    if not case_scores or not negative_scores:
        raise ValueError("calibration split must include damage cases and non-damage pixels")
    result = calibrate_case_threshold(
        np.asarray(case_scores), np.concatenate(negative_scores),
        target_recall=args.target_recall,
        positive_pixels=positive_pixel_count,
        minimum_damage_coverage=args.min_damage_coverage,
        exterior_floor=args.exterior_floor,
        tile_size=args.tile_size,
        overlap=args.overlap,
        horizontal_flip_tta=not args.no_tta,
        mixed_precision=args.mixed_precision,
        triage_scales=args.inference_scales,
    )
    result.save(args.output)
    print(json.dumps(asdict(result), indent=2))


def _evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    calibration = Calibration.load(args.calibration)
    tile_size, overlap, horizontal_flip_tta, mixed_precision = _frozen_inference_settings(
        calibration, args
    )
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    threshold = calibration.any_damage_threshold
    segmentation_threshold = calibration.effective_segmentation_threshold
    samples = [
        x for x in load_manifest(args.manifest, require_commercial=True)
        if x.split == args.split and x.damage_supervised
    ]
    if not samples:
        raise ValueError(f"no samples found for split {args.split!r}")
    coverage = calibration.minimum_damage_coverage or args.min_damage_coverage
    evaluator = CaseEvaluator(
        checkpoint["classes"],
        minimum_damage_coverage=coverage,
        minimum_component_pixels=args.min_component_pixels,
        minimum_region_iou=args.region_iou,
    )
    quality_review = 0
    for index, sample in enumerate(samples, 1):
        image = Image.open(sample.image).convert("RGB")
        target = np.asarray(Image.open(sample.mask).convert("L"))
        (triage_damage, triage_exterior), (mask_damage, mask_exterior) = (
            predict_profile_probabilities(
                model,
                image,
                device=device,
                triage_scales=calibration.triage_scales,
                segmentation_scales=calibration.effective_segmentation_scales,
                tile_size=tile_size,
                overlap=overlap,
                horizontal_flip_tta=horizontal_flip_tta,
                mixed_precision=mixed_precision,
            )
        )
        triage_prediction, _ = probabilities_to_mask(
            triage_damage,
            triage_exterior,
            threshold,
            calibration.exterior_floor,
            calibration.minimum_component_pixels,
            calibration.type_probability_multipliers,
        )
        if (
            segmentation_threshold == threshold
            and calibration.effective_segmentation_scales == calibration.triage_scales
            and calibration.effective_segmentation_minimum_component_pixels
            == calibration.minimum_component_pixels
        ):
            prediction = triage_prediction
        else:
            prediction, _ = probabilities_to_mask(
                mask_damage,
                mask_exterior,
                segmentation_threshold,
                calibration.exterior_floor,
                calibration.effective_segmentation_minimum_component_pixels,
                calibration.type_probability_multipliers,
            )
        evaluator.update(
            prediction,
            target,
            sample.tags,
            sample.group_id,
            case_prediction=triage_prediction,
        )
        quality_review += int(bool(assess_quality(image).review_reasons))
        print(json.dumps({"evaluated": index, "total": len(samples), "image": str(sample.image)}))
    report = evaluator.report()
    report["split"] = args.split
    report["quality_review_images"] = quality_review
    report["quality_review_rate"] = quality_review / len(samples)
    report["inference_settings"] = {
        "tile_size": tile_size,
        "overlap": overlap,
        "horizontal_flip_tta": horizontal_flip_tta,
        "mixed_precision": mixed_precision,
        "triage_scales": calibration.triage_scales,
        "segmentation_scales": calibration.effective_segmentation_scales,
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
        "triage_threshold": threshold,
        "segmentation_threshold": segmentation_threshold,
        "segmentation_threshold_basis": calibration.segmentation_threshold_basis,
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _audit_release(args: argparse.Namespace) -> None:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    audit = audit_release(report)
    if args.output:
        Path(args.output).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["approved"] and not args.no_fail:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vehicle-damage")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--config", default="configs/base.yaml")
    validate.add_argument("--skip-files", action="store_true")
    validate.add_argument("--allow-noncommercial", action="store_true", help="audit only; training still rejects it")
    validate.set_defaults(func=_validate)

    training = sub.add_parser("train")
    training.add_argument("--manifest", required=True)
    training.add_argument("--config", default="configs/base.yaml")
    training.add_argument("--output-dir", default="runs/baseline")
    training.add_argument("--device")
    training.set_defaults(func=lambda a: train(a.config, a.manifest, a.output_dir, a.device))

    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--checkpoint", required=True)
    calibrate.add_argument("--manifest", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--split", default="calibration")
    calibrate.add_argument("--target-recall", type=float, default=0.97)
    calibrate.add_argument("--min-damage-coverage", type=float, default=0.05)
    calibrate.add_argument(
        "--exterior-floor", type=float, default=0.5,
        help="1 disables exterior-score suppression; tune only on calibration data",
    )
    calibrate.add_argument("--pixels-per-image", type=int, default=200000)
    calibrate.add_argument("--seed", type=int, default=17)
    calibrate.add_argument("--device")
    calibrate.add_argument("--tile-size", type=int, default=768)
    calibrate.add_argument("--overlap", type=int, default=192)
    calibrate.add_argument("--no-tta", action="store_true")
    calibrate.add_argument(
        "--inference-scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="input scales to average and freeze into calibration",
    )
    calibrate.add_argument(
        "--mixed-precision",
        action="store_true",
        help="calibrate and freeze CUDA FP16 autocast inference",
    )
    calibrate.set_defaults(func=_calibrate)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--calibration", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--min-damage-coverage", type=float, default=0.05)
    evaluate.add_argument("--min-component-pixels", type=int, default=16)
    evaluate.add_argument("--region-iou", type=float, default=0.10)
    evaluate.add_argument("--device")
    evaluate.add_argument("--tile-size", type=int)
    evaluate.add_argument("--overlap", type=int)
    evaluate.add_argument("--no-tta", action="store_true")
    evaluate.set_defaults(func=_evaluate)

    infer = sub.add_parser("infer")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--calibration", required=True)
    infer.add_argument("--image", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--device")
    infer.add_argument("--tile-size", type=int)
    infer.add_argument("--overlap", type=int)
    infer.add_argument("--no-tta", action="store_true")
    infer.add_argument("--uncertainty-margin", type=float, default=0.05)
    infer.add_argument("--max-uncertain-fraction", type=float, default=0.15)
    infer.set_defaults(func=_infer)

    audit = sub.add_parser("audit-release")
    audit.add_argument("--report", required=True)
    audit.add_argument("--output")
    audit.add_argument("--no-fail", action="store_true", help="report failures without a nonzero exit")
    audit.set_defaults(func=_audit_release)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
