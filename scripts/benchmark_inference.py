from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

from vehicle_damage.calibration import Calibration
from vehicle_damage.inference import predict_profile_probabilities, probabilities_to_mask
from vehicle_damage.model import load_checkpoint


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("at least one timing is required")
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction + 0.999999)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the complete tiled inference path")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats must be positive")

    calibration = Calibration.load(args.calibration) if args.calibration else None
    if calibration:
        if args.tile_size is not None and args.tile_size != calibration.tile_size:
            parser.error("--tile-size conflicts with the calibration artifact")
        if args.overlap is not None and args.overlap != calibration.overlap:
            parser.error("--overlap conflicts with the calibration artifact")
        if args.no_tta and calibration.horizontal_flip_tta:
            parser.error("--no-tta conflicts with the calibration artifact")
        tile_size = calibration.tile_size
        overlap = calibration.overlap
        horizontal_flip_tta = calibration.horizontal_flip_tta
        mixed_precision = calibration.mixed_precision
    else:
        tile_size = args.tile_size or 768
        overlap = 192 if args.overlap is None else args.overlap
        horizontal_flip_tta = not args.no_tta
        mixed_precision = args.mixed_precision

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    model, checkpoint = load_checkpoint(str(checkpoint_path), device)
    with Image.open(args.image) as source:
        image = source.convert("RGB")

    def predict() -> None:
        triage_scales = calibration.triage_scales if calibration else (1.0,)
        segmentation_scales = (
            calibration.effective_segmentation_scales if calibration else (1.0,)
        )
        (triage_damage, triage_exterior), (mask_damage, mask_exterior) = (
            predict_profile_probabilities(
                model,
                image,
                device=device,
                triage_scales=triage_scales,
                segmentation_scales=segmentation_scales,
                tile_size=tile_size,
                overlap=overlap,
                horizontal_flip_tta=horizontal_flip_tta,
                mixed_precision=mixed_precision,
            )
        )
        if calibration:
            probabilities_to_mask(
                triage_damage,
                triage_exterior,
                calibration.any_damage_threshold,
                calibration.exterior_floor,
                calibration.minimum_component_pixels,
                calibration.type_probability_multipliers,
            )
            if (
                calibration.effective_segmentation_threshold
                != calibration.any_damage_threshold
                or calibration.effective_segmentation_scales
                != calibration.triage_scales
                or calibration.effective_segmentation_minimum_component_pixels
                != calibration.minimum_component_pixels
            ):
                probabilities_to_mask(
                    mask_damage,
                    mask_exterior,
                    calibration.effective_segmentation_threshold,
                    calibration.exterior_floor,
                    calibration.effective_segmentation_minimum_component_pixels,
                    calibration.type_probability_multipliers,
                )

    for _ in range(args.warmup):
        predict()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    timings_ms: list[float] = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        predict()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    median_ms = statistics.median(timings_ms)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "image": str(Path(args.image).resolve()),
        "image_width": image.width,
        "image_height": image.height,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "calibration": args.calibration,
        "includes_mask_postprocessing": calibration is not None,
        "includes_dual_mask_postprocessing": bool(
            calibration
            and calibration.effective_segmentation_threshold
            != calibration.any_damage_threshold
        ),
        "triage_threshold": calibration.any_damage_threshold if calibration else None,
        "segmentation_threshold": (
            calibration.effective_segmentation_threshold if calibration else None
        ),
        "minimum_component_pixels": (
            calibration.minimum_component_pixels if calibration else 0
        ),
        "triage_minimum_component_pixels": (
            calibration.minimum_component_pixels if calibration else 0
        ),
        "segmentation_minimum_component_pixels": (
            calibration.effective_segmentation_minimum_component_pixels
            if calibration else 0
        ),
        "segmentation_component_filter_basis": (
            calibration.segmentation_component_filter_basis if calibration else ""
        ),
        "type_probability_multipliers": (
            calibration.type_probability_multipliers if calibration else None
        ),
        "type_probability_multiplier_basis": (
            calibration.type_probability_multiplier_basis if calibration else ""
        ),
        "tile_size": tile_size,
        "overlap": overlap,
        "horizontal_flip_tta": horizontal_flip_tta,
        "mixed_precision": mixed_precision,
        "triage_scales": calibration.triage_scales if calibration else [1.0],
        "segmentation_scales": (
            calibration.effective_segmentation_scales if calibration else [1.0]
        ),
        "warmup_runs": args.warmup,
        "measured_runs": args.repeats,
        "latency_ms_median": median_ms,
        "latency_ms_p95": _percentile(timings_ms, 0.95),
        "latency_ms_min": min(timings_ms),
        "latency_ms_max": max(timings_ms),
        "throughput_images_per_second_from_median": 1000.0 / median_ms,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
