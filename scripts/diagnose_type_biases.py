"""Search relative damage-type multipliers without changing damage presence.

The diagnostic is calibration-only. Multipliers affect only the type argmax
inside pixels already accepted by the frozen triage or segmentation path, so
they cannot create a new damage alert or change the binary damage mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vehicle_damage.calibration import Calibration
from vehicle_damage.inference import predict_profile_probabilities
from vehicle_damage.manifest import load_manifest
from vehicle_damage.model import load_checkpoint


@dataclass
class CasePixels:
    group_id: str
    class_index: int
    scores: np.ndarray
    accepted: np.ndarray


@dataclass
class PixelBlock:
    scores: np.ndarray
    accepted: np.ndarray
    target: np.ndarray


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision + recall == 0:
        return 0.0
    beta_squared = beta * beta
    return (1 + beta_squared) * precision * recall / (
        beta_squared * precision + recall
    )


def _metrics(
    multipliers: np.ndarray,
    cases: list[CasePixels],
    pixels: list[PixelBlock],
    class_names: list[str],
    minimum_coverage: float,
) -> dict:
    class_count = len(class_names) - 1
    group_hits: dict[tuple[str, int], bool] = {}
    group_cases: set[tuple[str, int]] = set()
    for entry in cases:
        key = (entry.group_id, entry.class_index)
        group_cases.add(key)
        predicted = np.argmax(entry.scores * multipliers, axis=1)
        covered = np.mean(entry.accepted & (predicted == entry.class_index))
        group_hits[key] = group_hits.get(key, False) or covered >= minimum_coverage

    confusion = np.zeros((class_count + 1, class_count + 1), dtype=np.int64)
    for block in pixels:
        predicted = np.argmax(block.scores * multipliers, axis=1) + 1
        predicted = np.where(block.accepted, predicted, 0)
        encoded = block.target.astype(np.int64) * (class_count + 1) + predicted
        confusion += np.bincount(
            encoded, minlength=(class_count + 1) ** 2
        ).reshape(class_count + 1, class_count + 1)

    per_class = {}
    recalls = []
    f2_values = []
    for class_index, name in enumerate(class_names[1:]):
        class_id = class_index + 1
        keys = [key for key in group_cases if key[1] == class_index]
        hits = int(sum(group_hits.get(key, False) for key in keys))
        case_recall = float(hits / len(keys)) if keys else 0.0
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        pixel_recall = tp / (tp + fn) if tp + fn else 0.0
        pixel_f2 = _f_beta(precision, pixel_recall)
        recalls.append(case_recall)
        f2_values.append(pixel_f2)
        per_class[name] = {
            "cases": len(keys),
            "hits": hits,
            "case_recall": case_recall,
            "pixel_precision": precision,
            "pixel_recall": pixel_recall,
            "pixel_f2": pixel_f2,
        }
    return {
        "multipliers": [float(value) for value in multipliers],
        "classes_at_or_above_90_percent_case_recall": int(
            sum(value >= 0.90 for value in recalls)
        ),
        "minimum_case_recall": float(min(recalls)),
        "macro_case_recall": float(np.mean(recalls)),
        "macro_pixel_f2": float(np.mean(f2_values)),
        "per_class": per_class,
    }


def _selection_key(metrics: dict) -> tuple:
    return (
        metrics["classes_at_or_above_90_percent_case_recall"],
        metrics["macro_case_recall"] + metrics["macro_pixel_f2"],
        metrics["minimum_case_recall"],
        metrics["macro_pixel_f2"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose relative type biases inside frozen damage masks"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--candidates", type=int, default=400)
    parser.add_argument("--search-pixels", type=int, default=250000)
    parser.add_argument("--case-pixels", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        help="evaluate one frozen multiplier vector instead of running a search",
    )
    parser.add_argument(
        "--multiplier-exponents",
        type=float,
        nargs="+",
        default=[1.0],
        help="evaluate geometric interpolations multiplier**exponent",
    )
    parser.add_argument(
        "--additional-multipliers",
        type=float,
        nargs="+",
        action="append",
        default=[],
        help="additional frozen multiplier vectors to compare in the same pass",
    )
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.candidates <= 0 or args.search_pixels <= 0 or args.case_pixels <= 0:
        parser.error("candidate and sampling counts must be positive")
    if any(not np.isfinite(value) or value <= 0 for value in args.multiplier_exponents):
        parser.error("--multiplier-exponents must contain positive finite values")

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    calibration = Calibration.load(args.calibration)
    if calibration.mixed_precision and device.type != "cuda":
        raise ValueError("the frozen profile requires CUDA mixed precision")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    class_names = checkpoint["classes"]
    class_count = len(class_names) - 1
    if args.multipliers is not None and (
        len(args.multipliers) != class_count
        or any(not np.isfinite(value) or value <= 0 for value in args.multipliers)
    ):
        parser.error(
            f"--multipliers requires {class_count} positive finite values"
        )
    for values in args.additional_multipliers:
        if len(values) != class_count or any(
            not np.isfinite(value) or value <= 0 for value in values
        ):
            parser.error(
                f"each --additional-multipliers vector requires {class_count} "
                "positive finite values"
            )
    samples = [
        sample
        for sample in load_manifest(args.manifest, require_commercial=True)
        if sample.split == args.split and sample.damage_supervised
    ]
    if not samples:
        raise ValueError(f"no damage-supervised samples in split {args.split!r}")

    cases: list[CasePixels] = []
    pixels: list[PixelBlock] = []
    search_score_parts: list[np.ndarray] = []
    search_accept_parts: list[np.ndarray] = []
    search_target_parts: list[np.ndarray] = []
    search_cases: list[CasePixels] = []
    exterior_floor = calibration.exterior_floor
    for index, sample in enumerate(samples, 1):
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
        with Image.open(sample.mask) as source:
            target = np.asarray(source.convert("L"))
        (triage_damage, triage_exterior), (mask_damage, mask_exterior) = (
            predict_profile_probabilities(
                model,
                image,
                device=device,
                triage_scales=calibration.triage_scales,
                segmentation_scales=calibration.effective_segmentation_scales,
                tile_size=calibration.tile_size,
                overlap=calibration.overlap,
                horizontal_flip_tta=calibration.horizontal_flip_tta,
                mixed_precision=calibration.mixed_precision,
            )
        )
        triage_score = (1.0 - triage_damage[0]) * (
            exterior_floor + (1.0 - exterior_floor) * triage_exterior
        )
        mask_score = (1.0 - mask_damage[0]) * (
            exterior_floor + (1.0 - exterior_floor) * mask_exterior
        )
        triage_accepted = triage_score.numpy() >= calibration.any_damage_threshold
        mask_accepted = mask_score.numpy() >= calibration.effective_segmentation_threshold
        triage_types = triage_damage[1:].numpy()
        mask_types = mask_damage[1:].numpy()

        for class_index in range(class_count):
            selected = target == class_index + 1
            if not selected.any():
                continue
            scores = triage_types[:, selected].T.astype(np.float16)
            accepted = triage_accepted[selected]
            cases.append(CasePixels(sample.group_id, class_index, scores, accepted))
            if scores.shape[0] > args.case_pixels:
                chosen = rng.choice(scores.shape[0], args.case_pixels, replace=False)
                scores = scores[chosen]
                accepted = accepted[chosen]
            search_cases.append(
                CasePixels(sample.group_id, class_index, scores, accepted)
            )

        union = mask_accepted | (target > 0)
        scores = mask_types[:, union].T.astype(np.float16)
        accepted = mask_accepted[union]
        target_values = target[union]
        pixels.append(PixelBlock(scores, accepted, target_values))
        sample_count = min(10000, scores.shape[0])
        chosen = rng.choice(scores.shape[0], sample_count, replace=False)
        search_score_parts.append(scores[chosen])
        search_accept_parts.append(accepted[chosen])
        search_target_parts.append(target_values[chosen])
        print(json.dumps({"scored": index, "total": len(samples)}), flush=True)

    search_scores = np.concatenate(search_score_parts)
    search_accepted = np.concatenate(search_accept_parts)
    search_targets = np.concatenate(search_target_parts)
    if search_scores.shape[0] > args.search_pixels:
        chosen = rng.choice(search_scores.shape[0], args.search_pixels, replace=False)
        search_scores = search_scores[chosen]
        search_accepted = search_accepted[chosen]
        search_targets = search_targets[chosen]
    search_pixels = [PixelBlock(search_scores, search_accepted, search_targets)]

    candidates = [np.ones(class_count, dtype=np.float32)]
    if args.multipliers is not None:
        fixed = np.asarray(args.multipliers, dtype=np.float32)
        candidates.extend(
            np.power(fixed, exponent).astype(np.float32)
            for exponent in sorted(set(args.multiplier_exponents))
        )
    else:
        anchor = 2 if class_count > 2 else 0
        for _ in range(args.candidates - 1):
            values = np.exp(rng.uniform(-1.5, 1.5, size=class_count)).astype(np.float32)
            values /= values[anchor]
            candidates.append(values)
        for class_index in range(class_count):
            for value in (0.5, 0.75, 1.5, 2.0, 3.0):
                candidate = np.ones(class_count, dtype=np.float32)
                candidate[class_index] = value
                candidates.append(candidate)
    candidates.extend(
        np.asarray(values, dtype=np.float32)
        for values in args.additional_multipliers
    )

    approximate = [
        _metrics(
            candidate,
            search_cases,
            search_pixels,
            class_names,
            calibration.minimum_damage_coverage,
        )
        for candidate in candidates
    ]
    ranked_indices = sorted(
        range(len(candidates)),
        key=lambda index: _selection_key(approximate[index]),
        reverse=True,
    )[:20]
    baseline = _metrics(
        candidates[0],
        cases,
        pixels,
        class_names,
        calibration.minimum_damage_coverage,
    )
    exact = [
        _metrics(
            candidates[index],
            cases,
            pixels,
            class_names,
            calibration.minimum_damage_coverage,
        )
        for index in ranked_indices
    ]
    eligible = [
        result
        for result in [baseline, *exact]
        if result["macro_pixel_f2"] >= 0.98 * baseline["macro_pixel_f2"]
    ]
    selected = max(eligible, key=_selection_key)
    report = {
        "status": "development_diagnostic_only",
        "warning": (
            "Type multipliers are searched on development calibration data and do "
            "not change binary damage presence. Fresh calibration and test data are "
            "required before deployment."
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "calibration": str(Path(args.calibration).resolve()),
        "calibration_sha256": _sha256(args.calibration),
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "split": args.split,
        "samples": len(samples),
        "candidate_count": len(candidates),
        "fixed_multiplier_evaluation": args.multipliers is not None,
        "multiplier_exponents": args.multiplier_exponents,
        "additional_multiplier_vectors": len(args.additional_multipliers),
        "selection_rule": (
            "maximize classes at >=90% case recall, then macro case recall plus "
            "macro pixel F2, subject to retaining >=98% of baseline macro pixel F2"
        ),
        "baseline": baseline,
        "selected": selected,
        "exact_finalists": sorted(exact, key=_selection_key, reverse=True),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
