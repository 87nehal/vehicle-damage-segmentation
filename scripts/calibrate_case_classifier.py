"""Calibrate independent image-level damage-class thresholds on one split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vehicle_damage.inference import predict_case_probabilities
from vehicle_damage.manifest import load_manifest
from vehicle_damage.model import load_checkpoint


def _threshold_for_recall(positive_scores: np.ndarray, target_recall: float) -> float:
    descending = np.sort(positive_scores)[::-1]
    required = int(np.ceil(target_recall * descending.size))
    return float(descending[required - 1])


def _auroc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    if not positive.size or not negative.size:
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()) / comparisons.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--device")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()
    if not 0 < args.target_recall <= 1:
        parser.error("--target-recall must be in (0, 1]")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.checkpoint).resolve()
    model, checkpoint = load_checkpoint(str(checkpoint_path), device)
    if model.case_classifier is None:
        raise ValueError("checkpoint has no case-classification head")
    image_size = args.image_size or int(checkpoint.get("config", {}).get("image_size", 384))
    samples = [
        sample
        for sample in load_manifest(args.manifest, require_commercial=True)
        if sample.split == args.split and sample.damage_supervised
    ]
    if not samples:
        raise ValueError(f"no supervised samples in split {args.split!r}")

    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index, sample in enumerate(samples, 1):
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
        with Image.open(sample.mask) as source:
            target = np.asarray(source.convert("L"))
        scores.append(
            predict_case_probabilities(
                model,
                image,
                device=device,
                image_size=image_size,
                horizontal_flip_tta=not args.no_tta,
            ).numpy()
        )
        targets.append(
            np.asarray([(target == class_id).any() for class_id in range(1, len(checkpoint["classes"]))])
        )
        print(json.dumps({"scored": index, "total": len(samples), "image": str(sample.image)}), flush=True)

    score_matrix = np.stack(scores)
    target_matrix = np.stack(targets)
    class_results: dict[str, dict] = {}
    for offset, name in enumerate(checkpoint["classes"][1:]):
        positive = score_matrix[target_matrix[:, offset], offset]
        negative = score_matrix[~target_matrix[:, offset], offset]
        threshold = _threshold_for_recall(positive, args.target_recall)
        tp = int((positive >= threshold).sum())
        fn = int((positive < threshold).sum())
        fp = int((negative >= threshold).sum())
        tn = int((negative < threshold).sum())
        class_results[name] = {
            "threshold": threshold,
            "positive_cases": int(positive.size),
            "negative_cases": int(negative.size),
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "true_negative": tn,
            "recall": tp / (tp + fn),
            "precision": tp / (tp + fp) if tp + fp else None,
            "absent_case_false_alert_rate": fp / (fp + tn) if fp + tn else None,
            "auroc": _auroc(positive, negative),
        }

    report = {
        "status": "development_calibration_only",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "manifest": str(Path(args.manifest).resolve()),
        "split": args.split,
        "target_recall": args.target_recall,
        "image_size": image_size,
        "horizontal_flip_tta": not args.no_tta,
        "classes": class_results,
    }
    rendered = json.dumps(report, indent=2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
