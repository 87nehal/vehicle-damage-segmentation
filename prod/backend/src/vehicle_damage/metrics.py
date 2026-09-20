from __future__ import annotations

import math

import torch


def confusion_matrix(prediction: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    valid = (target >= 0) & (target < num_classes)
    indices = target[valid].to(torch.int64) * num_classes + prediction[valid].to(torch.int64)
    return torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion(matrix: torch.Tensor) -> dict:
    matrix = matrix.to(torch.float64)
    tp = matrix.diag()
    fp = matrix.sum(0) - tp
    fn = matrix.sum(1) - tp
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    iou = tp / (tp + fp + fn).clamp_min(1)
    any_tp = matrix[1:, 1:].sum()
    any_fn = matrix[1:, 0].sum()
    any_fp = matrix[0, 1:].sum()
    any_precision = any_tp / (any_tp + any_fp).clamp_min(1)
    any_recall = any_tp / (any_tp + any_fn).clamp_min(1)
    beta2 = 4.0
    f2 = (1 + beta2) * any_precision * any_recall / (beta2 * any_precision + any_recall).clamp_min(1e-12)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "iou": iou.tolist(),
        "mean_damage_iou": float(iou[1:].mean()),
        "any_damage_precision": float(any_precision),
        "any_damage_recall": float(any_recall),
        "any_damage_f2": float(f2),
    }


def wilson_lower_bound(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (centre - margin) / denominator


def wilson_upper_bound(successes: int, total: int, z: float = 1.959963984540054) -> float:
    """Upper endpoint of a two-sided Wilson binomial confidence interval."""
    if total <= 0:
        return 1.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return min(1.0, (centre + margin) / denominator)


def binary_auroc(scores: torch.Tensor, target: torch.Tensor) -> float | None:
    """Exact AUROC for a small binary validation set, with half credit for ties."""
    values = scores.detach().to(torch.float64).flatten()
    labels = target.detach().bool().flatten()
    if values.shape != labels.shape:
        raise ValueError("scores and target must have the same shape")
    positive = values[labels]
    negative = values[~labels]
    if not positive.numel() or not negative.numel():
        return None
    differences = positive[:, None] - negative[None, :]
    return float(
        ((differences > 0).sum() + 0.5 * (differences == 0).sum()).double()
        / differences.numel()
    )
