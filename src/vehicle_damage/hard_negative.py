from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter


def select_hard_negative_mask(
    score: np.ndarray,
    target: np.ndarray,
    *,
    minimum_score: float,
    maximum_fraction: float = 0.10,
    guard_radius: int = 5,
) -> np.ndarray:
    """Select the highest-scoring known-background pixels for retraining.

    A guard band around annotated damage avoids teaching uncertain polygon
    boundaries as negatives. The fraction is measured over eligible
    background, and the returned uint8 mask uses values 0 and 255.
    """
    score = np.asarray(score, dtype=np.float32)
    target = np.asarray(target)
    if score.shape != target.shape or score.ndim != 2:
        raise ValueError("score and target must be same-shape 2D arrays")
    if not np.isfinite(score).all():
        raise ValueError("score must contain only finite values")
    if not 0 <= minimum_score <= 1:
        raise ValueError("minimum_score must be in [0, 1]")
    if not 0 < maximum_fraction <= 1:
        raise ValueError("maximum_fraction must be in (0, 1]")
    if guard_radius < 0:
        raise ValueError("guard_radius cannot be negative")

    damage = target > 0
    if guard_radius:
        guard_image = Image.fromarray(damage.astype(np.uint8) * 255, mode="L")
        guarded = np.asarray(
            guard_image.filter(ImageFilter.MaxFilter(2 * guard_radius + 1))
        ) > 0
    else:
        guarded = damage
    eligible = ~guarded
    eligible_count = int(eligible.sum())
    selected = np.zeros(score.shape, dtype=np.uint8)
    if eligible_count == 0:
        return selected

    candidates = np.flatnonzero((eligible & (score >= minimum_score)).ravel())
    limit = max(1, math.ceil(maximum_fraction * eligible_count))
    if candidates.size > limit:
        candidate_scores = score.ravel()[candidates]
        top = np.argpartition(candidate_scores, -limit)[-limit:]
        candidates = candidates[top]
    selected.ravel()[candidates] = 255
    return selected


def combine_hard_negative_masks(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Union two same-resolution hard-negative masks without mutating either."""
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("hard-negative masks must be same-shape 2D arrays")
    return ((first > 0) | (second > 0)).astype(np.uint8) * 255
