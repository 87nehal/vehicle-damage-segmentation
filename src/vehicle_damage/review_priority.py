from __future__ import annotations

import math

import numpy as np


def summarize_review_candidate(
    triage_score: np.ndarray,
    precise_mask: np.ndarray,
    *,
    triage_threshold: float,
    uncertainty_margin: float,
    quality_review_reasons: tuple[str, ...] | list[str] = (),
) -> dict[str, float | int]:
    """Summarize an unverified image for review ordering, never as a label."""
    score = np.asarray(triage_score, dtype=np.float32)
    mask = np.asarray(precise_mask)
    if score.ndim != 2 or mask.shape != score.shape:
        raise ValueError("triage_score and precise_mask must be aligned 2D arrays")
    if score.size == 0 or not np.isfinite(score).all():
        raise ValueError("triage_score must contain finite pixels")
    if not 0 <= triage_threshold <= 1:
        raise ValueError("triage_threshold must be in [0, 1]")
    if not 0 < uncertainty_margin <= 1:
        raise ValueError("uncertainty_margin must be in (0, 1]")

    triage_fraction = float((score >= triage_threshold).mean())
    precise_fraction = float((mask > 0).mean())
    near_threshold_fraction = float(
        (np.abs(score - triage_threshold) <= uncertainty_margin).mean()
    )
    peak = float(score.max())
    quantile_99 = float(np.quantile(score, 0.99))
    quality_count = len(tuple(quality_review_reasons))

    # Broad precise predictions expose likely false-positive texture; square
    # roots also keep small, high-value predicted regions near the front.
    priority_score = (
        3.0 * math.sqrt(precise_fraction)
        + math.sqrt(triage_fraction)
        + 0.25 * quantile_99
        + 0.15 * near_threshold_fraction
        + 0.05 * min(quality_count, 4)
    )
    return {
        "priority_score": priority_score,
        "triage_peak_score": peak,
        "triage_score_quantile_99": quantile_99,
        "triage_positive_fraction": triage_fraction,
        "precise_mask_fraction": precise_fraction,
        "near_triage_threshold_fraction": near_threshold_fraction,
        "quality_review_reason_count": quality_count,
    }


def rank_review_records(records: list[dict]) -> list[dict]:
    """Return deterministic highest-value-first review records."""
    if any("image_sha256" not in row or "priority_score" not in row for row in records):
        raise ValueError("review records require image_sha256 and priority_score")
    if len({str(row["image_sha256"]) for row in records}) != len(records):
        raise ValueError("review records must have unique image_sha256 values")
    ordered = sorted(
        records,
        key=lambda row: (-float(row["priority_score"]), str(row["image_sha256"])),
    )
    return [{**row, "rank": index} for index, row in enumerate(ordered, 1)]
