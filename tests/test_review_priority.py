import numpy as np
import pytest

from vehicle_damage.review_priority import (
    rank_review_records,
    summarize_review_candidate,
)


def test_review_priority_favors_broad_precise_predictions():
    low = summarize_review_candidate(
        np.full((10, 10), 0.44),
        np.zeros((10, 10), dtype=np.uint8),
        triage_threshold=0.45,
        uncertainty_margin=0.05,
    )
    high = summarize_review_candidate(
        np.full((10, 10), 0.8),
        np.ones((10, 10), dtype=np.uint8),
        triage_threshold=0.45,
        uncertainty_margin=0.05,
    )
    assert high["priority_score"] > low["priority_score"]
    assert high["precise_mask_fraction"] == 1.0
    assert low["near_triage_threshold_fraction"] == 1.0


def test_review_priority_ranking_is_deterministic_and_rejects_duplicates():
    rows = [
        {"image_sha256": "b", "priority_score": 0.5},
        {"image_sha256": "a", "priority_score": 0.5},
        {"image_sha256": "c", "priority_score": 0.7},
    ]
    ranked = rank_review_records(rows)
    assert [row["image_sha256"] for row in ranked] == ["c", "a", "b"]
    assert [row["rank"] for row in ranked] == [1, 2, 3]

    with pytest.raises(ValueError, match="unique"):
        rank_review_records([rows[0], rows[0]])


def test_review_priority_rejects_misaligned_or_invalid_inputs():
    with pytest.raises(ValueError, match="aligned"):
        summarize_review_candidate(
            np.zeros((2, 2)),
            np.zeros((3, 3)),
            triage_threshold=0.45,
            uncertainty_margin=0.05,
        )
    with pytest.raises(ValueError, match="finite"):
        summarize_review_candidate(
            np.array([[np.nan]], dtype=np.float32),
            np.zeros((1, 1)),
            triage_threshold=0.45,
            uncertainty_margin=0.05,
        )
