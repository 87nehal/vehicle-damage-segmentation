import numpy as np

from vehicle_damage.hard_negative import combine_hard_negative_masks, select_hard_negative_mask


def test_hard_negative_selection_respects_fraction_and_damage_guard():
    score = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
    target = np.zeros((10, 10), dtype=np.uint8)
    target[4:6, 4:6] = 1
    selected = select_hard_negative_mask(
        score, target, minimum_score=0.2, maximum_fraction=0.10, guard_radius=1
    )
    assert set(np.unique(selected)) <= {0, 255}
    assert 0 < (selected > 0).sum() <= 10
    assert not selected[3:7, 3:7].any()


def test_hard_negative_selection_rejects_invalid_fraction():
    with np.testing.assert_raises(ValueError):
        select_hard_negative_mask(
            np.zeros((2, 2)), np.zeros((2, 2)), minimum_score=0.5,
            maximum_fraction=0.0,
        )


def test_hard_negative_mask_union_preserves_both_sources():
    first = np.zeros((4, 4), dtype=np.uint8)
    second = np.zeros_like(first)
    first[0, 0] = 255
    second[3, 3] = 1

    combined = combine_hard_negative_masks(first, second)

    assert combined[0, 0] == 255
    assert combined[3, 3] == 255
    assert int((combined > 0).sum()) == 2
