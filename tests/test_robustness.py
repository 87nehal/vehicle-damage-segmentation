import numpy as np
import pytest
from PIL import Image

from vehicle_damage.robustness import (
    STRESS_CONDITIONS,
    STRESS_DESCRIPTIONS,
    apply_stress,
    stress_seed,
)


def _inputs():
    image_array = np.zeros((60, 80, 3), dtype=np.uint8)
    image_array[..., 0] = np.arange(80, dtype=np.uint8)
    image_array[..., 1] = np.arange(60, dtype=np.uint8)[:, None]
    image_array[..., 2] = 140
    mask_array = np.zeros((60, 80), dtype=np.uint8)
    mask_array[15:45, 25:55] = 3
    return Image.fromarray(image_array), Image.fromarray(mask_array)


@pytest.mark.parametrize("condition", STRESS_CONDITIONS)
def test_stress_conditions_are_deterministic_and_label_safe(condition):
    image, mask = _inputs()
    first_image, first_mask = apply_stress(image, mask, condition, seed=1234)
    second_image, second_mask = apply_stress(image, mask, condition, seed=1234)

    assert first_image.mode == "RGB"
    assert first_mask.mode == "L"
    assert first_image.size == image.size == first_mask.size
    np.testing.assert_array_equal(np.asarray(first_image), np.asarray(second_image))
    np.testing.assert_array_equal(np.asarray(first_mask), np.asarray(second_mask))
    assert set(np.unique(first_mask)).issubset({0, 3})


def test_photometric_conditions_do_not_move_labels():
    image, mask = _inputs()
    geometric = {"viewpoint", "distance"}
    for condition in set(STRESS_CONDITIONS) - geometric:
        _, stressed_mask = apply_stress(image, mask, condition, seed=17)
        np.testing.assert_array_equal(np.asarray(stressed_mask), np.asarray(mask))


def test_stress_seed_is_stable_and_condition_specific():
    assert stress_seed("sample", "glare") == stress_seed("sample", "glare")
    assert stress_seed("sample", "glare") != stress_seed("sample", "shadow")


def test_every_stress_condition_has_a_description():
    assert set(STRESS_DESCRIPTIONS) == set(STRESS_CONDITIONS)


def test_unknown_condition_fails_closed():
    image, mask = _inputs()
    with pytest.raises(ValueError, match="unknown stress condition"):
        apply_stress(image, mask, "future-condition", seed=1)
