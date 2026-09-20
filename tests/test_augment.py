import random

import numpy as np
import pytest
from PIL import Image

from vehicle_damage.augment import (
    RobustAugment,
    _degrade_resolution,
    _random_viewpoint,
    _sensor_noise,
    _synthetic_shadow,
)


def test_positive_crop_keeps_damage_when_forced():
    random.seed(11)
    image = Image.new("RGB", (512, 512), "gray")
    mask_array = np.zeros((512, 512), dtype=np.uint8)
    mask_array[460:470, 460:470] = 1
    mask = Image.fromarray(mask_array)
    transform = RobustAugment(128, training=True, positive_crop_probability=1.0)
    _, transformed, _, _ = transform(
        image, mask, Image.new("L", image.size, 255), Image.new("L", image.size, 0)
    )
    assert (transformed > 0).any()


def test_positive_crop_can_target_tiny_class():
    random.seed(3)
    image = Image.new("RGB", (512, 512), "gray")
    mask_array = np.ones((512, 512), dtype=np.uint8)
    mask_array[490:492, 490:492] = 2
    mask = Image.fromarray(mask_array)
    transform = RobustAugment(64, training=True, positive_crop_probability=1.0)
    seen_tiny = False
    for _ in range(20):
        _, transformed, _, _ = transform(
            image, mask, Image.new("L", image.size, 255), Image.new("L", image.size, 0)
        )
        seen_tiny |= bool((transformed == 2).any())
    assert seen_tiny


def test_viewpoint_transform_keeps_all_supervision_masks_aligned():
    array = np.zeros((96, 96), dtype=np.uint8)
    array[20:75, 30:70] = 1
    mask = Image.fromarray(array)
    items = [Image.new("RGB", mask.size, "gray"), mask, mask.copy(), mask.copy()]

    transformed = _random_viewpoint(items, random.Random(19))

    damage = np.asarray(transformed[1])
    assert damage.any()
    np.testing.assert_array_equal(damage, np.asarray(transformed[2]))
    np.testing.assert_array_equal(damage, np.asarray(transformed[3]))
    assert set(np.unique(damage)).issubset({0, 1})


def test_shadow_and_sensor_noise_preserve_image_contract():
    image = Image.new("RGB", (128, 96), (140, 150, 160))
    shadowed = _synthetic_shadow(image, random.Random(5))
    noisy = _sensor_noise(shadowed, random.Random(7))

    assert shadowed.mode == "RGB"
    assert noisy.mode == "RGB"
    assert noisy.size == image.size
    assert not np.array_equal(np.asarray(noisy), np.asarray(image))


def test_augmentation_profile_fails_closed_on_unknown_version():
    with pytest.raises(ValueError, match="unknown augmentation profile"):
        RobustAugment(128, profile="future_unreviewed")


def test_resolution_degradation_is_deterministic_and_preserves_geometry():
    array = np.zeros((96, 128, 3), dtype=np.uint8)
    array[::2, ::2] = 255
    image = Image.fromarray(array)
    first = _degrade_resolution(image, random.Random(11))
    second = _degrade_resolution(image, random.Random(11))

    assert first.size == image.size
    assert first.mode == "RGB"
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), np.asarray(image))


def test_robust_v3_full_scene_path_keeps_damage(monkeypatch):
    sequence = iter([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    monkeypatch.setattr(random, "random", lambda: next(sequence, 1.0))
    image = Image.new("RGB", (640, 480), "gray")
    mask_array = np.zeros((480, 640), dtype=np.uint8)
    mask_array[200:240, 300:350] = 1
    mask = Image.fromarray(mask_array)
    transform = RobustAugment(196, training=True, profile="robust_v3")

    _, transformed, _, _ = transform(
        image, mask, Image.new("L", image.size, 255), Image.new("L", image.size, 0)
    )

    assert transformed.shape == (196, 196)
    assert (transformed == 1).any()
