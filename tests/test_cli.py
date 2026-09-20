from argparse import Namespace

import pytest

from vehicle_damage.calibration import Calibration
from vehicle_damage.cli import _frozen_inference_settings


def test_inference_settings_are_frozen_with_threshold():
    calibration = Calibration(
        any_damage_threshold=0.25,
        target_recall=0.97,
        measured_recall=0.98,
        negative_pixel_fpr=0.2,
        positive_pixels=10,
        negative_pixels=20,
        tile_size=768,
        overlap=192,
        horizontal_flip_tta=True,
    )
    assert _frozen_inference_settings(
        calibration, Namespace(tile_size=None, overlap=None, no_tta=False)
    ) == (768, 192, True, False)

    with pytest.raises(ValueError, match="conflicts"):
        _frozen_inference_settings(
            calibration, Namespace(tile_size=640, overlap=None, no_tta=False)
        )
    with pytest.raises(ValueError, match="conflicts"):
        _frozen_inference_settings(
            calibration, Namespace(tile_size=None, overlap=None, no_tta=True)
        )
