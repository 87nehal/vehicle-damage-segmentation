import numpy as np
import pytest

from vehicle_damage.calibration import Calibration, calibrate_case_threshold, calibrate_threshold


def test_calibration_meets_target_recall():
    probability = np.array([0.95, 0.85, 0.75, 0.65, 0.20, 0.10, 0.05, 0.01])
    target = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    result = calibrate_threshold(probability, target, target_recall=0.75)
    assert result.measured_recall >= 0.75
    assert result.any_damage_threshold == 0.75
    assert result.negative_pixel_fpr == 0.0


def test_calibration_requires_both_classes():
    with np.testing.assert_raises(ValueError):
        calibrate_threshold(np.array([0.1, 0.2]), np.array([1, 1]))


def test_case_calibration_weights_cases_equally():
    result = calibrate_case_threshold(
        np.array([0.9, 0.8, 0.4, 0.2]),
        np.array([0.3, 0.1, 0.05]),
        target_recall=0.75,
        positive_pixels=123,
        minimum_damage_coverage=0.05,
        exterior_floor=0.75,
        tile_size=640,
        overlap=160,
        horizontal_flip_tta=False,
    )
    assert result.any_damage_threshold == 0.4
    assert result.measured_recall == 0.75
    assert result.calibration_basis == "damage_case"
    assert result.positive_cases == 4
    assert result.exterior_floor == 0.75
    assert result.tile_size == 640
    assert result.overlap == 160
    assert not result.horizontal_flip_tta


def test_dual_threshold_requires_a_higher_documented_segmentation_threshold(tmp_path):
    profile = Calibration(
        any_damage_threshold=0.4,
        target_recall=0.97,
        measured_recall=1.0,
        negative_pixel_fpr=0.1,
        positive_pixels=10,
        negative_pixels=90,
        segmentation_threshold=0.7,
        segmentation_threshold_basis="validation region F2",
    )
    path = tmp_path / "dual.json"
    profile.save(path)
    loaded = Calibration.load(path)
    assert loaded.effective_segmentation_threshold == 0.7

    with pytest.raises(ValueError, match="at least"):
        Calibration(
            any_damage_threshold=0.7,
            target_recall=0.97,
            measured_recall=1.0,
            negative_pixel_fpr=0.1,
            positive_pixels=10,
            negative_pixels=90,
            segmentation_threshold=0.4,
            segmentation_threshold_basis="invalid",
        )


def test_calibration_round_trips_distinct_triage_and_segmentation_scales(tmp_path):
    profile = Calibration(
        any_damage_threshold=0.4,
        target_recall=0.97,
        measured_recall=1.0,
        negative_pixel_fpr=0.1,
        positive_pixels=10,
        negative_pixels=90,
        triage_scales=(1.0, 1.5),
        segmentation_scales=(1.0,),
        segmentation_threshold=0.7,
        segmentation_threshold_basis="validation region F2",
    )
    path = tmp_path / "scales.json"
    profile.save(path)
    loaded = Calibration.load(path)
    assert loaded.triage_scales == (1.0, 1.5)
    assert loaded.effective_segmentation_scales == (1.0,)

    with pytest.raises(ValueError, match="unique positive finite"):
        Calibration(
            any_damage_threshold=0.4,
            target_recall=0.97,
            measured_recall=1.0,
            negative_pixel_fpr=0.1,
            positive_pixels=10,
            negative_pixels=90,
            triage_scales=(1.0, 1.0),
        )


def test_calibration_freezes_distinct_segmentation_component_filter(tmp_path):
    profile = Calibration(
        any_damage_threshold=0.4,
        target_recall=0.97,
        measured_recall=1.0,
        negative_pixel_fpr=0.1,
        positive_pixels=10,
        negative_pixels=90,
        minimum_component_pixels=0,
        segmentation_minimum_component_pixels=32,
        segmentation_component_filter_basis="calibration recall guard",
    )
    path = tmp_path / "component-filter.json"
    profile.save(path)
    loaded = Calibration.load(path)
    assert loaded.minimum_component_pixels == 0
    assert loaded.effective_segmentation_minimum_component_pixels == 32

    legacy = Calibration(
        any_damage_threshold=0.4,
        target_recall=0.97,
        measured_recall=1.0,
        negative_pixel_fpr=0.1,
        positive_pixels=10,
        negative_pixels=90,
        minimum_component_pixels=16,
    )
    assert legacy.effective_segmentation_minimum_component_pixels == 16

    with pytest.raises(ValueError, match="non-negative"):
        Calibration(
            any_damage_threshold=0.4,
            target_recall=0.97,
            measured_recall=1.0,
            negative_pixel_fpr=0.1,
            positive_pixels=10,
            negative_pixels=90,
            segmentation_minimum_component_pixels=-1,
            segmentation_component_filter_basis="invalid",
        )

    with pytest.raises(ValueError, match="selection basis"):
        Calibration(
            any_damage_threshold=0.4,
            target_recall=0.97,
            measured_recall=1.0,
            negative_pixel_fpr=0.1,
            positive_pixels=10,
            negative_pixels=90,
            segmentation_minimum_component_pixels=32,
        )


def test_calibration_freezes_positive_documented_type_multipliers(tmp_path):
    profile = Calibration(
        any_damage_threshold=0.4,
        target_recall=0.97,
        measured_recall=1.0,
        negative_pixel_fpr=0.1,
        positive_pixels=10,
        negative_pixels=90,
        type_probability_multipliers=(1.2, 0.9),
        type_probability_multiplier_basis="calibration Pareto search",
    )
    path = tmp_path / "type-bias.json"
    profile.save(path)
    loaded = Calibration.load(path)
    assert loaded.type_probability_multipliers == (1.2, 0.9)

    with pytest.raises(ValueError, match="positive finite"):
        Calibration(
            any_damage_threshold=0.4,
            target_recall=0.97,
            measured_recall=1.0,
            negative_pixel_fpr=0.1,
            positive_pixels=10,
            negative_pixels=90,
            type_probability_multipliers=(1.0, 0.0),
            type_probability_multiplier_basis="invalid",
        )
    with pytest.raises(ValueError, match="selection basis"):
        Calibration(
            any_damage_threshold=0.4,
            target_recall=0.97,
            measured_recall=1.0,
            negative_pixel_fpr=0.1,
            positive_pixels=10,
            negative_pixels=90,
            type_probability_multipliers=(1.0, 1.0),
        )
