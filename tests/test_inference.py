import numpy as np
import torch
import pytest
from PIL import Image, ImageDraw

from vehicle_damage.inference import (
    QualityReport,
    assess_quality,
    filter_small_damage_components,
    fuse_probability_pyramid,
    predict_case_probabilities,
    predict_multiscale_probabilities,
    predict_profile_probabilities,
    predict_probability_pyramid,
    predict_probabilities,
    probabilities_to_mask,
    route_inference_decision,
)


def test_small_component_filter_preserves_classes_and_removes_islands():
    mask = torch.zeros((10, 10), dtype=torch.uint8).numpy()
    mask[1:4, 1:4] = 2
    mask[8, 8] = 1

    filtered = filter_small_damage_components(mask, minimum_pixels=4)

    assert (filtered[1:4, 1:4] == 2).all()
    assert filtered[8, 8] == 0
    assert mask[8, 8] == 1


class DummyModel(torch.nn.Module):
    def forward(self, image):
        batch, _, height, width = image.shape
        presence = torch.full((batch, 1, height, width), 3.0, device=image.device)
        damage_type = torch.zeros(batch, 2, height, width, device=image.device)
        damage_type[:, 0] = 2.0
        exterior = torch.full((batch, 1, height, width), 3.0, device=image.device)
        return {"presence": presence, "type": damage_type, "exterior": exterior}


class DummyCaseModel(DummyModel):
    def forward(self, image):
        result = super().forward(image)
        result["case_logits"] = torch.tensor([[2.0, -2.0]], device=image.device).repeat(
            image.shape[0], 1
        )
        return result


class DummyRoleSplitModel(DummyModel):
    def __init__(self):
        super().__init__()
        self.triage_presence_head = object()

    def forward(self, image):
        result = super().forward(image)
        batch, _, height, width = image.shape
        result["triage_presence"] = torch.full(
            (batch, 1, height, width), 3.0, device=image.device
        )
        triage_type = torch.zeros(batch, 2, height, width, device=image.device)
        triage_type[:, 1] = 4.0
        result["triage_type"] = triage_type
        result["triage_exterior"] = torch.full(
            (batch, 1, height, width), 3.0, device=image.device
        )
        return result


class SelectiveDummyRoleSplitModel(DummyRoleSplitModel):
    def __init__(self):
        super().__init__()
        self.requested_branches = []

    def forward_roles(self, image, *, include_main=True, include_triage=True):
        self.requested_branches.append((include_main, include_triage))
        result = super().forward(image)
        if not include_main:
            for key in ("presence", "type", "exterior"):
                result.pop(key)
        if not include_triage:
            for key in ("triage_presence", "triage_type", "triage_exterior"):
                result.pop(key)
        return result


def test_tiled_inference_preserves_shape():
    image = Image.new("RGB", (91, 73), "gray")
    probability, exterior = predict_probabilities(
        DummyModel(), image, device="cpu", tile_size=48, overlap=12, horizontal_flip_tta=True
    )
    assert probability.shape == (3, 73, 91)
    assert exterior.shape == (73, 91)
    mask, uncertainty = probabilities_to_mask(probability, exterior, threshold=0.5)
    assert mask.shape == (73, 91)
    assert uncertainty.shape == (73, 91)
    assert (mask == 1).all()


def test_multiscale_inference_aligns_outputs_to_original_shape():
    image = Image.new("RGB", (91, 73), "gray")
    probability, exterior = predict_multiscale_probabilities(
        DummyModel(),
        image,
        device="cpu",
        scale_factors=(1.0, 1.5),
        tile_size=48,
        overlap=12,
        horizontal_flip_tta=False,
    )
    assert probability.shape == (3, 73, 91)
    assert exterior.shape == (73, 91)
    assert torch.allclose(probability.sum(dim=0), torch.ones((73, 91)), atol=1e-5)

    pyramid = predict_probability_pyramid(
        DummyModel(),
        image,
        device="cpu",
        scale_factors=(1.0, 1.5),
        tile_size=48,
        overlap=12,
        horizontal_flip_tta=False,
    )
    native_probability, native_exterior = fuse_probability_pyramid(pyramid, (1.0,))
    assert native_probability.shape == probability.shape
    assert native_exterior.shape == exterior.shape
    with pytest.raises(ValueError, match="missing scales"):
        fuse_probability_pyramid(pyramid, (2.0,))

    (triage_probability, _), (mask_probability, _) = predict_profile_probabilities(
        DummyModel(),
        image,
        device="cpu",
        triage_scales=(1.0, 1.5),
        segmentation_scales=(1.0,),
        tile_size=48,
        overlap=12,
        horizontal_flip_tta=False,
    )
    assert triage_probability.shape == mask_probability.shape == probability.shape


def test_profile_inference_uses_role_split_triage_heads_in_one_path():
    image = Image.new("RGB", (91, 73), "gray")
    (triage, _), (segmentation, _) = predict_profile_probabilities(
        DummyRoleSplitModel(),
        image,
        device="cpu",
        triage_scales=(1.0, 1.5),
        segmentation_scales=(1.0,),
        tile_size=48,
        overlap=12,
        horizontal_flip_tta=True,
    )
    assert (triage[2] > triage[1]).all()
    assert (segmentation[1] > segmentation[2]).all()


def test_profile_inference_skips_unused_role_split_heads_by_scale():
    image = Image.new("RGB", (91, 73), "gray")
    model = SelectiveDummyRoleSplitModel()
    predict_profile_probabilities(
        model,
        image,
        device="cpu",
        triage_scales=(1.0, 1.5),
        segmentation_scales=(1.0,),
        tile_size=48,
        overlap=12,
        horizontal_flip_tta=False,
    )
    assert (True, True) in model.requested_branches
    assert (False, True) in model.requested_branches
    assert (True, False) not in model.requested_branches


def test_multiscale_inference_rejects_invalid_or_duplicate_scales():
    image = Image.new("RGB", (32, 32), "gray")
    for scales in ((), (0.0,), (1.0, 1.0)):
        with pytest.raises(ValueError, match="scale_factors"):
            predict_multiscale_probabilities(
                DummyModel(), image, device="cpu", scale_factors=scales
            )


def test_exterior_floor_one_disables_exterior_suppression():
    probability = torch.tensor([[[0.4]], [[0.6]], [[0.0]]])
    exterior = torch.tensor([[0.0]])
    suppressed, _ = probabilities_to_mask(probability, exterior, threshold=0.5, exterior_floor=0.5)
    unsuppressed, _ = probabilities_to_mask(probability, exterior, threshold=0.5, exterior_floor=1.0)
    assert suppressed.item() == 0
    assert unsuppressed.item() == 1


def test_probability_conversion_applies_frozen_component_filter():
    probability = torch.zeros((2, 5, 5))
    probability[0] = 1.0
    probability[0, 2, 2] = 0.0
    probability[1, 2, 2] = 1.0
    exterior = torch.ones((5, 5))

    filtered, _ = probabilities_to_mask(
        probability,
        exterior,
        threshold=0.5,
        minimum_component_pixels=2,
    )

    assert not filtered.any()


def test_type_multipliers_change_only_the_damage_type_assignment():
    probability = torch.tensor([[[0.2]], [[0.45]], [[0.35]]])
    exterior = torch.ones((1, 1))
    baseline, _ = probabilities_to_mask(
        probability, exterior, threshold=0.5, exterior_floor=1.0
    )
    biased, _ = probabilities_to_mask(
        probability,
        exterior,
        threshold=0.5,
        exterior_floor=1.0,
        type_probability_multipliers=(1.0, 2.0),
    )
    assert baseline.item() == 1
    assert biased.item() == 2
    assert bool(baseline) == bool(biased)

    with pytest.raises(ValueError, match="match the number"):
        probabilities_to_mask(
            probability,
            exterior,
            threshold=0.5,
            type_probability_multipliers=(1.0,),
        )


def test_case_classifier_returns_independent_probabilities():
    image = Image.new("RGB", (91, 73), "gray")
    probability = predict_case_probabilities(
        DummyCaseModel(), image, device="cpu", image_size=48
    )
    assert probability.shape == (2,)
    assert probability[0] > 0.8
    assert probability[1] < 0.2


def test_localized_specular_highlight_routes_image_to_review():
    image = Image.new("RGB", (640, 480), (80, 80, 80))
    ImageDraw.Draw(image).rectangle((100, 100, 220, 220), fill=(255, 255, 250))
    report = assess_quality(image)
    assert report.specular_highlight_fraction > 0.015
    assert "possible_specular_glare" in report.review_reasons


def _quality(*reasons):
    return QualityReport(
        width=640,
        height=480,
        dark_fraction=0.0,
        clipped_highlight_fraction=0.0,
        specular_highlight_fraction=0.0,
        sharpness=0.01,
        review_reasons=tuple(reasons),
    )


def test_decision_router_allows_only_unambiguous_automatic_results():
    clean = np.zeros((2, 3), dtype=np.uint8)
    no_damage = route_inference_decision(
        _quality(),
        clean,
        clean,
        near_threshold_fraction=0.01,
        max_uncertain_fraction=0.15,
    )
    assert no_damage.decision == "no_damage_detected"
    assert no_damage.automated_decision_allowed
    assert not no_damage.manual_review

    damage = clean.copy()
    damage[0, 0] = 2
    detected = route_inference_decision(
        _quality(),
        damage,
        damage,
        near_threshold_fraction=0.01,
        max_uncertain_fraction=0.15,
    )
    assert detected.decision == "damage_detected"
    assert detected.automated_decision_allowed


def test_decision_router_exposes_every_branch_disagreement():
    triage = np.array([[1, 2, 0], [0, 0, 0]], dtype=np.uint8)
    segmentation = np.array([[0, 3, 4], [0, 0, 0]], dtype=np.uint8)
    result = route_inference_decision(
        _quality(),
        triage,
        segmentation,
        near_threshold_fraction=0.20,
        max_uncertain_fraction=0.15,
    )
    assert result.decision == "manual_review_required"
    assert not result.automated_decision_allowed
    assert result.manual_review
    assert result.manual_review_required
    assert not result.recapture_required
    assert result.triage_only_fraction == pytest.approx(1 / 6)
    assert result.segmentation_only_fraction == pytest.approx(1 / 6)
    assert result.damage_type_disagreement_fraction == pytest.approx(1 / 6)
    assert result.decision_reasons == (
        "triage_only_damage_evidence",
        "segmentation_only_damage_evidence",
        "damage_type_disagreement",
        "excessive_near_threshold_area",
    )


def test_decision_router_separates_glare_review_from_failed_capture():
    mask = np.zeros((2, 2), dtype=np.uint8)
    glare = route_inference_decision(
        _quality("possible_specular_glare"),
        mask,
        mask,
        near_threshold_fraction=0.0,
        max_uncertain_fraction=0.15,
    )
    assert glare.decision == "manual_review_required"
    assert glare.decision_reasons == ("quality:possible_specular_glare",)

    failed = route_inference_decision(
        _quality("possible_specular_glare", "blur_or_low_detail"),
        mask,
        mask,
        near_threshold_fraction=0.0,
        max_uncertain_fraction=0.15,
    )
    assert failed.decision == "recapture_required"
    assert failed.recapture_required
    assert failed.manual_review
    assert not failed.manual_review_required
    assert failed.recapture_reasons == ("blur_or_low_detail",)


@pytest.mark.parametrize(
    ("near_fraction", "maximum"),
    [(-0.1, 0.15), (1.1, 0.15), (float("nan"), 0.15), (0.1, -0.1)],
)
def test_decision_router_rejects_invalid_uncertainty_inputs(near_fraction, maximum):
    mask = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="fraction"):
        route_inference_decision(
            _quality(),
            mask,
            mask,
            near_threshold_fraction=near_fraction,
            max_uncertain_fraction=maximum,
        )


def test_mixed_precision_fails_closed_without_cuda():
    image = Image.new("RGB", (91, 73), "gray")
    with pytest.raises(ValueError, match="requires a CUDA"):
        predict_probabilities(
            DummyModel(),
            image,
            device="cpu",
            tile_size=48,
            overlap=12,
            mixed_precision=True,
        )
