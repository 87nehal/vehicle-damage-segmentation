import pytest
import torch

from vehicle_damage.model import (
    DamageSegmenter,
    Dinov2DamageSegmenter,
    Dinov2FourLayerDamageSegmenter,
    Dinov2FourLayerDualTypeDamageSegmenter,
    Dinov2FourLayerSplitDamageSegmenter,
    Dinov2FourLayerTypeRescueDamageSegmenter,
    Dinov2FourLayerRoleSplitDamageSegmenter,
    Dinov2FourLayerDetailRoleSplitDamageSegmenter,
    damage_probabilities,
)


def test_deeplab_model_and_probability_shapes():
    model = DamageSegmenter(6, pretrained_backbone=False).eval()
    output = model(torch.randn(1, 3, 64, 64))
    probability = damage_probabilities(output)

    assert output["presence"].shape == (1, 1, 64, 64)
    assert output["type"].shape == (1, 5, 64, 64)
    assert probability.shape == (1, 6, 64, 64)


def test_dinov2_model_shapes_for_non_multiple_input():
    model = Dinov2DamageSegmenter(6, pretrained_backbone=False).eval()
    output = model(torch.randn(1, 3, 57, 71))

    assert output["presence"].shape == (1, 1, 57, 71)
    assert output["type"].shape == (1, 5, 57, 71)
    assert output["exterior"].shape == (1, 1, 57, 71)


def test_four_layer_dinov2_starts_equivalent_to_single_layer_decoder():
    torch.manual_seed(7)
    baseline = Dinov2DamageSegmenter(6, pretrained_backbone=False).eval()
    fused = Dinov2FourLayerDamageSegmenter(6, pretrained_backbone=False).eval()
    incompatible = fused.load_state_dict(baseline.state_dict(), strict=False)
    assert incompatible.missing_keys == ["feature_fusion.weight"]
    assert not incompatible.unexpected_keys

    image = torch.randn(1, 3, 57, 71)
    with torch.inference_mode():
        baseline_output = baseline(image)
        fused_output = fused(image)
    for key in ("presence", "type", "exterior"):
        torch.testing.assert_close(fused_output[key], baseline_output[key])


def test_split_four_layer_dinov2_starts_equivalent_to_single_layer_decoder():
    baseline = Dinov2DamageSegmenter(6, pretrained_backbone=False).eval()
    split = Dinov2FourLayerSplitDamageSegmenter(6, pretrained_backbone=False).eval()
    source = baseline.state_dict()
    destination = split.state_dict()
    for name, value in source.items():
        destination[name] = value
    split.load_state_dict(destination)

    image = torch.randn(1, 3, 57, 71)
    with torch.inference_mode():
        baseline_output = baseline(image)
        split_output = split(image)
    for key in ("presence", "type", "exterior"):
        torch.testing.assert_close(split_output[key], baseline_output[key])


def test_dual_type_fusion_validates_configuration_and_shapes():
    model = Dinov2FourLayerDualTypeDamageSegmenter(
        6,
        pretrained_backbone=False,
        type_probability_fusion="maximum",
    ).eval()
    output = model(torch.randn(1, 3, 57, 71))
    assert output["type"].shape == (1, 5, 57, 71)

    with pytest.raises(ValueError, match="type_probability_fusion"):
        Dinov2FourLayerDualTypeDamageSegmenter(
            6, pretrained_backbone=False, type_probability_fusion="invalid"
        )

    with pytest.raises(ValueError, match=r"one \[0, 1\] weight"):
        Dinov2FourLayerDualTypeDamageSegmenter(
            6,
            pretrained_backbone=False,
            type_probability_fusion="classwise",
            type_probability_class_weights_four_layer=[1.0, 0.0],
        )


def test_type_rescue_model_shapes_and_configuration():
    model = Dinov2FourLayerTypeRescueDamageSegmenter(
        6, pretrained_backbone=False, rescue_type_index=2, rescue_threshold=0.69
    ).eval()
    output = model(torch.randn(1, 3, 57, 71))
    assert output["presence"].shape == (1, 1, 57, 71)
    assert output["type"].shape == (1, 5, 57, 71)
    assert output["exterior"].shape == (1, 1, 57, 71)

    with pytest.raises(ValueError, match="outside"):
        Dinov2FourLayerTypeRescueDamageSegmenter(
            6, pretrained_backbone=False, rescue_type_index=5
        )


def test_role_split_model_emits_distinct_triage_heads():
    model = Dinov2FourLayerRoleSplitDamageSegmenter(
        6, pretrained_backbone=False
    ).eval()
    image = torch.randn(1, 3, 57, 71)
    output = model(image)
    for key, channels in (
        ("presence", 1),
        ("type", 5),
        ("exterior", 1),
        ("triage_presence", 1),
        ("triage_type", 5),
        ("triage_exterior", 1),
    ):
        assert output[key].shape == (1, channels, 57, 71)

    main_only = model.forward_roles(image, include_triage=False)
    assert set(main_only) == {"presence", "type", "exterior"}
    for key in main_only:
        torch.testing.assert_close(main_only[key], output[key])

    triage_only = model.forward_roles(image, include_main=False)
    assert set(triage_only) == {
        "triage_presence",
        "triage_type",
        "triage_exterior",
    }
    for key in triage_only:
        torch.testing.assert_close(triage_only[key], output[key])

    with pytest.raises(ValueError, match="at least one"):
        model.forward_roles(image, include_main=False, include_triage=False)


def test_role_split_direct_triage_features_equal_identity_projection():
    model = Dinov2FourLayerRoleSplitDamageSegmenter(
        6, pretrained_backbone=False
    ).eval()
    image = torch.randn(1, 3, 56, 70)
    with torch.inference_mode():
        levels = model._extract_intermediate_features(image)
        projected = model.triage_feature_fusion(torch.cat(levels, dim=1))
    torch.testing.assert_close(projected, levels[-1], rtol=0, atol=0)


def test_detail_role_split_starts_exactly_equivalent_and_bypasses_triage():
    torch.manual_seed(23)
    parent = Dinov2FourLayerRoleSplitDamageSegmenter(
        6, pretrained_backbone=False
    ).eval()
    detail = Dinov2FourLayerDetailRoleSplitDamageSegmenter(
        6, pretrained_backbone=False
    ).eval()
    incompatible = detail.load_state_dict(parent.state_dict(), strict=False)
    assert incompatible.missing_keys
    assert all(
        key.startswith(("detail_encoder.", "detail_refiner."))
        for key in incompatible.missing_keys
    )
    assert not incompatible.unexpected_keys

    image = torch.randn(1, 3, 57, 71)
    with torch.inference_mode():
        parent_output = parent(image)
        detail_output = detail(image)
        triage_only = detail.forward_roles(image, include_main=False)
    for key in (
        "presence",
        "type",
        "exterior",
        "triage_presence",
        "triage_type",
        "triage_exterior",
    ):
        torch.testing.assert_close(detail_output[key], parent_output[key], rtol=0, atol=0)
    assert set(triage_only) == {
        "triage_presence",
        "triage_type",
        "triage_exterior",
    }
    for key in triage_only:
        torch.testing.assert_close(triage_only[key], parent_output[key], rtol=0, atol=0)


def test_noncommercial_dinov2_variants_are_rejected():
    with pytest.raises(ValueError, match="reviewed allowlist"):
        Dinov2DamageSegmenter(6, backbone_name="xray_dinov2", pretrained_backbone=False)

    with pytest.raises(ValueError, match="reviewed allowlist"):
        Dinov2DamageSegmenter(6, backbone_name="unknown_dinov2", pretrained_backbone=False)
