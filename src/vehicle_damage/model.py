from __future__ import annotations

import copy

import torch
import timm
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


APPROVED_DINOV2_BACKBONES = {"vit_small_patch14_dinov2.lvd142m"}


def _identity_four_layer_projection(feature_dim: int) -> nn.Conv2d:
    projection = nn.Conv2d(4 * feature_dim, feature_dim, kernel_size=1, bias=False)
    with torch.no_grad():
        projection.weight.zero_()
        channels = torch.arange(feature_dim)
        projection.weight[channels, 3 * feature_dim + channels, 0, 0] = 1.0
    return projection


class DamageSegmenter(nn.Module):
    """DeepLabV3 with presence, damage-type, and vehicle-exterior heads.

    Pretrained weights are opt-in because code licenses do not automatically
    establish commercial rights to the data used to create a checkpoint.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        pretrained_backbone: bool = False,
        case_classifier: bool = False,
        case_pooling: str = "avg",
    ) -> None:
        super().__init__()
        backbone_weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        base = deeplabv3_resnet50(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
            aux_loss=False,
        )
        self.backbone = base.backbone
        self.presence_head = DeepLabHead(2048, 1)
        self.type_head = DeepLabHead(2048, num_classes - 1)
        self.exterior_head = DeepLabHead(2048, 1)
        if case_pooling not in {"avg", "avg_max"}:
            raise ValueError("case_pooling must be 'avg' or 'avg_max'")
        self.case_pooling = case_pooling
        case_features = 4096 if case_pooling == "avg_max" else 2048
        self.case_classifier = (
            nn.Linear(case_features, num_classes - 1) if case_classifier else None
        )
        self.num_classes = num_classes

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        size = images.shape[-2:]
        features = self.backbone(images)["out"]
        presence = F.interpolate(self.presence_head(features), size=size, mode="bilinear", align_corners=False)
        damage_type = F.interpolate(self.type_head(features), size=size, mode="bilinear", align_corners=False)
        exterior = F.interpolate(self.exterior_head(features), size=size, mode="bilinear", align_corners=False)
        result = {"presence": presence, "type": damage_type, "exterior": exterior}
        if self.case_classifier is not None:
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
            if self.case_pooling == "avg_max":
                pooled = torch.cat(
                    (pooled, F.adaptive_max_pool2d(features, 1).flatten(1)), dim=1
                )
            result["case_logits"] = self.case_classifier(pooled)
        return result


class Dinov2DamageSegmenter(nn.Module):
    """DINOv2 patch features with lightweight dense prediction heads."""

    def __init__(
        self,
        num_classes: int,
        *,
        backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained_backbone: bool = False,
        case_classifier: bool = False,
        case_pooling: str = "avg",
    ) -> None:
        super().__init__()
        if case_pooling not in {"avg", "avg_max"}:
            raise ValueError("case_pooling must be 'avg' or 'avg_max'")
        if backbone_name not in APPROVED_DINOV2_BACKBONES:
            raise ValueError("DINOv2 backbone is not in the commercially reviewed allowlist")
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,
            num_classes=0,
            dynamic_img_size=True,
        )
        feature_dim = int(self.backbone.num_features)
        patch_size = self.backbone.patch_embed.patch_size
        self.patch_size = (
            (int(patch_size), int(patch_size))
            if isinstance(patch_size, int)
            else (int(patch_size[0]), int(patch_size[1]))
        )
        hidden = min(256, feature_dim)

        def head(channels: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(feature_dim, hidden, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(min(32, hidden), hidden),
                nn.GELU(),
                nn.Dropout2d(0.1),
                nn.Conv2d(hidden, channels, kernel_size=1),
            )

        self.presence_head = head(1)
        self.type_head = head(num_classes - 1)
        self.exterior_head = head(1)
        self.case_pooling = case_pooling
        case_features = feature_dim * (2 if case_pooling == "avg_max" else 1)
        self.case_classifier = (
            nn.Linear(case_features, num_classes - 1) if case_classifier else None
        )
        self.num_classes = num_classes

    def _extract_patch_features(self, padded: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(padded)
        prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        patch_tokens = tokens[:, prefix_tokens:]
        patch_height, patch_width = self.patch_size
        grid_height = padded.shape[-2] // patch_height
        grid_width = padded.shape[-1] // patch_width
        if patch_tokens.shape[1] != grid_height * grid_width:
            raise RuntimeError("DINOv2 patch-token shape does not match the input grid")
        return patch_tokens.transpose(1, 2).reshape(
            padded.shape[0], patch_tokens.shape[2], grid_height, grid_width
        )

    def _extract_head_features(self, padded: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self._extract_patch_features(padded)
        return {
            "presence": features,
            "type": features,
            "exterior": features,
            "case": features,
        }

    def _type_logits(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.type_head(features["type"])

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        original_height, original_width = images.shape[-2:]
        patch_height, patch_width = self.patch_size
        pad_height = (-original_height) % patch_height
        pad_width = (-original_width) % patch_width
        padded = F.pad(images, (0, pad_width, 0, pad_height))
        features = self._extract_head_features(padded)

        def upsample(head: nn.Module, value: torch.Tensor) -> torch.Tensor:
            value = F.interpolate(
                head(value),
                size=padded.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return value[..., :original_height, :original_width]

        result = {
            "presence": upsample(self.presence_head, features["presence"]),
            "type": upsample(nn.Identity(), self._type_logits(features)),
            "exterior": upsample(self.exterior_head, features["exterior"]),
        }
        if self.case_classifier is not None:
            pooled = features["case"].mean(dim=(2, 3))
            if self.case_pooling == "avg_max":
                pooled = torch.cat(
                    (pooled, features["case"].amax(dim=(2, 3))), dim=1
                )
            result["case_logits"] = self.case_classifier(pooled)
        return result


class Dinov2FourLayerDamageSegmenter(Dinov2DamageSegmenter):
    """Fuse the last four DINOv2 blocks before the dense prediction heads.

    DINOv2's official semantic-segmentation evaluation exposes a four-layer
    linear head. The fusion projection is initialized to select only the final
    block, making a warm-started model numerically equivalent to the existing
    single-layer decoder before optimization.
    """

    def __init__(self, num_classes: int, **kwargs) -> None:
        super().__init__(num_classes, **kwargs)
        feature_dim = int(self.backbone.num_features)
        self.feature_fusion = _identity_four_layer_projection(feature_dim)

    def _extract_intermediate_features(self, padded: torch.Tensor) -> list[torch.Tensor]:
        features = self.backbone.get_intermediate_layers(
            padded,
            n=4,
            reshape=True,
            norm=True,
        )
        if len(features) != 4:
            raise RuntimeError("DINOv2 did not return the requested four feature layers")
        return features

    def _extract_patch_features(self, padded: torch.Tensor) -> torch.Tensor:
        features = self._extract_intermediate_features(padded)
        return self.feature_fusion(torch.cat(features, dim=1))


class Dinov2FourLayerSplitDamageSegmenter(Dinov2FourLayerDamageSegmenter):
    """Use independently selectable four-layer features for each dense head."""

    def __init__(self, num_classes: int, **kwargs) -> None:
        super().__init__(num_classes, **kwargs)
        feature_dim = int(self.backbone.num_features)
        del self.feature_fusion
        self.presence_feature_fusion = _identity_four_layer_projection(feature_dim)
        self.type_feature_fusion = _identity_four_layer_projection(feature_dim)
        self.exterior_feature_fusion = _identity_four_layer_projection(feature_dim)

    def _extract_head_features(self, padded: torch.Tensor) -> dict[str, torch.Tensor]:
        levels = self._extract_intermediate_features(padded)
        concatenated = torch.cat(levels, dim=1)
        return {
            "presence": self.presence_feature_fusion(concatenated),
            "type": self.type_feature_fusion(concatenated),
            "exterior": self.exterior_feature_fusion(concatenated),
            "case": levels[-1],
        }


class Dinov2FourLayerDualTypeDamageSegmenter(Dinov2FourLayerSplitDamageSegmenter):
    """Fuse single-layer and four-layer type probabilities in one model."""

    def __init__(
        self,
        num_classes: int,
        *,
        type_probability_fusion: str = "average",
        type_probability_weight_four_layer: float = 0.5,
        type_probability_class_weights_four_layer: tuple[float, ...] | list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(num_classes, **kwargs)
        if type_probability_fusion not in {
            "average",
            "maximum",
            "classwise",
            "classwise_maximum",
        }:
            raise ValueError(
                "type_probability_fusion must be 'average', 'maximum', "
                "'classwise', or 'classwise_maximum'"
            )
        if not 0.0 <= type_probability_weight_four_layer <= 1.0:
            raise ValueError("type_probability_weight_four_layer must be in [0, 1]")
        class_count = num_classes - 1
        if type_probability_class_weights_four_layer is None:
            class_weights = (type_probability_weight_four_layer,) * class_count
        else:
            class_weights = tuple(
                float(value) for value in type_probability_class_weights_four_layer
            )
        if len(class_weights) != class_count or any(
            not 0.0 <= value <= 1.0 for value in class_weights
        ):
            raise ValueError(
                "type_probability_class_weights_four_layer must provide one [0, 1] "
                "weight per damage class"
            )
        feature_dim = int(self.backbone.num_features)
        self.type_head_four_layer = copy.deepcopy(self.type_head)
        self.type_four_layer_feature_fusion = _identity_four_layer_projection(feature_dim)
        self.type_probability_fusion = type_probability_fusion
        self.type_probability_weight_four_layer = type_probability_weight_four_layer
        self.type_probability_class_weights_four_layer = class_weights

    def _extract_head_features(self, padded: torch.Tensor) -> dict[str, torch.Tensor]:
        levels = self._extract_intermediate_features(padded)
        concatenated = torch.cat(levels, dim=1)
        return {
            "presence": self.presence_feature_fusion(concatenated),
            "type": self.type_feature_fusion(concatenated),
            "type_four_layer": self.type_four_layer_feature_fusion(concatenated),
            "exterior": self.exterior_feature_fusion(concatenated),
            "case": levels[-1],
        }

    def _type_logits(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        parent = self.type_head(features["type"]).softmax(dim=1)
        four_layer = self.type_head_four_layer(features["type_four_layer"]).softmax(dim=1)
        if self.type_probability_fusion == "maximum":
            probability = torch.maximum(parent, four_layer)
            probability = probability / probability.sum(dim=1, keepdim=True)
        elif self.type_probability_fusion == "classwise":
            weights = parent.new_tensor(
                self.type_probability_class_weights_four_layer
            ).view(1, -1, 1, 1)
            probability = (1.0 - weights) * parent + weights * four_layer
            probability = probability / probability.sum(dim=1, keepdim=True)
        elif self.type_probability_fusion == "classwise_maximum":
            mask = parent.new_tensor(
                self.type_probability_class_weights_four_layer
            ).view(1, -1, 1, 1)
            selected_maximum = torch.maximum(parent, four_layer)
            probability = (1.0 - mask) * four_layer + mask * selected_maximum
            probability = probability / probability.sum(dim=1, keepdim=True)
        else:
            weight = self.type_probability_weight_four_layer
            probability = (1.0 - weight) * parent + weight * four_layer
        return probability.clamp_min(1e-7).log()


class Dinov2FourLayerTypeRescueDamageSegmenter(Dinov2FourLayerDamageSegmenter):
    """Union one parent high-confidence damage type into a four-layer mask."""

    def __init__(
        self,
        num_classes: int,
        *,
        rescue_type_index: int = 2,
        rescue_threshold: float = 0.69,
        **kwargs,
    ) -> None:
        super().__init__(num_classes, **kwargs)
        class_count = num_classes - 1
        if not 0 <= rescue_type_index < class_count:
            raise ValueError("rescue_type_index is outside the damage-type range")
        if not 0.0 < rescue_threshold < 1.0:
            raise ValueError("rescue_threshold must be in (0, 1)")
        feature_dim = int(self.backbone.num_features)
        self.rescue_feature_fusion = _identity_four_layer_projection(feature_dim)
        self.rescue_presence_head = copy.deepcopy(self.presence_head)
        self.rescue_type_head = copy.deepcopy(self.type_head)
        self.rescue_exterior_head = copy.deepcopy(self.exterior_head)
        self.rescue_type_index = rescue_type_index
        self.rescue_threshold = rescue_threshold

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        original_height, original_width = images.shape[-2:]
        patch_height, patch_width = self.patch_size
        pad_height = (-original_height) % patch_height
        pad_width = (-original_width) % patch_width
        padded = F.pad(images, (0, pad_width, 0, pad_height))
        levels = self._extract_intermediate_features(padded)
        concatenated = torch.cat(levels, dim=1)
        four_layer_features = self.feature_fusion(concatenated)
        parent_features = self.rescue_feature_fusion(concatenated)

        def upsample_logits(head: nn.Module, features: torch.Tensor) -> torch.Tensor:
            return F.interpolate(
                head(features),
                size=padded.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        main_presence = upsample_logits(self.presence_head, four_layer_features).sigmoid()
        main_exterior = upsample_logits(self.exterior_head, four_layer_features).sigmoid()
        main_type = upsample_logits(self.type_head, four_layer_features).softmax(dim=1)
        rescue_presence = upsample_logits(
            self.rescue_presence_head, parent_features
        ).sigmoid()
        rescue_exterior = upsample_logits(
            self.rescue_exterior_head, parent_features
        ).sigmoid()
        rescue_type = upsample_logits(
            self.rescue_type_head, parent_features
        ).softmax(dim=1)

        main_score = main_presence * main_exterior
        rescue_score = rescue_presence * rescue_exterior
        eligible = (
            (rescue_score >= self.rescue_threshold)
            & (rescue_type.argmax(dim=1, keepdim=True) == self.rescue_type_index)
        )
        use_rescue = eligible & (rescue_score > main_score)
        combined_score = torch.where(use_rescue, rescue_score, main_score)
        combined_type = torch.where(use_rescue.expand_as(main_type), rescue_type, main_type)
        result = {
            "presence": torch.logit(combined_score.clamp(1e-6, 1.0 - 1e-6)),
            "type": combined_type.clamp_min(1e-7).log(),
            "exterior": torch.full_like(combined_score, 20.0),
        }
        if self.case_classifier is not None:
            pooled = four_layer_features.mean(dim=(2, 3))
            if self.case_pooling == "avg_max":
                pooled = torch.cat(
                    (pooled, four_layer_features.amax(dim=(2, 3))), dim=1
                )
            result["case_logits"] = self.case_classifier(pooled)
        return {
            key: value[..., :original_height, :original_width]
            if value.ndim == 4
            else value
            for key, value in result.items()
        }


class Dinov2FourLayerRoleSplitDamageSegmenter(Dinov2FourLayerDamageSegmenter):
    """Use parent final-layer heads for triage and four-layer heads for masks."""

    def __init__(self, num_classes: int, **kwargs) -> None:
        super().__init__(num_classes, **kwargs)
        feature_dim = int(self.backbone.num_features)
        self.triage_feature_fusion = _identity_four_layer_projection(feature_dim)
        self.triage_presence_head = copy.deepcopy(self.presence_head)
        self.triage_type_head = copy.deepcopy(self.type_head)
        self.triage_exterior_head = copy.deepcopy(self.exterior_head)

    def forward_roles(
        self,
        images: torch.Tensor,
        *,
        include_main: bool = True,
        include_triage: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run only the head sets required by a calibrated inference scale."""
        if not include_main and not include_triage:
            raise ValueError("at least one role-split branch must be requested")
        original_height, original_width = images.shape[-2:]
        patch_height, patch_width = self.patch_size
        pad_height = (-original_height) % patch_height
        pad_width = (-original_width) % patch_width
        padded = F.pad(images, (0, pad_width, 0, pad_height))
        levels = self._extract_intermediate_features(padded)
        concatenated = torch.cat(levels, dim=1)

        def upsample(head: nn.Module, features: torch.Tensor) -> torch.Tensor:
            value = F.interpolate(
                head(features),
                size=padded.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return value[..., :original_height, :original_width]

        result: dict[str, torch.Tensor] = {}
        if include_main:
            mask_features = self.feature_fusion(concatenated)
            result.update(
                {
                    "presence": upsample(self.presence_head, mask_features),
                    "type": upsample(self.type_head, mask_features),
                    "exterior": upsample(self.exterior_head, mask_features),
                }
            )
        if include_triage:
            # The role-split triage branch is the bit-identical parent
            # final-layer decoder. The compatibility projection remains in
            # the state dict, but its frozen identity selected exactly this
            # final level and need not run.
            triage_features = levels[-1]
            result.update(
                {
                    "triage_presence": upsample(
                        self.triage_presence_head, triage_features
                    ),
                    "triage_type": upsample(self.triage_type_head, triage_features),
                    "triage_exterior": upsample(
                        self.triage_exterior_head, triage_features
                    ),
                }
            )
        if include_main and self.case_classifier is not None:
            pooled = mask_features.mean(dim=(2, 3))
            if self.case_pooling == "avg_max":
                pooled = torch.cat((pooled, mask_features.amax(dim=(2, 3))), dim=1)
            result["case_logits"] = self.case_classifier(pooled)
        return result

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_roles(images)


class Dinov2FourLayerDetailRoleSplitDamageSegmenter(
    Dinov2FourLayerRoleSplitDamageSegmenter
):
    """Refine only the precise mask with quarter-resolution RGB detail.

    The residual predictor is zero-initialized, so a warm-start is exactly
    equivalent to its role-split parent. Triage bypasses this branch entirely.
    """

    def __init__(self, num_classes: int, **kwargs) -> None:
        super().__init__(num_classes, **kwargs)
        output_channels = num_classes + 1
        self.detail_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.detail_refiner = nn.Sequential(
            nn.Conv2d(
                64 + output_channels,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, output_channels, kernel_size=1),
        )
        final = self.detail_refiner[-1]
        assert isinstance(final, nn.Conv2d)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward_roles(
        self,
        images: torch.Tensor,
        *,
        include_main: bool = True,
        include_triage: bool = True,
    ) -> dict[str, torch.Tensor]:
        result = super().forward_roles(
            images,
            include_main=include_main,
            include_triage=include_triage,
        )
        if not include_main:
            return result
        coarse = torch.cat(
            (result["presence"], result["type"], result["exterior"]),
            dim=1,
        )
        detail = self.detail_encoder(images)
        coarse_detail = F.interpolate(
            coarse,
            size=detail.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        residual = F.interpolate(
            self.detail_refiner(torch.cat((detail, coarse_detail), dim=1)),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        type_channels = self.num_classes - 1
        result["presence"] = result["presence"] + residual[:, :1]
        result["type"] = result["type"] + residual[:, 1 : 1 + type_channels]
        result["exterior"] = result["exterior"] + residual[:, -1:]
        return result


def damage_probabilities(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compose mutually exclusive semantic probabilities from decoupled heads."""
    presence = outputs["presence"].sigmoid()
    damage_type = outputs["type"].softmax(dim=1)
    return torch.cat((1.0 - presence, presence * damage_type), dim=1)


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    has_case_classifier = any(
        key.startswith("case_classifier.") for key in checkpoint["model"]
    )
    case_weight = checkpoint["model"].get("case_classifier.weight")
    case_pooling = checkpoint.get("config", {}).get("case_classifier", {}).get("pooling")
    if case_pooling is None:
        case_pooling = (
            "avg_max" if case_weight is not None and case_weight.shape[1] == 4096 else "avg"
        )
    architecture = checkpoint.get(
        "architecture", checkpoint.get("config", {}).get("architecture", "deeplabv3_resnet50")
    )
    if architecture in {
        "dinov2",
        "dinov2_4layer",
        "dinov2_4layer_split",
        "dinov2_4layer_dual_type",
        "dinov2_4layer_type_rescue",
        "dinov2_4layer_role_split",
        "dinov2_4layer_detail_role_split",
    }:
        model_class = {
            "dinov2": Dinov2DamageSegmenter,
            "dinov2_4layer": Dinov2FourLayerDamageSegmenter,
            "dinov2_4layer_split": Dinov2FourLayerSplitDamageSegmenter,
            "dinov2_4layer_dual_type": Dinov2FourLayerDualTypeDamageSegmenter,
            "dinov2_4layer_type_rescue": Dinov2FourLayerTypeRescueDamageSegmenter,
            "dinov2_4layer_role_split": Dinov2FourLayerRoleSplitDamageSegmenter,
            "dinov2_4layer_detail_role_split": (
                Dinov2FourLayerDetailRoleSplitDamageSegmenter
            ),
        }[architecture]
        architecture_options = {}
        if architecture == "dinov2_4layer_dual_type":
            architecture_options = {
                "type_probability_fusion": checkpoint.get("config", {}).get(
                    "type_probability_fusion", "average"
                ),
                "type_probability_weight_four_layer": checkpoint.get("config", {}).get(
                    "type_probability_weight_four_layer", 0.5
                ),
                "type_probability_class_weights_four_layer": checkpoint.get(
                    "config", {}
                ).get("type_probability_class_weights_four_layer"),
            }
        elif architecture == "dinov2_4layer_type_rescue":
            architecture_options = {
                "rescue_type_index": int(
                    checkpoint.get("config", {}).get("rescue_type_index", 2)
                ),
                "rescue_threshold": float(
                    checkpoint.get("config", {}).get("rescue_threshold", 0.69)
                ),
            }
        model = model_class(
            len(classes),
            backbone_name=checkpoint.get("config", {}).get(
                "backbone_name", "vit_small_patch14_dinov2.lvd142m"
            ),
            pretrained_backbone=False,
            case_classifier=has_case_classifier,
            case_pooling=case_pooling,
            **architecture_options,
        )
    elif architecture == "deeplabv3_resnet50":
        model = DamageSegmenter(
            len(classes),
            pretrained_backbone=False,
            case_classifier=has_case_classifier,
            case_pooling=case_pooling,
        )
    else:
        raise ValueError(f"unsupported checkpoint architecture: {architecture!r}")
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint
