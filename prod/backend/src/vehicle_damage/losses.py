from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RecallOrientedLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        focal_weight: float = 1.0,
        tversky_weight: float = 1.0,
        exterior_weight: float = 0.2,
        type_weight: float = 1.0,
        classification_weight: float = 0.0,
        hard_negative_weight: float = 3.0,
        class_weights: list[float] | None = None,
        classification_pos_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        if abs(tversky_alpha + tversky_beta - 1.0) > 1e-6:
            raise ValueError("tversky alpha and beta must sum to 1")
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.alpha = tversky_alpha
        self.beta = tversky_beta
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.exterior_weight = exterior_weight
        self.type_weight = type_weight
        self.classification_weight = classification_weight
        self.hard_negative_weight = hard_negative_weight
        weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.register_buffer("class_weights", weights)
        case_weights = (
            torch.tensor(classification_pos_weights, dtype=torch.float32)
            if classification_pos_weights else None
        )
        if case_weights is not None and case_weights.numel() != num_classes - 1:
            raise ValueError("classification_pos_weights must have num_classes - 1 entries")
        self.register_buffer("classification_pos_weights", case_weights)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
        exterior: torch.Tensor,
        hard_negative: torch.Tensor,
        damage_supervised: torch.Tensor | None = None,
        exterior_supervised: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        presence_logits = outputs["presence"].squeeze(1)
        batch_size, height, width = presence_logits.shape
        if damage_supervised is None:
            damage_supervised = torch.ones(batch_size, device=presence_logits.device)
        if exterior_supervised is None:
            exterior_supervised = torch.ones(batch_size, device=presence_logits.device)
        damage_valid = damage_supervised.to(presence_logits).view(-1, 1, 1)
        exterior_valid = exterior_supervised.to(presence_logits).view(-1, 1, 1)
        presence_target = (target > 0).float()
        presence_probability = presence_logits.sigmoid()
        presence_bce = F.binary_cross_entropy_with_logits(presence_logits, presence_target, reduction="none")
        true_probability = torch.where(presence_target.bool(), presence_probability, 1.0 - presence_probability)
        pixel_weight = 1.0 + hard_negative * (self.hard_negative_weight - 1.0)
        if self.class_weights is not None:
            pixel_weight = pixel_weight * self.class_weights[target]
        focal_map = ((1.0 - true_probability) ** self.focal_gamma) * presence_bce * pixel_weight
        damage_pixel_count = damage_valid.sum() * height * width
        if damage_pixel_count > 0:
            focal = (focal_map * damage_valid).sum() / damage_pixel_count
        else:
            focal = presence_logits.sum() * 0.0

        type_target = (target - 1).clamp_min(0)
        type_weights = self.class_weights[1:] if self.class_weights is not None else None
        type_ce_map = F.cross_entropy(outputs["type"], type_target, weight=type_weights, reduction="none")
        supervised_positive = presence_target * damage_valid
        positive_count = supervised_positive.sum()
        if positive_count > 0:
            type_loss = (type_ce_map * supervised_positive).sum() / positive_count
        else:
            type_loss = outputs["type"].sum() * 0.0

        if self.classification_weight:
            if "case_logits" not in outputs:
                raise ValueError("classification_weight requires a case classifier head")
            case_target = torch.stack(
                [(target == class_id).flatten(1).any(1) for class_id in range(1, self.num_classes)],
                dim=1,
            ).to(outputs["case_logits"].dtype)
            case_map = F.binary_cross_entropy_with_logits(
                outputs["case_logits"],
                case_target,
                pos_weight=self.classification_pos_weights,
                reduction="none",
            )
            case_valid = damage_supervised.to(case_map).view(-1, 1)
            valid_values = case_valid.sum() * (self.num_classes - 1)
            classification_loss = (
                (case_map * case_valid).sum() / valid_values
                if valid_values > 0
                else outputs["case_logits"].sum() * 0.0
            )
        else:
            classification_loss = presence_logits.sum() * 0.0

        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        if damage_pixel_count > 0:
            valid = damage_valid.unsqueeze(1)
            pred_damage = presence_probability.unsqueeze(1) * outputs["type"].softmax(dim=1)
            true_damage = target_one_hot[:, 1:]
            dims = (0, 2, 3)
            tp = (pred_damage * true_damage * valid).sum(dims)
            fp = (pred_damage * (1.0 - true_damage) * valid).sum(dims)
            fn = ((1.0 - pred_damage) * true_damage * valid).sum(dims)
            tversky = ((tp + 1.0) / (tp + self.alpha * fp + self.beta * fn + 1.0)).mean()
            tversky_loss = 1.0 - tversky
        else:
            tversky_loss = presence_logits.sum() * 0.0

        exterior_map = F.binary_cross_entropy_with_logits(
            outputs["exterior"].squeeze(1), exterior, reduction="none"
        )
        exterior_pixel_count = exterior_valid.sum() * height * width
        if exterior_pixel_count > 0:
            exterior_loss = (exterior_map * exterior_valid).sum() / exterior_pixel_count
        else:
            exterior_loss = outputs["exterior"].sum() * 0.0
        total = (
            self.focal_weight * focal
            + self.tversky_weight * tversky_loss
            + self.type_weight * type_loss
            + self.classification_weight * classification_loss
            + self.exterior_weight * exterior_loss
        )
        parts = {
            "total": float(total.detach()),
            "presence_focal": float(focal.detach()),
            "type": float(type_loss.detach()),
            "classification": float(classification_loss.detach()),
            "tversky": float(tversky_loss.detach()),
            "exterior": float(exterior_loss.detach()),
        }
        return total, parts
