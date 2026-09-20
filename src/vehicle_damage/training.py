from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import DamageDataset
from .losses import RecallOrientedLoss
from .manifest import load_manifest, validate_manifest
from .metrics import binary_auroc, confusion_matrix, metrics_from_confusion
from .model import (
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
from .taxonomy import validate_classes


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_trainable_parameters(
    model: torch.nn.Module,
    *,
    train_only_case_classifier: bool,
    train_only_type_head: bool,
    train_only_detail_refiner: bool,
    case_classifier_enabled: bool,
) -> None:
    """Freeze all but the explicitly selected auxiliary head, when requested."""
    selected_modes = sum(
        (train_only_case_classifier, train_only_type_head, train_only_detail_refiner)
    )
    if selected_modes > 1:
        raise ValueError(
            "case_classifier.train_only, type_head.train_only, and "
            "detail_refiner.train_only are exclusive"
        )
    if train_only_case_classifier:
        if not case_classifier_enabled:
            raise ValueError("case_classifier.train_only requires the case classifier")
        trainable_prefix = "case_classifier."
    elif train_only_type_head:
        trainable_prefix = "type_head."
    elif train_only_detail_refiner:
        trainable_prefix = ("detail_encoder.", "detail_refiner.")
    else:
        return
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(trainable_prefix)


def train(config_path: str, manifest_path: str, output_dir: str, device_name: str | None = None) -> None:
    config = load_config(config_path)
    classes = config["classes"]
    validate_classes(classes)
    samples = load_manifest(manifest_path, require_commercial=True)
    report = validate_manifest(samples, len(classes), check_files=True)
    if not report["valid"]:
        raise ValueError("invalid manifest:\n" + "\n".join(report["errors"]))
    train_samples = [x for x in samples if x.split == "train"]
    validation_samples = [x for x in samples if x.split == "validation" and x.damage_supervised]
    if not train_samples or not validation_samples:
        raise ValueError("manifest must contain train and validation splits")
    if config.get("pretrained_backbone") and (
        config.get("pretrained_backbone_commercial_use") is not True
        or not str(config.get("pretrained_backbone_license_id", "")).strip()
    ):
        raise ValueError(
            "pretrained_backbone requires explicit commercial-use approval and a license/provenance id"
        )

    _seed_everything(int(config["seed"]))
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    case_config = config.get("case_classifier", {})
    case_classifier_enabled = bool(case_config.get("enabled", False))
    architecture = str(config.get("architecture", "deeplabv3_resnet50"))
    model_options = {
        "pretrained_backbone": bool(config["pretrained_backbone"]),
        "case_classifier": case_classifier_enabled,
        "case_pooling": str(case_config.get("pooling", "avg")),
    }
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
                "type_probability_fusion": str(
                    config.get("type_probability_fusion", "average")
                ),
                "type_probability_weight_four_layer": float(
                    config.get("type_probability_weight_four_layer", 0.5)
                ),
                "type_probability_class_weights_four_layer": config.get(
                    "type_probability_class_weights_four_layer"
                ),
            }
        elif architecture == "dinov2_4layer_type_rescue":
            architecture_options = {
                "rescue_type_index": int(config.get("rescue_type_index", 2)),
                "rescue_threshold": float(config.get("rescue_threshold", 0.69)),
            }
        model = model_class(
            len(classes),
            backbone_name=str(
                config.get("backbone_name", "vit_small_patch14_dinov2.lvd142m")
            ),
            **model_options,
            **architecture_options,
        ).to(device)
    elif architecture == "deeplabv3_resnet50":
        model = DamageSegmenter(len(classes), **model_options).to(device)
    else:
        raise ValueError(f"unsupported architecture: {architecture!r}")
    initial_checkpoint = str(config.get("initial_checkpoint", "")).strip()
    backbone_checkpoint = str(config.get("backbone_checkpoint", "")).strip()
    if initial_checkpoint and backbone_checkpoint:
        raise ValueError("use either initial_checkpoint or backbone_checkpoint, not both")
    if initial_checkpoint:
        if config.get("initial_checkpoint_commercial_use") is not True or not str(
            config.get("initial_checkpoint_license_id", "")
        ).strip():
            raise ValueError("initial checkpoint requires explicit commercial-use approval and provenance id")
        source_checkpoint = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        if source_checkpoint.get("classes") != classes:
            raise ValueError("initial checkpoint classes do not match config classes")
        allow_new_heads = bool(config.get("initial_checkpoint_allow_new_heads", False))
        allow_new_decoder = bool(config.get("initial_checkpoint_allow_new_decoder", False))
        incompatible = model.load_state_dict(
            source_checkpoint["model"], strict=not (allow_new_heads or allow_new_decoder)
        )
        if allow_new_heads or allow_new_decoder:
            allowed_missing = set()
            if allow_new_heads:
                allowed_missing.update(
                    {"case_classifier.weight", "case_classifier.bias"}
                )
            if allow_new_decoder and architecture == "dinov2_4layer":
                allowed_missing.add("feature_fusion.weight")
            if allow_new_decoder and architecture == "dinov2_4layer_split":
                allowed_missing.update(
                    {
                        "presence_feature_fusion.weight",
                        "type_feature_fusion.weight",
                        "exterior_feature_fusion.weight",
                    }
                )
            if (
                allow_new_decoder
                and architecture == "dinov2_4layer_detail_role_split"
            ):
                allowed_missing.update(
                    key
                    for key in model.state_dict()
                    if key.startswith(("detail_encoder.", "detail_refiner."))
                )
            if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
                raise ValueError(
                    "initial checkpoint differs beyond the explicitly allowed new modules"
                )
    elif backbone_checkpoint:
        if config.get("backbone_checkpoint_commercial_use") is not True or not str(
            config.get("backbone_checkpoint_license_id", "")
        ).strip():
            raise ValueError("backbone checkpoint requires explicit commercial-use approval and provenance id")
        source = torch.load(backbone_checkpoint, map_location="cpu", weights_only=True)["model"]
        backbone_state = {
            key.removeprefix("backbone."): value for key, value in source.items() if key.startswith("backbone.")
        }
        model.backbone.load_state_dict(backbone_state, strict=True)
    criterion = RecallOrientedLoss(len(classes), **config["loss"]).to(device)
    train_only_case_classifier = bool(case_config.get("train_only", False))
    type_config = config.get("type_head", {})
    train_only_type_head = bool(type_config.get("train_only", False))
    detail_config = config.get("detail_refiner", {})
    train_only_detail_refiner = bool(detail_config.get("train_only", False))
    _configure_trainable_parameters(
        model,
        train_only_case_classifier=train_only_case_classifier,
        train_only_type_head=train_only_type_head,
        train_only_detail_refiner=train_only_detail_refiner,
        case_classifier_enabled=case_classifier_enabled,
    )
    freeze_backbone = bool(config.get("freeze_backbone", False))
    if freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    learning_rate = float(config["learning_rate"])
    backbone_parameters = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    classifier_parameters = (
        [
            parameter
            for parameter in model.case_classifier.parameters()  # type: ignore[union-attr]
            if parameter.requires_grad
        ]
        if case_classifier_enabled
        else []
    )
    classifier_ids = {id(parameter) for parameter in classifier_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) not in backbone_ids
        and id(parameter) not in classifier_ids
    ]
    optimizer_parameters = []
    if head_parameters:
        optimizer_parameters.append({"params": head_parameters, "lr": learning_rate})
    if backbone_parameters:
        optimizer_parameters.append(
            {
                "params": backbone_parameters,
                "lr": learning_rate
                * float(config.get("backbone_learning_rate_multiplier", 1.0)),
            }
        )
    if classifier_parameters:
        optimizer_parameters.append(
            {
                "params": classifier_parameters,
                "lr": learning_rate
                * float(case_config.get("learning_rate_multiplier", 1.0)),
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=learning_rate,
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"]), eta_min=float(config["learning_rate"]) * 0.01
    )
    supervised_count = sum(x.damage_supervised for x in train_samples)
    auxiliary_count = len(train_samples) - supervised_count
    sampler = None
    if supervised_count and auxiliary_count:
        fraction = float(config.get("damage_sample_fraction", 0.7))
        if not 0 < fraction < 1:
            raise ValueError("damage_sample_fraction must be in (0, 1)")
        weights = [
            fraction / supervised_count if x.damage_supervised else (1.0 - fraction) / auxiliary_count
            for x in train_samples
        ]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_samples), replacement=True)
    augmentation_profile = str(config.get("augmentation_profile", "baseline_v1"))
    train_loader = DataLoader(
        DamageDataset(
            train_samples,
            int(config["image_size"]),
            training=True,
            augmentation_profile=augmentation_profile,
        ),
        batch_size=int(config["batch_size"]), shuffle=sampler is None, sampler=sampler,
        num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda", drop_last=True,
    )
    validation_loader = DataLoader(
        DamageDataset(
            validation_samples,
            int(config["image_size"]),
            training=False,
            augmentation_profile=augmentation_profile,
        ),
        batch_size=int(config["batch_size"]), shuffle=False,
        num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_f2 = -1.0
    best_mean_damage_iou = -1.0
    best_case_macro_auroc = -1.0

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        if freeze_backbone:
            # A frozen backbone must also keep stochastic layers disabled.
            model.backbone.eval()
        if train_only_case_classifier:
            # Keep frozen BatchNorm statistics and segmentation outputs fixed.
            model.eval()
            model.case_classifier.train()  # type: ignore[union-attr]
        elif train_only_type_head:
            # Preserve frozen backbone/head normalization and stochastic state;
            # only the type head receives gradients or training-time dropout.
            model.eval()
            model.type_head.train()
        elif train_only_detail_refiner:
            # Keep the parent predictor bit-stable; only the zero-initialized
            # RGB-detail residual learns.
            model.eval()
            model.detail_encoder.train()
            model.detail_refiner.train()
        running = 0.0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            exterior = batch["exterior"].to(device, non_blocking=True)
            hard_negative = batch["hard_negative"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                loss, _ = criterion(
                    outputs, target, exterior, hard_negative,
                    batch["damage_supervised"].to(device),
                    batch["exterior_supervised"].to(device),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())

        model.eval()
        matrix = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
        case_tp = torch.zeros(len(classes) - 1, dtype=torch.int64)
        case_fp = torch.zeros(len(classes) - 1, dtype=torch.int64)
        case_fn = torch.zeros(len(classes) - 1, dtype=torch.int64)
        case_tn = torch.zeros(len(classes) - 1, dtype=torch.int64)
        case_score_batches: list[torch.Tensor] = []
        case_target_batches: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch in validation_loader:
                images = batch["image"].to(device)
                target = batch["mask"].to(device)
                outputs = model(images)
                probability = damage_probabilities(outputs)
                damage_type = probability[:, 1:].argmax(1) + 1
                prediction = torch.where(
                    outputs["presence"].sigmoid().squeeze(1) >= 0.5,
                    damage_type,
                    torch.zeros_like(damage_type),
                )
                matrix += confusion_matrix(prediction.cpu(), target.cpu(), len(classes))
                if "case_logits" in outputs:
                    case_target = torch.stack(
                        [
                            (target == class_id).flatten(1).any(1)
                            for class_id in range(1, len(classes))
                        ],
                        dim=1,
                    )
                    case_prediction = outputs["case_logits"].sigmoid() >= 0.5
                    case_tp += (case_prediction & case_target).sum(0).cpu()
                    case_fp += (case_prediction & ~case_target).sum(0).cpu()
                    case_fn += (~case_prediction & case_target).sum(0).cpu()
                    case_tn += (~case_prediction & ~case_target).sum(0).cpu()
                    case_score_batches.append(outputs["case_logits"].sigmoid().cpu())
                    case_target_batches.append(case_target.cpu())
        metrics = metrics_from_confusion(matrix)
        case_metrics = None
        if case_classifier_enabled:
            precision = case_tp.float() / (case_tp + case_fp).clamp_min(1)
            recall = case_tp.float() / (case_tp + case_fn).clamp_min(1)
            specificity = case_tn.float() / (case_tn + case_fp).clamp_min(1)
            f2 = 5.0 * precision * recall / (4.0 * precision + recall).clamp_min(1e-12)
            score_matrix = torch.cat(case_score_batches)
            target_matrix = torch.cat(case_target_batches)
            auroc = [
                binary_auroc(score_matrix[:, index], target_matrix[:, index])
                for index in range(len(classes) - 1)
            ]
            valid_auroc = [value for value in auroc if value is not None]
            case_metrics = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "specificity": specificity.tolist(),
                "f2": f2.tolist(),
                "macro_f2": float(f2.mean()),
                "auroc": auroc,
                "macro_auroc": (
                    float(sum(valid_auroc) / len(valid_auroc)) if valid_auroc else None
                ),
            }
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, len(train_loader)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        if case_metrics is not None:
            record["case_classification"] = case_metrics
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        checkpoint = {
            "model": model.state_dict(),
            "architecture": architecture,
            "classes": classes,
            "config": config,
            "manifest_report": report,
            "epoch": epoch,
            "validation_metrics": metrics,
            "case_classification_metrics": case_metrics,
        }
        torch.save(checkpoint, output / "latest.pt")
        if metrics["any_damage_f2"] > best_f2:
            best_f2 = metrics["any_damage_f2"]
            torch.save(checkpoint, output / "best.pt")
        if metrics["mean_damage_iou"] > best_mean_damage_iou:
            best_mean_damage_iou = metrics["mean_damage_iou"]
            torch.save(checkpoint, output / "best_damage_iou.pt")
        if (
            case_metrics is not None
            and case_metrics["macro_auroc"] is not None
            and case_metrics["macro_auroc"] > best_case_macro_auroc
        ):
            best_case_macro_auroc = case_metrics["macro_auroc"]
            torch.save(checkpoint, output / "best_case_classifier.pt")
