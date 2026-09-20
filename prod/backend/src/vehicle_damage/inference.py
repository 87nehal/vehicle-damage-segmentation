from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .model import damage_probabilities


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def filter_small_damage_components(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    """Remove 8-connected predicted damage islands smaller than a frozen area."""
    if minimum_pixels < 0:
        raise ValueError("minimum_pixels must be non-negative")
    result = np.asarray(mask).copy()
    if minimum_pixels <= 1 or not np.any(result):
        return result
    labels, count = ndimage.label(
        result > 0,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= minimum_pixels
    keep[0] = False
    result[~keep[labels]] = 0
    return result


@dataclass(frozen=True)
class QualityReport:
    width: int
    height: int
    dark_fraction: float
    clipped_highlight_fraction: float
    specular_highlight_fraction: float
    sharpness: float
    review_reasons: tuple[str, ...]


@dataclass(frozen=True)
class InferenceDecision:
    """Fail-closed disposition for one image and its two dense predictions."""

    decision: str
    automated_decision_allowed: bool
    manual_review: bool
    manual_review_required: bool
    recapture_required: bool
    decision_reasons: tuple[str, ...]
    recapture_reasons: tuple[str, ...]
    triage_only_fraction: float
    segmentation_only_fraction: float
    damage_type_disagreement_fraction: float
    near_threshold_fraction: float


_RECAPTURE_QUALITY_REASONS = frozenset(
    {
        "low_resolution",
        "underexposed",
        "excessive_glare_or_overexposure",
        "blur_or_low_detail",
    }
)


def capture_quality_disposition(
    quality: QualityReport,
) -> tuple[str, tuple[str, ...]]:
    """Return the pre-inference capture gate and its recapture-only reasons."""
    recapture_reasons = tuple(
        reason
        for reason in quality.review_reasons
        if reason in _RECAPTURE_QUALITY_REASONS
    )
    if recapture_reasons:
        return "recapture_required", recapture_reasons
    if quality.review_reasons:
        return "manual_quality_review", ()
    return "quality_pass", ()


def route_inference_decision(
    quality: QualityReport,
    triage_mask: np.ndarray,
    segmentation_mask: np.ndarray,
    *,
    near_threshold_fraction: float,
    max_uncertain_fraction: float,
) -> InferenceDecision:
    """Route unsafe predictions to review/recapture without altering masks.

    The two branches intentionally serve different operating points. Any
    one-sided damage evidence therefore remains visible to a reviewer instead
    of being silently converted into either a positive or a negative result.
    """
    triage = np.asarray(triage_mask)
    segmentation = np.asarray(segmentation_mask)
    if triage.shape != segmentation.shape or triage.ndim != 2 or triage.size == 0:
        raise ValueError("triage and segmentation masks must be non-empty 2D arrays of equal shape")
    if not np.isfinite(near_threshold_fraction) or not 0 <= near_threshold_fraction <= 1:
        raise ValueError("near_threshold_fraction must be finite and in [0, 1]")
    if not np.isfinite(max_uncertain_fraction) or not 0 <= max_uncertain_fraction <= 1:
        raise ValueError("max_uncertain_fraction must be finite and in [0, 1]")

    triage_positive = triage > 0
    segmentation_positive = segmentation > 0
    triage_only = triage_positive & ~segmentation_positive
    segmentation_only = segmentation_positive & ~triage_positive
    type_disagreement = (
        triage_positive
        & segmentation_positive
        & (triage != segmentation)
    )
    triage_only_fraction = float(triage_only.mean())
    segmentation_only_fraction = float(segmentation_only.mean())
    type_disagreement_fraction = float(type_disagreement.mean())

    reasons = [f"quality:{reason}" for reason in quality.review_reasons]
    if np.any(triage_only):
        reasons.append("triage_only_damage_evidence")
    if np.any(segmentation_only):
        reasons.append("segmentation_only_damage_evidence")
    if np.any(type_disagreement):
        reasons.append("damage_type_disagreement")
    if near_threshold_fraction > max_uncertain_fraction:
        reasons.append("excessive_near_threshold_area")

    quality_disposition, recapture_reasons = capture_quality_disposition(quality)
    if quality_disposition == "recapture_required":
        decision = "recapture_required"
    elif reasons:
        decision = "manual_review_required"
    elif np.any(triage_positive) or np.any(segmentation_positive):
        decision = "damage_detected"
    else:
        decision = "no_damage_detected"

    automated = decision in {"damage_detected", "no_damage_detected"}
    return InferenceDecision(
        decision=decision,
        automated_decision_allowed=automated,
        # Preserve the original fail-closed boolean contract: older consumers
        # must continue to stop for both manual review and recapture outcomes.
        manual_review=not automated,
        manual_review_required=decision == "manual_review_required",
        recapture_required=decision == "recapture_required",
        decision_reasons=tuple(reasons),
        recapture_reasons=recapture_reasons,
        triage_only_fraction=triage_only_fraction,
        segmentation_only_fraction=segmentation_only_fraction,
        damage_type_disagreement_fraction=type_disagreement_fraction,
        near_threshold_fraction=float(near_threshold_fraction),
    )


def assess_quality(image: Image.Image) -> QualityReport:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    dark = float((luminance < 0.04).mean())
    clipped = float((luminance > 0.985).mean())
    maximum = rgb.max(axis=2)
    saturation = (maximum - rgb.min(axis=2)) / np.maximum(maximum, 1e-6)
    specular = float(((luminance > 0.90) & (saturation < 0.12)).mean())
    dx = np.diff(luminance, axis=1)
    dy = np.diff(luminance, axis=0)
    sharpness = float((dx * dx).mean() + (dy * dy).mean())
    reasons: list[str] = []
    if min(image.size) < 480:
        reasons.append("low_resolution")
    if dark > 0.35:
        reasons.append("underexposed")
    if clipped > 0.20:
        reasons.append("excessive_glare_or_overexposure")
    if specular > 0.015:
        reasons.append("possible_specular_glare")
    if sharpness < 0.00035:
        reasons.append("blur_or_low_detail")
    return QualityReport(
        image.width,
        image.height,
        dark,
        clipped,
        specular,
        sharpness,
        tuple(reasons),
    )


def _starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")
    points = list(range(0, length - tile_size + 1, stride))
    if points[-1] != length - tile_size:
        points.append(length - tile_size)
    return points


def _prepare(image: Image.Image, device: torch.device | str) -> torch.Tensor:
    tensor = TF.to_tensor(image.convert("RGB"))
    return ((tensor - MEAN) / STD).to(device)


@torch.inference_mode()
def predict_case_probabilities(
    model: torch.nn.Module,
    image: Image.Image,
    *,
    device: torch.device | str,
    image_size: int = 384,
    horizontal_flip_tta: bool = True,
    mixed_precision: bool = False,
) -> torch.Tensor:
    """Predict independent image-level damage classes from the optional case head."""
    rgb = image.convert("RGB")
    scale = image_size / max(rgb.size)
    resized = TF.resize(
        rgb,
        [max(1, round(rgb.height * scale)), max(1, round(rgb.width * scale))],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    pad_w = image_size - resized.width
    pad_h = image_size - resized.height
    tensor = _prepare(
        TF.pad(
            resized,
            [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2],
            fill=0,
        ),
        device,
    ).unsqueeze(0)
    device_type = torch.device(device).type
    if mixed_precision and device_type != "cuda":
        raise ValueError("mixed-precision inference requires a CUDA device")
    with torch.autocast(device_type, dtype=torch.float16, enabled=mixed_precision):
        output = model(tensor)
    if "case_logits" not in output:
        raise ValueError("checkpoint does not contain a case-classification head")
    probability = output["case_logits"].sigmoid()[0]
    if horizontal_flip_tta:
        with torch.autocast(device_type, dtype=torch.float16, enabled=mixed_precision):
            flipped = model(torch.flip(tensor, dims=[3]))
        probability = 0.5 * (probability + flipped["case_logits"].sigmoid()[0])
    return probability.cpu()


@torch.inference_mode()
def predict_probabilities(
    model: torch.nn.Module,
    image: Image.Image,
    *,
    device: torch.device | str,
    tile_size: int = 768,
    overlap: int = 192,
    horizontal_flip_tta: bool = True,
    mixed_precision: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sliding-window prediction preserving small damage in full-car photos."""
    tensor = _prepare(image, device)
    device_type = torch.device(device).type
    if mixed_precision and device_type != "cuda":
        raise ValueError("mixed-precision inference requires a CUDA device")
    _, height, width = tensor.shape
    pad_h, pad_w = max(0, tile_size - height), max(0, tile_size - width)
    pad_mode = "reflect" if pad_h < height and pad_w < width else "replicate"
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode=pad_mode)
    padded_h, padded_w = tensor.shape[-2:]
    accum: torch.Tensor | None = None
    exterior_accum = torch.zeros((padded_h, padded_w), device=device)
    weight_accum = torch.zeros((padded_h, padded_w), device=device)
    window_1d = torch.hann_window(tile_size, periodic=False, device=device).clamp_min(0.05)
    window = window_1d[:, None] * window_1d[None, :]

    for top in _starts(padded_h, tile_size, overlap):
        for left in _starts(padded_w, tile_size, overlap):
            tile = tensor[:, top : top + tile_size, left : left + tile_size].unsqueeze(0)
            with torch.autocast(device_type, dtype=torch.float16, enabled=mixed_precision):
                output = model(tile)
            damage = damage_probabilities(output)[0]
            exterior = output["exterior"].sigmoid()[0, 0]
            if horizontal_flip_tta:
                with torch.autocast(device_type, dtype=torch.float16, enabled=mixed_precision):
                    flipped = model(torch.flip(tile, dims=[3]))
                damage = 0.5 * (damage + torch.flip(damage_probabilities(flipped)[0], dims=[2]))
                exterior = 0.5 * (exterior + torch.flip(flipped["exterior"].sigmoid()[0, 0], dims=[1]))
            if accum is None:
                accum = torch.zeros((damage.shape[0], padded_h, padded_w), device=device)
            accum[:, top : top + tile_size, left : left + tile_size] += damage * window
            exterior_accum[top : top + tile_size, left : left + tile_size] += exterior * window
            weight_accum[top : top + tile_size, left : left + tile_size] += window
    assert accum is not None
    damage_probability = accum / weight_accum.clamp_min(1e-6)
    exterior_probability = exterior_accum / weight_accum.clamp_min(1e-6)
    return damage_probability[:, :height, :width].cpu(), exterior_probability[:height, :width].cpu()


def _branch_probabilities(
    output: dict[str, torch.Tensor], prefix: str
) -> tuple[torch.Tensor, torch.Tensor]:
    branch = {
        "presence": output[f"{prefix}presence"],
        "type": output[f"{prefix}type"],
        "exterior": output[f"{prefix}exterior"],
    }
    return damage_probabilities(branch)[0], branch["exterior"].sigmoid()[0, 0]


@torch.inference_mode()
def _predict_role_split_probabilities(
    model: torch.nn.Module,
    image: Image.Image,
    *,
    device: torch.device | str,
    tile_size: int,
    overlap: int,
    horizontal_flip_tta: bool,
    mixed_precision: bool,
    include_main: bool = True,
    include_triage: bool = True,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Accumulate segmentation and optional triage heads in one tiled pass."""
    if not include_main and not include_triage:
        raise ValueError("at least one role-split branch must be requested")
    tensor = _prepare(image, device)
    device_type = torch.device(device).type
    if mixed_precision and device_type != "cuda":
        raise ValueError("mixed-precision inference requires a CUDA device")
    _, height, width = tensor.shape
    pad_h, pad_w = max(0, tile_size - height), max(0, tile_size - width)
    pad_mode = "reflect" if pad_h < height and pad_w < width else "replicate"
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode=pad_mode)
    padded_h, padded_w = tensor.shape[-2:]
    accumulators: dict[str, torch.Tensor] = {}
    exterior_accumulators: dict[str, torch.Tensor] = {}
    weight_accum = torch.zeros((padded_h, padded_w), device=device)
    window_1d = torch.hann_window(tile_size, periodic=False, device=device).clamp_min(0.05)
    window = window_1d[:, None] * window_1d[None, :]

    for top in _starts(padded_h, tile_size, overlap):
        for left in _starts(padded_w, tile_size, overlap):
            tile = tensor[:, top : top + tile_size, left : left + tile_size].unsqueeze(0)
            forward_roles = getattr(model, "forward_roles", None)
            with torch.autocast(device_type, dtype=torch.float16, enabled=mixed_precision):
                output = (
                    forward_roles(
                        tile,
                        include_main=include_main,
                        include_triage=include_triage,
                    )
                    if callable(forward_roles)
                    else model(tile)
                )
            branches = []
            if include_main:
                branches.append("")
            if include_triage and all(
                key in output
                for key in ("triage_presence", "triage_type", "triage_exterior")
            ):
                branches.append("triage_")
            flipped_output = None
            if horizontal_flip_tta:
                with torch.autocast(device_type, dtype=torch.float16, enabled=mixed_precision):
                    flipped_tile = torch.flip(tile, dims=[3])
                    flipped_output = (
                        forward_roles(
                            flipped_tile,
                            include_main=include_main,
                            include_triage=include_triage,
                        )
                        if callable(forward_roles)
                        else model(flipped_tile)
                    )
            for prefix in branches:
                name = "triage" if prefix else "main"
                damage, exterior = _branch_probabilities(output, prefix)
                if flipped_output is not None:
                    flipped_damage, flipped_exterior = _branch_probabilities(
                        flipped_output, prefix
                    )
                    damage = 0.5 * (
                        damage + torch.flip(flipped_damage, dims=[2])
                    )
                    exterior = 0.5 * (
                        exterior + torch.flip(flipped_exterior, dims=[1])
                    )
                if name not in accumulators:
                    accumulators[name] = torch.zeros(
                        (damage.shape[0], padded_h, padded_w), device=device
                    )
                    exterior_accumulators[name] = torch.zeros(
                        (padded_h, padded_w), device=device
                    )
                accumulators[name][:, top : top + tile_size, left : left + tile_size] += (
                    damage * window
                )
                exterior_accumulators[name][
                    top : top + tile_size, left : left + tile_size
                ] += exterior * window
            weight_accum[top : top + tile_size, left : left + tile_size] += window

    result = {}
    for name, accumulator in accumulators.items():
        damage = accumulator / weight_accum.clamp_min(1e-6)
        exterior = exterior_accumulators[name] / weight_accum.clamp_min(1e-6)
        result[name] = (
            damage[:, :height, :width].cpu(),
            exterior[:height, :width].cpu(),
        )
    return result


def _resize_probability_pair(
    pair: tuple[torch.Tensor, torch.Tensor],
    size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    damage, exterior = pair
    return (
        F.interpolate(
            damage.unsqueeze(0), size=size, mode="bilinear", align_corners=False
        )[0],
        F.interpolate(
            exterior[None, None], size=size, mode="bilinear", align_corners=False
        )[0, 0],
    )


def _predict_role_split_pyramid(
    model: torch.nn.Module,
    image: Image.Image,
    *,
    device: torch.device | str,
    triage_scales: tuple[float, ...] | list[float],
    segmentation_scales: tuple[float, ...] | list[float],
    tile_size: int,
    overlap: int,
    horizontal_flip_tta: bool,
    mixed_precision: bool,
) -> dict[float, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    rgb = image.convert("RGB")
    pyramid = {}
    triage_scale_tuple = tuple(float(value) for value in triage_scales)
    segmentation_scale_tuple = tuple(float(value) for value in segmentation_scales)
    triage_scale_set = set(triage_scale_tuple)
    segmentation_scale_set = set(segmentation_scale_tuple)
    scale_factors = tuple(
        dict.fromkeys((*triage_scale_tuple, *segmentation_scale_tuple))
    )
    for scale in scale_factors:
        scaled = (
            rgb
            if scale == 1.0
            else rgb.resize(
                (
                    max(1, round(rgb.width * scale)),
                    max(1, round(rgb.height * scale)),
                ),
                Image.Resampling.BICUBIC,
            )
        )
        variants = _predict_role_split_probabilities(
            model,
            scaled,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
            horizontal_flip_tta=horizontal_flip_tta,
            mixed_precision=mixed_precision,
            include_main=scale in segmentation_scale_set,
            include_triage=scale in triage_scale_set,
        )
        if scaled.size != rgb.size:
            variants = {
                name: _resize_probability_pair(pair, (rgb.height, rgb.width))
                for name, pair in variants.items()
            }
        pyramid[scale] = variants
    return pyramid


@torch.inference_mode()
def predict_probability_pyramid(
    model: torch.nn.Module,
    image: Image.Image,
    *,
    device: torch.device | str,
    scale_factors: tuple[float, ...] | list[float] = (1.0,),
    tile_size: int = 768,
    overlap: int = 192,
    horizontal_flip_tta: bool = True,
    mixed_precision: bool = False,
) -> dict[float, tuple[torch.Tensor, torch.Tensor]]:
    """Return aligned dense probabilities for unique input scales."""
    scales = tuple(float(value) for value in scale_factors)
    if not scales or any(not np.isfinite(value) or value <= 0 for value in scales):
        raise ValueError("scale_factors must contain positive finite values")
    if len(set(scales)) != len(scales):
        raise ValueError("scale_factors must be unique")
    rgb = image.convert("RGB")
    pyramid: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
    for scale in scales:
        if scale == 1.0:
            scaled = rgb
        else:
            scaled = rgb.resize(
                (
                    max(1, round(rgb.width * scale)),
                    max(1, round(rgb.height * scale)),
                ),
                Image.Resampling.BICUBIC,
            )
        damage, exterior = predict_probabilities(
            model,
            scaled,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
            horizontal_flip_tta=horizontal_flip_tta,
            mixed_precision=mixed_precision,
        )
        if scaled.size != rgb.size:
            damage = F.interpolate(
                damage.unsqueeze(0),
                size=(rgb.height, rgb.width),
                mode="bilinear",
                align_corners=False,
            )[0]
            exterior = F.interpolate(
                exterior[None, None],
                size=(rgb.height, rgb.width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        pyramid[scale] = damage, exterior
    return pyramid


def fuse_probability_pyramid(
    pyramid: dict[float, tuple[torch.Tensor, torch.Tensor]],
    scale_factors: tuple[float, ...] | list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average a declared subset of an aligned probability pyramid."""
    scales = tuple(float(value) for value in scale_factors)
    if not scales:
        raise ValueError("scale_factors cannot be empty")
    missing = [scale for scale in scales if scale not in pyramid]
    if missing:
        raise ValueError(f"probability pyramid is missing scales: {missing}")
    return (
        torch.stack([pyramid[scale][0] for scale in scales]).mean(dim=0),
        torch.stack([pyramid[scale][1] for scale in scales]).mean(dim=0),
    )


@torch.inference_mode()
def predict_multiscale_probabilities(
    model: torch.nn.Module,
    image: Image.Image,
    *,
    device: torch.device | str,
    scale_factors: tuple[float, ...] | list[float] = (1.0,),
    tile_size: int = 768,
    overlap: int = 192,
    horizontal_flip_tta: bool = True,
    mixed_precision: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average aligned dense probabilities from one or more input scales."""
    pyramid = predict_probability_pyramid(
        model,
        image,
        device=device,
        scale_factors=scale_factors,
        tile_size=tile_size,
        overlap=overlap,
        horizontal_flip_tta=horizontal_flip_tta,
        mixed_precision=mixed_precision,
    )
    return fuse_probability_pyramid(pyramid, scale_factors)


@torch.inference_mode()
def predict_profile_probabilities(
    model: torch.nn.Module,
    image: Image.Image,
    *,
    device: torch.device | str,
    triage_scales: tuple[float, ...] | list[float],
    segmentation_scales: tuple[float, ...] | list[float],
    tile_size: int = 768,
    overlap: int = 192,
    horizontal_flip_tta: bool = True,
    mixed_precision: bool = False,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]:
    """Score distinct triage/mask scales while sharing every common pass."""
    all_scales = tuple(dict.fromkeys((*triage_scales, *segmentation_scales)))
    if hasattr(model, "triage_presence_head"):
        variants = _predict_role_split_pyramid(
            model,
            image,
            device=device,
            triage_scales=triage_scales,
            segmentation_scales=segmentation_scales,
            tile_size=tile_size,
            overlap=overlap,
            horizontal_flip_tta=horizontal_flip_tta,
            mixed_precision=mixed_precision,
        )

        def fuse_variant(
            scales: tuple[float, ...] | list[float], name: str
        ) -> tuple[torch.Tensor, torch.Tensor]:
            pairs = []
            for scale in scales:
                scale_variants = variants[float(scale)]
                pairs.append(
                    scale_variants[name]
                    if name in scale_variants
                    else scale_variants["main"]
                )
            return (
                torch.stack([pair[0] for pair in pairs]).mean(dim=0),
                torch.stack([pair[1] for pair in pairs]).mean(dim=0),
            )

        return (
            fuse_variant(triage_scales, "triage"),
            fuse_variant(segmentation_scales, "main"),
        )
    pyramid = predict_probability_pyramid(
        model,
        image,
        device=device,
        scale_factors=all_scales,
        tile_size=tile_size,
        overlap=overlap,
        horizontal_flip_tta=horizontal_flip_tta,
        mixed_precision=mixed_precision,
    )
    return (
        fuse_probability_pyramid(pyramid, triage_scales),
        fuse_probability_pyramid(pyramid, segmentation_scales),
    )


def probabilities_to_mask(
    damage_probability: torch.Tensor,
    exterior_probability: torch.Tensor,
    threshold: float,
    exterior_floor: float = 0.5,
    minimum_component_pixels: int = 0,
    type_probability_multipliers: tuple[float, ...] | list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # Exterior confidence is deliberately a soft factor. A hard vehicle-mask
    # cutoff can erase true damage and is inappropriate for a recall-first path.
    if not 0 <= exterior_floor <= 1:
        raise ValueError("exterior_floor must be in [0, 1]")
    any_damage = (1.0 - damage_probability[0]) * (
        exterior_floor + (1.0 - exterior_floor) * exterior_probability
    )
    type_probability = damage_probability[1:]
    if type_probability_multipliers is not None:
        multipliers = torch.as_tensor(
            type_probability_multipliers,
            dtype=type_probability.dtype,
            device=type_probability.device,
        )
        if multipliers.ndim != 1 or multipliers.numel() != type_probability.shape[0]:
            raise ValueError(
                "type_probability_multipliers must match the number of damage types"
            )
        if not torch.isfinite(multipliers).all() or bool((multipliers <= 0).any()):
            raise ValueError(
                "type_probability_multipliers must contain positive finite values"
            )
        type_probability = type_probability * multipliers[:, None, None]
    _, damage_class = type_probability.max(dim=0)
    mask = damage_class.add(1)
    accepted = any_damage >= threshold
    mask = torch.where(accepted, mask, torch.zeros_like(mask))
    uncertainty = (any_damage - threshold).abs()
    mask_array = mask.numpy().astype(np.uint8)
    mask_array = filter_small_damage_components(mask_array, minimum_component_pixels)
    return mask_array, uncertainty.numpy().astype(np.float32)
