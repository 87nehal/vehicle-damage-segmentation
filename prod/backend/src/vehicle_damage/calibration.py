from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Calibration:
    any_damage_threshold: float
    target_recall: float
    measured_recall: float
    negative_pixel_fpr: float
    positive_pixels: int
    negative_pixels: int
    calibration_basis: str = "pixel"
    positive_cases: int = 0
    minimum_damage_coverage: float = 0.0
    exterior_floor: float = 0.5
    tile_size: int = 768
    overlap: int = 192
    horizontal_flip_tta: bool = True
    mixed_precision: bool = False
    minimum_component_pixels: int = 0
    segmentation_minimum_component_pixels: int | None = None
    segmentation_component_filter_basis: str = ""
    type_probability_multipliers: tuple[float, ...] | None = None
    type_probability_multiplier_basis: str = ""
    triage_scales: tuple[float, ...] = (1.0,)
    segmentation_scales: tuple[float, ...] | None = None
    segmentation_threshold: float | None = None
    segmentation_threshold_basis: str = ""

    def __post_init__(self) -> None:
        triage_scales = tuple(float(value) for value in self.triage_scales)
        segmentation_scales = (
            None
            if self.segmentation_scales is None
            else tuple(float(value) for value in self.segmentation_scales)
        )
        type_probability_multipliers = (
            None
            if self.type_probability_multipliers is None
            else tuple(float(value) for value in self.type_probability_multipliers)
        )
        for name, values in (
            ("triage_scales", triage_scales),
            ("segmentation_scales", segmentation_scales),
        ):
            if values is not None and (
                not values
                or len(set(values)) != len(values)
                or any(not np.isfinite(value) or value <= 0 for value in values)
            ):
                raise ValueError(f"{name} must contain unique positive finite values")
        object.__setattr__(self, "triage_scales", triage_scales)
        object.__setattr__(self, "segmentation_scales", segmentation_scales)
        object.__setattr__(
            self, "type_probability_multipliers", type_probability_multipliers
        )
        if self.minimum_component_pixels < 0:
            raise ValueError("minimum_component_pixels must be non-negative")
        if (
            self.segmentation_minimum_component_pixels is not None
            and self.segmentation_minimum_component_pixels < 0
        ):
            raise ValueError(
                "segmentation_minimum_component_pixels must be non-negative"
            )
        if type_probability_multipliers is not None and (
            not type_probability_multipliers
            or any(
                not np.isfinite(value) or value <= 0
                for value in type_probability_multipliers
            )
        ):
            raise ValueError(
                "type_probability_multipliers must contain positive finite values"
            )
        if (
            type_probability_multipliers is not None
            and not self.type_probability_multiplier_basis.strip()
        ):
            raise ValueError(
                "type_probability_multipliers requires a non-empty selection basis"
            )
        if (
            self.segmentation_minimum_component_pixels is not None
            and not self.segmentation_component_filter_basis.strip()
        ):
            raise ValueError(
                "segmentation_minimum_component_pixels requires a non-empty "
                "selection basis"
            )
        if not 0 <= self.any_damage_threshold <= 1:
            raise ValueError("any_damage_threshold must be in [0, 1]")
        if self.segmentation_threshold is not None:
            if not 0 <= self.segmentation_threshold <= 1:
                raise ValueError("segmentation_threshold must be in [0, 1]")
            if self.segmentation_threshold < self.any_damage_threshold:
                raise ValueError(
                    "segmentation_threshold must be at least the recall-first threshold"
                )
            if not self.segmentation_threshold_basis.strip():
                raise ValueError(
                    "segmentation_threshold requires a non-empty selection basis"
                )

    @property
    def effective_segmentation_threshold(self) -> float:
        """Return the precise-mask threshold, falling back for legacy profiles."""
        return (
            self.any_damage_threshold
            if self.segmentation_threshold is None
            else self.segmentation_threshold
        )

    @property
    def effective_segmentation_scales(self) -> tuple[float, ...]:
        return (
            self.triage_scales
            if self.segmentation_scales is None
            else self.segmentation_scales
        )

    @property
    def effective_segmentation_minimum_component_pixels(self) -> int:
        """Return the precise-mask filter, falling back for legacy profiles."""
        return (
            self.minimum_component_pixels
            if self.segmentation_minimum_component_pixels is None
            else self.segmentation_minimum_component_pixels
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def calibrate_threshold(probability: np.ndarray, target: np.ndarray, target_recall: float = 0.97) -> Calibration:
    """Select the highest threshold meeting pixel recall on a held-out calibration set."""
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    target = np.asarray(target).reshape(-1).astype(bool)
    if probability.shape != target.shape:
        raise ValueError("probability and target must have the same shape")
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    positives = probability[target]
    negatives = probability[~target]
    if positives.size == 0 or negatives.size == 0:
        raise ValueError("calibration requires both damage and non-damage pixels")
    # Pick the highest observed threshold whose empirical recall still meets
    # the target. Quantile interpolation can otherwise choose a needlessly low
    # threshold and inflate false positives on small calibration sets.
    descending = np.sort(positives)[::-1]
    required = int(np.ceil(target_recall * positives.size))
    threshold = float(descending[required - 1])
    measured_recall = float((positives >= threshold).mean())
    fpr = float((negatives >= threshold).mean())
    return Calibration(
        any_damage_threshold=threshold,
        target_recall=target_recall,
        measured_recall=measured_recall,
        negative_pixel_fpr=fpr,
        positive_pixels=int(positives.size),
        negative_pixels=int(negatives.size),
    )


def calibrate_case_threshold(
    case_scores: np.ndarray,
    negative_pixel_scores: np.ndarray,
    *,
    target_recall: float = 0.97,
    positive_pixels: int = 0,
    minimum_damage_coverage: float = 0.05,
    exterior_floor: float = 0.5,
    tile_size: int = 768,
    overlap: int = 192,
    horizontal_flip_tta: bool = True,
    mixed_precision: bool = False,
    triage_scales: tuple[float, ...] | list[float] = (1.0,),
) -> Calibration:
    """Choose a threshold giving every damage case equal weight.

    Each case score is the highest threshold at which the required fraction of
    that case's ground-truth damage pixels remains detected.
    """
    cases = np.asarray(case_scores, dtype=np.float64).reshape(-1)
    negatives = np.asarray(negative_pixel_scores, dtype=np.float64).reshape(-1)
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    if not 0 < minimum_damage_coverage <= 1:
        raise ValueError("minimum_damage_coverage must be in (0, 1]")
    if not 0 <= exterior_floor <= 1:
        raise ValueError("exterior_floor must be in [0, 1]")
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_size must be positive and overlap must be in [0, tile_size)")
    if cases.size == 0 or negatives.size == 0:
        raise ValueError("case calibration requires positive cases and negative pixels")
    descending = np.sort(cases)[::-1]
    required = int(np.ceil(target_recall * cases.size))
    threshold = float(descending[required - 1])
    return Calibration(
        any_damage_threshold=threshold,
        target_recall=target_recall,
        measured_recall=float((cases >= threshold).mean()),
        negative_pixel_fpr=float((negatives >= threshold).mean()),
        positive_pixels=int(positive_pixels),
        negative_pixels=int(negatives.size),
        calibration_basis="damage_case",
        positive_cases=int(cases.size),
        minimum_damage_coverage=minimum_damage_coverage,
        exterior_floor=exterior_floor,
        tile_size=tile_size,
        overlap=overlap,
        horizontal_flip_tta=horizontal_flip_tta,
        mixed_precision=mixed_precision,
        triage_scales=tuple(triage_scales),
    )
