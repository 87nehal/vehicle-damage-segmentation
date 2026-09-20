from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import ndimage

from .metrics import wilson_lower_bound, wilson_upper_bound


def component_counts(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    minimum_pixels: int = 16,
    minimum_iou: float = 0.10,
) -> tuple[int, int, int, int]:
    """Return matched, false-positive, missed, and predicted component counts."""
    if minimum_pixels <= 0:
        raise ValueError("minimum_pixels must be positive")
    if not 0 < minimum_iou <= 1:
        raise ValueError("minimum_iou must be in (0, 1]")
    structure = np.ones((3, 3), dtype=np.uint8)

    def filtered_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
        labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
        if not count:
            return labels.astype(np.int32, copy=False), 0
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        keep = sizes >= minimum_pixels
        keep[0] = False
        return ndimage.label(keep[labels], structure=structure)

    target_labels, target_count = filtered_labels(target)
    prediction_labels, prediction_count = filtered_labels(prediction)
    if not target_count or not prediction_count:
        return 0, prediction_count, target_count, prediction_count

    encoded = (
        target_labels.astype(np.int64) * (prediction_count + 1)
        + prediction_labels.astype(np.int64)
    )
    intersections = np.bincount(
        encoded.ravel(), minlength=(target_count + 1) * (prediction_count + 1)
    ).reshape(target_count + 1, prediction_count + 1)
    target_areas = np.bincount(target_labels.ravel(), minlength=target_count + 1)
    prediction_areas = np.bincount(
        prediction_labels.ravel(), minlength=prediction_count + 1
    )
    candidates: list[tuple[float, int, int]] = []
    for target_id, prediction_id in np.argwhere(intersections[1:, 1:] > 0) + 1:
        intersection = intersections[target_id, prediction_id]
        union = target_areas[target_id] + prediction_areas[prediction_id] - intersection
        iou = float(intersection / union)
        if iou >= minimum_iou:
            candidates.append((iou, int(target_id), int(prediction_id)))
    matched_target: set[int] = set()
    matched_prediction: set[int] = set()
    for _, target_id, prediction_id in sorted(candidates, reverse=True):
        if target_id not in matched_target and prediction_id not in matched_prediction:
            matched_target.add(target_id)
            matched_prediction.add(prediction_id)
    matched = len(matched_target)
    return matched, prediction_count - matched, target_count - matched, prediction_count


class CaseEvaluator:
    """Case-level metrics where any overlap counts as localization success."""

    def __init__(
        self,
        class_names: list[str],
        minimum_damage_coverage: float = 0.05,
        *,
        minimum_component_pixels: int = 16,
        minimum_region_iou: float = 0.10,
    ) -> None:
        if not 0 < minimum_damage_coverage <= 1:
            raise ValueError("minimum_damage_coverage must be in (0, 1]")
        if minimum_component_pixels <= 0:
            raise ValueError("minimum_component_pixels must be positive")
        if not 0 < minimum_region_iou <= 1:
            raise ValueError("minimum_region_iou must be in (0, 1]")
        self.class_names = class_names
        self.minimum_damage_coverage = minimum_damage_coverage
        self.minimum_component_pixels = minimum_component_pixels
        self.minimum_region_iou = minimum_region_iou
        self.pixel_confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        self.images_evaluated = 0
        self.region_tp = 0
        self.region_fp = 0
        self.region_fn = 0
        self.clean_views = 0
        self.clean_view_false_positive_components = 0
        self._anonymous_group = 0
        self._groups: dict[str, dict] = {}

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        tags: tuple[str, ...],
        group_id: str | None = None,
        *,
        case_prediction: np.ndarray | None = None,
    ) -> None:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have the same shape")
        if case_prediction is None:
            case_prediction = prediction
        if case_prediction.shape != target.shape:
            raise ValueError("case_prediction and target must have the same shape")
        class_count = len(self.class_names)
        if prediction.size and (
            prediction.min() < 0 or prediction.max() >= class_count
            or target.min() < 0 or target.max() >= class_count
            or case_prediction.min() < 0 or case_prediction.max() >= class_count
        ):
            raise ValueError("prediction, case_prediction, or target contains an out-of-range class id")
        encoded = target.astype(np.int64).ravel() * class_count + prediction.astype(np.int64).ravel()
        self.pixel_confusion += np.bincount(encoded, minlength=class_count**2).reshape(
            class_count, class_count
        )
        region_tp, region_fp, region_fn, predicted_components = component_counts(
            prediction > 0,
            target > 0,
            minimum_pixels=self.minimum_component_pixels,
            minimum_iou=self.minimum_region_iou,
        )
        self.region_tp += region_tp
        self.region_fp += region_fp
        self.region_fn += region_fn
        if not (target > 0).any():
            self.clean_views += 1
            self.clean_view_false_positive_components += predicted_components
        self.images_evaluated += 1
        if group_id is None:
            group_id = f"__anonymous_image_{self._anonymous_group}"
            self._anonymous_group += 1
        state = self._groups.setdefault(
            group_id,
            {
                "actual_positive": False,
                "case_hit": False,
                "predicted_positive": False,
                "tags": set(),
                "class_present": np.zeros(class_count, dtype=bool),
                "class_hit": np.zeros(class_count, dtype=bool),
            },
        )
        actual_positive = bool((target > 0).any())
        predicted_positive = bool((case_prediction > 0).any())
        target_pixels = int((target > 0).sum())
        intersection = int(((case_prediction > 0) & (target > 0)).sum())
        overlap = bool(target_pixels and intersection / target_pixels >= self.minimum_damage_coverage)
        state["actual_positive"] |= actual_positive
        state["case_hit"] |= overlap
        state["predicted_positive"] |= predicted_positive
        state["tags"].update(tags)
        for class_id in range(1, len(self.class_names)):
            class_target = target == class_id
            if class_target.any():
                state["class_present"][class_id] = True
                class_coverage = (
                    ((case_prediction == class_id) & class_target).sum()
                    / class_target.sum()
                )
                state["class_hit"][class_id] |= bool(
                    class_coverage >= self.minimum_damage_coverage
                )

    def report(self) -> dict:
        cases = sum(bool(state["actual_positive"]) for state in self._groups.values())
        case_hits = sum(
            bool(state["actual_positive"] and state["case_hit"])
            for state in self._groups.values()
        )
        clean = len(self._groups) - cases
        clean_alerts = sum(
            bool(not state["actual_positive"] and state["predicted_positive"])
            for state in self._groups.values()
        )
        recall = case_hits / cases if cases else 0.0
        false_alert = clean_alerts / clean if clean else None
        class_totals = np.zeros(len(self.class_names), dtype=np.int64)
        class_hits = np.zeros(len(self.class_names), dtype=np.int64)
        slices: dict[str, dict[str, int]] = defaultdict(
            lambda: {"cases": 0, "hits": 0, "clean": 0, "clean_alerts": 0}
        )
        for state in self._groups.values():
            for class_id in range(1, len(self.class_names)):
                class_totals[class_id] += int(state["class_present"][class_id])
                class_hits[class_id] += int(
                    state["class_present"][class_id] and state["class_hit"][class_id]
                )
            for tag in ("overall", *sorted(state["tags"])):
                if state["actual_positive"]:
                    slices[tag]["cases"] += 1
                    slices[tag]["hits"] += int(state["case_hit"])
                else:
                    slices[tag]["clean"] += 1
                    slices[tag]["clean_alerts"] += int(state["predicted_positive"])
        class_recall = {}
        for i in range(1, len(self.class_names)):
            tp = int(self.pixel_confusion[i, i])
            fp = int(self.pixel_confusion[:, i].sum() - tp)
            fn = int(self.pixel_confusion[i, :].sum() - tp)
            class_recall[self.class_names[i]] = {
                "recall": float(class_hits[i] / class_totals[i]) if class_totals[i] else None,
                "hits": int(class_hits[i]),
                "cases": int(class_totals[i]),
                "recall_wilson_95_lower": (
                    wilson_lower_bound(int(class_hits[i]), int(class_totals[i]))
                    if class_totals[i] else None
                ),
                "pixel_precision": tp / (tp + fp) if tp + fp else None,
                "pixel_recall": tp / (tp + fn) if tp + fn else None,
                "pixel_iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            }
        tp = int(self.pixel_confusion[1:, 1:].sum())
        fp = int(self.pixel_confusion[0, 1:].sum())
        fn = int(self.pixel_confusion[1:, 0].sum())
        tn = int(self.pixel_confusion[0, 0])
        slice_report = {}
        for tag, values in sorted(slices.items()):
            slice_cases, slice_clean = values["cases"], values["clean"]
            slice_report[tag] = {
                **values,
                "case_recall": values["hits"] / slice_cases if slice_cases else None,
                "case_recall_wilson_95_lower": (
                    wilson_lower_bound(values["hits"], slice_cases) if slice_cases else None
                ),
                "clean_false_alert_rate": (
                    values["clean_alerts"] / slice_clean if slice_clean else None
                ),
                "clean_false_alert_rate_wilson_95_upper": (
                    wilson_upper_bound(values["clean_alerts"], slice_clean)
                    if slice_clean else None
                ),
            }
        return {
            "evaluation_unit": "group_id",
            "images_evaluated": self.images_evaluated,
            "groups_evaluated": len(self._groups),
            "damage_cases": cases,
            "detected_cases": case_hits,
            "case_recall": recall,
            "case_recall_wilson_95_lower": wilson_lower_bound(case_hits, cases),
            "clean_groups": clean,
            "clean_images": clean,
            "clean_alerts": clean_alerts,
            "clean_false_alert_rate": false_alert,
            "clean_false_alert_rate_wilson_95_upper": (
                wilson_upper_bound(clean_alerts, clean) if clean else None
            ),
            "minimum_damage_coverage": self.minimum_damage_coverage,
            "pixel_metrics": {
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "false_positive_rate": fp / (fp + tn) if fp + tn else None,
                "predicted_damage_fraction": (tp + fp) / self.pixel_confusion.sum()
                if self.pixel_confusion.sum() else None,
            },
            "region_metrics": {
                "minimum_component_pixels": self.minimum_component_pixels,
                "minimum_iou": self.minimum_region_iou,
                "true_positive_components": self.region_tp,
                "false_positive_components": self.region_fp,
                "false_negative_components": self.region_fn,
                "precision": (
                    self.region_tp / (self.region_tp + self.region_fp)
                    if self.region_tp + self.region_fp else None
                ),
                "recall": (
                    self.region_tp / (self.region_tp + self.region_fn)
                    if self.region_tp + self.region_fn else None
                ),
                "clean_views": self.clean_views,
                "false_positive_components_on_clean_views": (
                    self.clean_view_false_positive_components
                ),
                "false_positive_components_per_clean_view": (
                    self.clean_view_false_positive_components / self.clean_views
                    if self.clean_views else None
                ),
            },
            "per_class": class_recall,
            "slices": slice_report,
        }
