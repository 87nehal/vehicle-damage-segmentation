from __future__ import annotations

from .metrics import wilson_lower_bound, wilson_upper_bound
from .release import (
    HARD_NEGATIVE_SLICES,
    MIN_CASE_RECALL,
    MIN_CASE_RECALL_WILSON_LOWER,
    MIN_CLASS_CASES,
    MIN_CLASS_RECALL,
    MIN_CLEAN_TEST_IMAGES,
    MIN_HARD_NEGATIVE_SLICE_IMAGES,
    MIN_REGION_TEST_COMPONENTS,
    REQUIRED_DAMAGE_CLASSES,
    audit_release,
)


def minimum_cases_with_false_negative_budget(false_negatives: int) -> int:
    """Smallest test size meeting both overall recall gates for a fixed FN budget."""
    if (
        isinstance(false_negatives, bool)
        or not isinstance(false_negatives, int)
        or false_negatives < 0
    ):
        raise ValueError("false_negatives must be a non-negative integer")
    for total in range(max(1, false_negatives + 1), 1_000_001):
        hits = total - false_negatives
        if (
            hits / total >= MIN_CASE_RECALL
            and wilson_lower_bound(hits, total) >= MIN_CASE_RECALL_WILSON_LOWER
        ):
            return total
    raise ValueError("no sample size found within search bound")


def minimum_negatives_with_alert_budget(
    false_alerts: int,
    *,
    maximum_rate: float,
    coverage_floor: int,
) -> int:
    """Smallest clean set meeting point and Wilson-upper gates."""
    if (
        isinstance(false_alerts, bool)
        or not isinstance(false_alerts, int)
        or false_alerts < 0
    ):
        raise ValueError("false_alerts must be a non-negative integer")
    if not 0 < maximum_rate < 1:
        raise ValueError("maximum_rate must be in (0, 1)")
    if (
        isinstance(coverage_floor, bool)
        or not isinstance(coverage_floor, int)
        or coverage_floor <= 0
    ):
        raise ValueError("coverage_floor must be a positive integer")
    for total in range(max(coverage_floor, false_alerts + 1), 1_000_001):
        if (
            false_alerts / total <= maximum_rate
            and wilson_upper_bound(false_alerts, total) <= maximum_rate
        ):
            return total
    raise ValueError("no sample size found within search bound")


def plan_release_collection(report: dict) -> dict:
    """Translate release gates into fresh-test floors and useful buffer targets."""
    audit = audit_release(report)
    per_class = report.get("per_class", {})
    slices = report.get("slices", {})
    current_classes = {
        name: {
            "groups": int(per_class.get(name, {}).get("cases", 0)),
            "minimum_group_shortfall": max(
                0, MIN_CLASS_CASES - int(per_class.get(name, {}).get("cases", 0))
            ),
            "measured_recall": per_class.get(name, {}).get("recall"),
        }
        for name in REQUIRED_DAMAGE_CLASSES
    }
    current_slices = {
        name: {
            "clean_groups": int(slices.get(name, {}).get("clean", 0)),
            "minimum_group_shortfall": max(
                0,
                MIN_HARD_NEGATIVE_SLICE_IMAGES
                - int(slices.get(name, {}).get("clean", 0)),
            ),
        }
        for name in HARD_NEGATIVE_SLICES
    }
    region = report.get("region_metrics", {})
    annotated_regions = int(region.get("true_positive_components", 0)) + int(
        region.get("false_negative_components", 0)
    )
    clean_groups = int(report.get("clean_groups", report.get("clean_images", 0)))

    minimum_damage_groups = minimum_cases_with_false_negative_budget(0)
    buffered_damage_groups = minimum_cases_with_false_negative_budget(2)
    minimum_clean_groups = minimum_negatives_with_alert_budget(
        0,
        maximum_rate=0.05,
        coverage_floor=MIN_CLEAN_TEST_IMAGES,
    )
    buffered_clean_groups = minimum_negatives_with_alert_budget(
        2,
        maximum_rate=0.05,
        coverage_floor=MIN_CLEAN_TEST_IMAGES,
    )
    minimum_slice_groups = minimum_negatives_with_alert_budget(
        0,
        maximum_rate=0.10,
        coverage_floor=MIN_HARD_NEGATIVE_SLICE_IMAGES,
    )
    buffered_slice_groups = minimum_negatives_with_alert_budget(
        1,
        maximum_rate=0.10,
        coverage_floor=MIN_HARD_NEGATIVE_SLICE_IMAGES,
    )

    measured_performance_failures: list[str] = []
    damage_groups = int(report.get("damage_cases", 0))
    if damage_groups >= minimum_damage_groups:
        measured_performance_failures.extend(
            name
            for name in ("case_recall", "case_recall_wilson_95_lower")
            if name in audit["failed_gates"]
        )
    measured_performance_failures.extend(
        f"per_class_recall:{name}"
        for name, values in current_classes.items()
        if values["groups"] >= MIN_CLASS_CASES
        and f"per_class_recall:{name}" in audit["failed_gates"]
    )
    if annotated_regions >= MIN_REGION_TEST_COMPONENTS:
        measured_performance_failures.extend(
            name
            for name in ("region_precision", "region_recall")
            if name in audit["failed_gates"]
        )
    if clean_groups >= MIN_CLEAN_TEST_IMAGES and "clean_false_alert_rate" in audit[
        "failed_gates"
    ]:
        measured_performance_failures.append("clean_false_alert_rate")
    for name, values in current_slices.items():
        gate = f"hard_negative_false_alert_rate:{name}"
        if (
            values["clean_groups"] >= MIN_HARD_NEGATIVE_SLICE_IMAGES
            and gate in audit["failed_gates"]
        ):
            measured_performance_failures.append(gate)
    undercovered_or_unmeasured = [
        name
        for name in audit["failed_gates"]
        if name not in measured_performance_failures
    ]
    return {
        "status": (
            "release_ready"
            if audit["approved"]
            else "fresh_evidence_and_or_model_improvement_required"
        ),
        "current_release_approved": audit["approved"],
        "current_failed_gates": audit["failed_gates"],
        "current_report_coverage": {
            "damage_groups": damage_groups,
            "clean_groups": clean_groups,
            "clean_group_shortfall": max(0, MIN_CLEAN_TEST_IMAGES - clean_groups),
            "annotated_damage_components": annotated_regions,
            "annotated_component_shortfall": max(
                0, MIN_REGION_TEST_COMPONENTS - annotated_regions
            ),
            "per_class": current_classes,
            "hard_negative_slices": current_slices,
        },
        "fresh_untouched_test_minimum": {
            "damage_groups_overall": minimum_damage_groups,
            "maximum_false_negatives_at_that_size": 0,
            "damage_groups_per_class": {
                name: MIN_CLASS_CASES for name in REQUIRED_DAMAGE_CLASSES
            },
            "minimum_per_class_recall": MIN_CLASS_RECALL,
            "annotated_damage_components": MIN_REGION_TEST_COMPONENTS,
            "clean_groups": minimum_clean_groups,
            "maximum_clean_false_alerts_at_that_size": 0,
            "clean_groups_per_hard_negative_slice": {
                name: minimum_slice_groups for name in HARD_NEGATIVE_SLICES
            },
            "maximum_slice_false_alerts_at_that_size": 0,
        },
        "buffered_collection_target": {
            "damage_groups_overall": buffered_damage_groups,
            "false_negative_budget": 2,
            "clean_groups": buffered_clean_groups,
            "clean_false_alert_budget": 2,
            "clean_groups_per_hard_negative_slice": {
                name: buffered_slice_groups for name in HARD_NEGATIVE_SLICES
            },
            "false_alert_budget_per_hard_negative_slice": 1,
        },
        "measured_performance_failures_not_fixed_by_more_test_data": (
            measured_performance_failures
        ),
        "undercovered_or_unmeasured_gates": undercovered_or_unmeasured,
        "rules": [
            "Use independent vehicle/claim groups; multiple views do not increase sample count.",
            "One clean group may count in multiple nuisance slices only when every tag is true.",
            "Do not reuse checkpoint-selection, calibration, or consumed validation groups.",
            "Freeze the complete test manifest before scoring it.",
            "Collection can close coverage gaps but cannot repair failed precision or recall.",
        ],
        "statistical_basis": {
            "overall_recall_point_minimum": MIN_CASE_RECALL,
            "overall_recall_wilson_95_lower_minimum": (
                MIN_CASE_RECALL_WILSON_LOWER
            ),
            "clean_false_alert_point_and_wilson_95_upper_maximum": 0.05,
            "slice_false_alert_point_and_wilson_95_upper_maximum": 0.10,
        },
    }
