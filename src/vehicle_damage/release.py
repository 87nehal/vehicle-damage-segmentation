from __future__ import annotations

from .metrics import wilson_upper_bound


HARD_NEGATIVE_SLICES = ("glare", "dirt", "panel_gap", "reflection", "shadow")
REQUIRED_DAMAGE_CLASSES = (
    "dent",
    "scratch",
    "crack_or_breakage",
    "paint_damage",
    "deformation_or_detachment",
)
MIN_CLEAN_TEST_IMAGES = 100
MIN_HARD_NEGATIVE_SLICE_IMAGES = 40
MIN_REGION_TEST_COMPONENTS = 100
MIN_REGION_PRECISION = 0.80
MIN_REGION_RECALL = 0.90
MIN_CASE_RECALL = 0.97
MIN_CASE_RECALL_WILSON_LOWER = 0.95
MIN_CLASS_CASES = 30
MIN_CLASS_RECALL = 0.90


def audit_release(report: dict) -> dict:
    """Apply the predeclared human-review product gates to a frozen report."""
    gates: list[dict] = []

    def gate(name: str, passed: bool, value: object, requirement: str) -> None:
        gates.append({
            "name": name,
            "passed": bool(passed),
            "value": value,
            "requirement": requirement,
        })

    recall = report.get("case_recall")
    lower = report.get("case_recall_wilson_95_lower")
    gate(
        "case_recall",
        recall is not None and recall >= MIN_CASE_RECALL,
        recall,
        f">= {MIN_CASE_RECALL:.2f}",
    )
    gate(
        "case_recall_wilson_95_lower",
        lower is not None and lower >= MIN_CASE_RECALL_WILSON_LOWER,
        lower,
        f">= {MIN_CASE_RECALL_WILSON_LOWER:.2f}",
    )

    per_class = report.get("per_class", {})
    for class_name in REQUIRED_DAMAGE_CLASSES:
        metrics = per_class.get(class_name, {})
        value = metrics.get("recall")
        cases = int(metrics.get("cases", 0))
        gate(
            f"per_class_recall:{class_name}",
            cases >= MIN_CLASS_CASES
            and value is not None
            and value >= MIN_CLASS_RECALL,
            {"cases": cases, "recall": value},
            (
                f">= {MIN_CLASS_CASES} independent cases and recall "
                f">= {MIN_CLASS_RECALL:.2f}"
            ),
        )

    region = report.get("region_metrics", {})
    region_tp = int(region.get("true_positive_components", 0))
    region_fp = int(region.get("false_positive_components", 0))
    region_fn = int(region.get("false_negative_components", 0))
    annotated_regions = region_tp + region_fn
    predicted_regions = region_tp + region_fp
    region_precision = region_tp / predicted_regions if predicted_regions else None
    region_recall = region_tp / annotated_regions if annotated_regions else None
    gate(
        "region_test_coverage",
        annotated_regions >= MIN_REGION_TEST_COMPONENTS,
        annotated_regions,
        f">= {MIN_REGION_TEST_COMPONENTS} annotated damage components",
    )
    gate(
        "region_precision",
        annotated_regions >= MIN_REGION_TEST_COMPONENTS
        and region_precision is not None
        and region_precision >= MIN_REGION_PRECISION,
        {
            "true_positive_components": region_tp,
            "false_positive_components": region_fp,
            "precision": region_precision,
        },
        f">= {MIN_REGION_PRECISION:.2f} with sufficient region-test coverage",
    )
    gate(
        "region_recall",
        annotated_regions >= MIN_REGION_TEST_COMPONENTS
        and region_recall is not None
        and region_recall >= MIN_REGION_RECALL,
        {
            "true_positive_components": region_tp,
            "false_negative_components": region_fn,
            "recall": region_recall,
        },
        f">= {MIN_REGION_RECALL:.2f} with sufficient region-test coverage",
    )

    clean_count = int(report.get("clean_groups", report.get("clean_images", 0)))
    clean_alerts = int(report.get("clean_alerts", 0))
    clean_rate = report.get("clean_false_alert_rate")
    clean_upper = wilson_upper_bound(clean_alerts, clean_count) if clean_count else None
    gate(
        "clean_test_coverage",
        clean_count >= MIN_CLEAN_TEST_IMAGES,
        clean_count,
        f">= {MIN_CLEAN_TEST_IMAGES} independently reviewed clean vehicle/claim groups",
    )
    gate(
        "clean_false_alert_rate",
        clean_count >= MIN_CLEAN_TEST_IMAGES
        and clean_rate is not None
        and clean_rate <= 0.05
        and clean_upper is not None
        and clean_upper <= 0.05,
        {"rate": clean_rate, "wilson_95_upper": clean_upper},
        "point rate and 95% Wilson upper bound <= 0.05",
    )

    slices = report.get("slices", {})
    for name in HARD_NEGATIVE_SLICES:
        values = slices.get(name, {})
        count = int(values.get("clean", 0))
        alerts = int(values.get("clean_alerts", 0))
        rate = values.get("clean_false_alert_rate")
        upper = wilson_upper_bound(alerts, count) if count else None
        gate(
            f"hard_negative_false_alert_rate:{name}",
            count >= MIN_HARD_NEGATIVE_SLICE_IMAGES
            and rate is not None
            and rate <= 0.10
            and upper is not None
            and upper <= 0.10,
            {
                "clean_groups": count,
                "false_alert_rate": rate,
                "wilson_95_upper": upper,
            },
            (
                f">= {MIN_HARD_NEGATIVE_SLICE_IMAGES} independently reviewed clean groups; "
                "point rate and 95% Wilson upper bound <= 0.10"
            ),
        )

    failed = [item["name"] for item in gates if not item["passed"]]
    return {
        "approved": not failed,
        "passed_gates": len(gates) - len(failed),
        "total_gates": len(gates),
        "failed_gates": failed,
        "gates": gates,
    }
