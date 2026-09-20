import pytest

from vehicle_damage.collection_plan import (
    minimum_cases_with_false_negative_budget,
    minimum_negatives_with_alert_budget,
    plan_release_collection,
)
from vehicle_damage.release import REQUIRED_DAMAGE_CLASSES


def test_exact_wilson_buffer_sizes():
    assert minimum_cases_with_false_negative_budget(0) == 73
    assert minimum_cases_with_false_negative_budget(1) == 110
    assert minimum_cases_with_false_negative_budget(2) == 142
    assert minimum_negatives_with_alert_budget(
        0, maximum_rate=0.05, coverage_floor=100
    ) == 100
    assert minimum_negatives_with_alert_budget(
        2, maximum_rate=0.05, coverage_floor=100
    ) == 142
    assert minimum_negatives_with_alert_budget(
        0, maximum_rate=0.10, coverage_floor=40
    ) == 40
    assert minimum_negatives_with_alert_budget(
        1, maximum_rate=0.10, coverage_floor=40
    ) == 53


def test_plan_resets_consumed_evidence_and_reports_current_gaps():
    report = {
        "damage_cases": 98,
        "case_recall": 1.0,
        "case_recall_wilson_95_lower": 0.96,
        "clean_groups": 0,
        "per_class": {
            name: {"cases": 40, "recall": 0.95}
            for name in REQUIRED_DAMAGE_CLASSES
        },
        "region_metrics": {
            "true_positive_components": 90,
            "false_positive_components": 30,
            "false_negative_components": 10,
        },
        "slices": {},
    }

    plan = plan_release_collection(report)

    assert not plan["current_release_approved"]
    assert plan["current_report_coverage"]["damage_groups"] == 98
    assert plan["current_report_coverage"]["clean_group_shortfall"] == 100
    assert plan["fresh_untouched_test_minimum"]["damage_groups_overall"] == 73
    assert plan["buffered_collection_target"]["damage_groups_overall"] == 142
    assert plan["buffered_collection_target"]["clean_groups"] == 142
    assert set(
        plan["fresh_untouched_test_minimum"]["damage_groups_per_class"]
    ) == set(REQUIRED_DAMAGE_CLASSES)
    assert "region_precision" in plan[
        "measured_performance_failures_not_fixed_by_more_test_data"
    ]
    assert "clean_false_alert_rate" in plan["undercovered_or_unmeasured_gates"]
    assert "clean_false_alert_rate" not in plan[
        "measured_performance_failures_not_fixed_by_more_test_data"
    ]


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_quota_math_rejects_invalid_failure_counts(value):
    with pytest.raises(ValueError):
        minimum_cases_with_false_negative_budget(value)
