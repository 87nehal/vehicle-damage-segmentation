from vehicle_damage.release import REQUIRED_DAMAGE_CLASSES, audit_release


def _passing_classes():
    return {
        name: {"recall": 0.95, "hits": 38, "cases": 40}
        for name in REQUIRED_DAMAGE_CLASSES
    }


def test_release_audit_fails_closed_on_missing_clean_and_slice_data():
    report = {
        "case_recall": 1.0,
        "case_recall_wilson_95_lower": 0.96,
        "clean_images": 0,
        "clean_false_alert_rate": None,
        "per_class": {"dent": {"recall": 0.95}, "scratch": {"recall": 0.4}},
        "slices": {},
    }
    audit = audit_release(report)
    assert not audit["approved"]
    assert "per_class_recall:scratch" in audit["failed_gates"]
    assert "region_test_coverage" in audit["failed_gates"]
    assert "region_precision" in audit["failed_gates"]
    assert "region_recall" in audit["failed_gates"]
    assert "clean_test_coverage" in audit["failed_gates"]
    assert "hard_negative_false_alert_rate:glare" in audit["failed_gates"]


def test_release_audit_requires_statistically_meaningful_clean_slices():
    report = {
        "case_recall": 1.0,
        "case_recall_wilson_95_lower": 0.96,
        "clean_images": 100,
        "clean_alerts": 0,
        "clean_false_alert_rate": 0.0,
        "per_class": _passing_classes(),
        "region_metrics": {
            "true_positive_components": 100,
            "false_positive_components": 10,
            "false_negative_components": 5,
        },
        "slices": {
            name: {"clean": 40, "clean_alerts": 0, "clean_false_alert_rate": 0.0}
            for name in ("glare", "dirt", "panel_gap", "reflection", "shadow")
        },
    }
    audit = audit_release(report)
    assert audit["approved"]

    report["slices"]["glare"]["clean"] = 39
    audit = audit_release(report)
    assert not audit["approved"]
    assert "hard_negative_false_alert_rate:glare" in audit["failed_gates"]


def test_release_audit_recomputes_and_enforces_region_metrics():
    report = {
        "case_recall": 1.0,
        "case_recall_wilson_95_lower": 0.96,
        "clean_groups": 100,
        "clean_alerts": 0,
        "clean_false_alert_rate": 0.0,
        "per_class": _passing_classes(),
        "region_metrics": {
            "true_positive_components": 89,
            "false_positive_components": 0,
            "false_negative_components": 11,
            "precision": 1.0,
            "recall": 1.0,
        },
        "slices": {
            name: {"clean": 40, "clean_alerts": 0, "clean_false_alert_rate": 0.0}
            for name in ("glare", "dirt", "panel_gap", "reflection", "shadow")
        },
    }

    audit = audit_release(report)

    assert not audit["approved"]
    assert "region_recall" in audit["failed_gates"]
    assert "region_precision" not in audit["failed_gates"]


def test_release_audit_fails_closed_when_required_class_is_omitted():
    report = {
        "case_recall": 1.0,
        "case_recall_wilson_95_lower": 0.96,
        "clean_groups": 100,
        "clean_alerts": 0,
        "clean_false_alert_rate": 0.0,
        "per_class": _passing_classes(),
        "region_metrics": {
            "true_positive_components": 100,
            "false_positive_components": 0,
            "false_negative_components": 0,
        },
        "slices": {
            name: {"clean": 40, "clean_alerts": 0, "clean_false_alert_rate": 0.0}
            for name in ("glare", "dirt", "panel_gap", "reflection", "shadow")
        },
    }
    del report["per_class"]["paint_damage"]

    audit = audit_release(report)

    assert not audit["approved"]
    gate = next(
        item for item in audit["gates"]
        if item["name"] == "per_class_recall:paint_damage"
    )
    assert gate["passed"] is False
    assert gate["value"] == {"cases": 0, "recall": None}
