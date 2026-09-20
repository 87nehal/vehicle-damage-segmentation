import numpy as np

from vehicle_damage.evaluation import CaseEvaluator, component_counts


def test_component_counts_matches_regions_one_to_one():
    target = np.zeros((20, 20), dtype=np.uint8)
    target[2:7, 2:7] = 1
    target[12:17, 12:17] = 1
    prediction = np.zeros_like(target)
    prediction[2:7, 2:7] = 1
    prediction[0:2, 15:18] = 1

    counts = component_counts(
        prediction,
        target,
        minimum_pixels=4,
        minimum_iou=0.10,
    )

    assert counts == (1, 1, 1, 2)


def test_component_counts_filters_tiny_regions():
    target = np.zeros((10, 10), dtype=np.uint8)
    target[2:5, 2:5] = 1
    prediction = target.copy()
    prediction[9, 9] = 1

    assert component_counts(prediction, target, minimum_pixels=4) == (1, 0, 0, 1)


def test_case_and_hard_negative_slice_metrics():
    evaluator = CaseEvaluator(["background", "dent", "scratch"])
    target = np.zeros((8, 8), dtype=np.uint8)
    target[2:4, 2:4] = 1
    prediction = np.zeros_like(target)
    prediction[3:5, 3:5] = 1
    evaluator.update(prediction, target, ("glare",))
    evaluator.update(np.ones_like(target), np.zeros_like(target), ("panel_gap",))
    report = evaluator.report()
    assert report["case_recall"] == 1.0
    assert report["clean_false_alert_rate"] == 1.0
    assert report["per_class"]["dent"]["recall"] == 1.0
    assert report["pixel_metrics"]["true_positive"] == 1
    assert report["pixel_metrics"]["false_positive"] == 67
    assert report["pixel_metrics"]["false_negative"] == 3
    assert report["per_class"]["dent"]["pixel_recall"] == 0.25
    assert report["slices"]["glare"]["case_recall"] == 1.0
    assert report["slices"]["panel_gap"]["clean_false_alert_rate"] == 1.0
    assert report["clean_false_alert_rate_wilson_95_upper"] == 1.0
    assert report["per_class"]["dent"]["recall_wilson_95_lower"] > 0
    assert report["region_metrics"]["clean_views"] == 1
    assert report["region_metrics"]["false_positive_components_per_clean_view"] == 1.0


def test_multiple_views_are_counted_once_per_group():
    evaluator = CaseEvaluator(["background", "dent"])
    empty = np.zeros((8, 8), dtype=np.uint8)
    alerted = empty.copy()
    alerted[0, 0] = 1
    evaluator.update(empty, empty, ("glare",), "clean-vehicle")
    evaluator.update(alerted, empty, ("reflection",), "clean-vehicle")

    damaged = empty.copy()
    damaged[2:4, 2:4] = 1
    evaluator.update(empty, damaged, ("night",), "damaged-vehicle")
    evaluator.update(damaged, damaged, ("close_up",), "damaged-vehicle")
    report = evaluator.report()

    assert report["images_evaluated"] == 4
    assert report["groups_evaluated"] == 2
    assert report["damage_cases"] == 1
    assert report["detected_cases"] == 1
    assert report["clean_groups"] == 1
    assert report["clean_alerts"] == 1
    assert report["slices"]["glare"]["clean"] == 1
    assert report["slices"]["reflection"]["clean"] == 1


def test_case_prediction_can_preserve_triage_recall_with_a_precise_empty_mask():
    evaluator = CaseEvaluator(["background", "dent"])
    target = np.zeros((8, 8), dtype=np.uint8)
    target[2:6, 2:6] = 1
    precise = np.zeros_like(target)
    triage = target.copy()

    evaluator.update(
        precise,
        target,
        ("glare",),
        case_prediction=triage,
    )
    report = evaluator.report()

    assert report["detected_cases"] == 1
    assert report["case_recall"] == 1.0
    assert report["pixel_metrics"]["true_positive"] == 0
    assert report["pixel_metrics"]["false_negative"] == 16
    assert report["per_class"]["dent"]["recall"] == 1.0
    assert report["per_class"]["dent"]["pixel_recall"] == 0.0
