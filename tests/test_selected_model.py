import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_selected_development_model_artifacts_and_metrics_are_pinned():
    selection = json.loads(
        (ROOT / "runs" / "SELECTED_DEVELOPMENT_MODEL.json").read_text(
            encoding="utf-8"
        )
    )
    assert selection["production_approved"] is False
    for artifact in selection["artifacts"].values():
        path = ROOT / artifact["path"]
        assert path.is_file()
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]

    report_path = ROOT / selection["artifacts"]["validation_report"]["path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = selection["development_metrics"]
    assert metrics["damage_cases"] == report["damage_cases"]
    assert metrics["detected_cases"] == report["detected_cases"]
    assert metrics["case_recall"] == report["case_recall"]
    assert metrics["case_recall_wilson_95_lower"] == report[
        "case_recall_wilson_95_lower"
    ]
    assert metrics["pixel_precision"] == report["pixel_metrics"]["precision"]
    assert metrics["pixel_recall"] == report["pixel_metrics"]["recall"]
    assert metrics["background_false_positive_rate"] == report["pixel_metrics"][
        "false_positive_rate"
    ]
    assert metrics["region_precision"] == report["region_metrics"]["precision"]
    assert metrics["region_recall"] == report["region_metrics"]["recall"]
    assert metrics["region_true_positive_components"] == report["region_metrics"][
        "true_positive_components"
    ]
    assert metrics["region_false_positive_components"] == report["region_metrics"][
        "false_positive_components"
    ]
    assert metrics["region_false_negative_components"] == report["region_metrics"][
        "false_negative_components"
    ]
    assert report["inference_settings"]["triage_scales"] == [1.0, 1.5]
    assert report["inference_settings"]["segmentation_scales"] == [1.0]
    assert report["inference_settings"]["triage_minimum_component_pixels"] == 0
    assert report["inference_settings"][
        "segmentation_minimum_component_pixels"
    ] == 144
    assert metrics["per_class_case_recall"] == {
        name: values["recall"] for name, values in report["per_class"].items()
    }

    audit_path = ROOT / selection["artifacts"]["release_audit"]["path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert metrics["release_gates_passed"] == audit["passed_gates"]
    assert metrics["release_gates_total"] == audit["total_gates"]

    stress_path = ROOT / selection["artifacts"]["synthetic_stress_report"]["path"]
    stress = json.loads(stress_path.read_text(encoding="utf-8"))
    recalls = [stress["results"][name]["case_recall"] for name in stress["conditions"]]
    assert len(recalls) == metrics["synthetic_stress_conditions"]
    assert min(recalls) == metrics["synthetic_stress_minimum_case_recall"]
    previous_report_path = (
        ROOT / selection["artifacts"]["previous_selected_validation_report"]["path"]
    )
    previous_report = json.loads(previous_report_path.read_text(encoding="utf-8"))
    assert report["pixel_metrics"]["precision"] > previous_report["pixel_metrics"]["precision"]
    assert report["pixel_metrics"]["recall"] > previous_report["pixel_metrics"]["recall"]
    assert report["pixel_metrics"]["false_positive_rate"] < previous_report["pixel_metrics"]["false_positive_rate"]
    assert report["region_metrics"]["true_positive_components"] > previous_report["region_metrics"]["true_positive_components"]
    assert report["region_metrics"]["false_positive_components"] < previous_report["region_metrics"]["false_positive_components"]
    assert report["region_metrics"]["false_negative_components"] < previous_report["region_metrics"]["false_negative_components"]
    assert {
        name: values["hits"] for name, values in report["per_class"].items()
    } == {
        name: values["hits"] for name, values in previous_report["per_class"].items()
    }

    previous_stress_path = (
        ROOT / selection["artifacts"]["previous_selected_stress_report"]["path"]
    )
    previous_stress = json.loads(previous_stress_path.read_text(encoding="utf-8"))
    for condition in stress["conditions"]:
        current = stress["results"][condition]
        previous = previous_stress["results"][condition]
        assert current["detected_cases"] >= previous["detected_cases"]
        assert current["pixel_metrics"]["recall"] >= previous["pixel_metrics"]["recall"]
        assert current["region_metrics"]["true_positive_components"] >= previous[
            "region_metrics"
        ]["true_positive_components"]
        assert current["region_metrics"]["false_negative_components"] <= previous[
            "region_metrics"
        ]["false_negative_components"]
        assert {
            name: values["hits"] for name, values in current["per_class"].items()
        } == {
            name: values["hits"] for name, values in previous["per_class"].items()
        }

    benchmark_path = ROOT / selection["artifacts"]["latency_benchmark"]["path"]
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    assert metrics["latency_ms_median"] == benchmark["latency_ms_median"]
    assert metrics["latency_ms_p95"] == benchmark["latency_ms_p95"]
    assert metrics["throughput_images_per_second"] == benchmark[
        "throughput_images_per_second_from_median"
    ]
    assert metrics["peak_gpu_memory_bytes"] == benchmark["peak_gpu_memory_bytes"]

    routing_path = ROOT / selection["artifacts"]["decision_routing_smoke_report"]["path"]
    routing = json.loads(routing_path.read_text(encoding="utf-8"))
    assert routing["decision"] == "manual_review_required"
    assert routing["automated_decision_allowed"] is False
    assert routing["manual_review"] is True
    assert routing["manual_review_required"] is True
    assert routing["recapture_required"] is False
    assert routing["decision_reasons"] == [
        "quality:possible_specular_glare",
        "triage_only_damage_evidence",
        "damage_type_disagreement",
    ]
    assert routing["triage_only_fraction"] > 0
    assert routing["damage_type_disagreement_fraction"] > 0
    assert routing["segmentation_only_fraction"] == 0

    plan_path = ROOT / selection["artifacts"]["release_data_collection_plan"]["path"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["current_release_approved"] is False
    assert plan["fresh_untouched_test_minimum"]["damage_groups_overall"] == 73
    assert plan["fresh_untouched_test_minimum"]["clean_groups"] == 100
    assert plan["buffered_collection_target"]["damage_groups_overall"] == 142
    assert plan["buffered_collection_target"]["clean_groups"] == 142
    assert set(
        plan["measured_performance_failures_not_fixed_by_more_test_data"]
    ) == {
        "per_class_recall:dent",
        "per_class_recall:scratch",
        "per_class_recall:paint_damage",
        "per_class_recall:deformation_or_detachment",
        "region_precision",
        "region_recall",
    }
    assert "clean_false_alert_rate" in plan["undercovered_or_unmeasured_gates"]

    detail_path = (
        ROOT / selection["artifacts"]["rejected_detail_refiner_experiment"]["path"]
    )
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    assert detail["status"] == "development_experiment_rejected_not_selected"
    assert detail["selected_model_changed"] is False
    assert detail["lineage"]["changed_common_tensors"] == 0
    assert detail["validation_comparison"]["candidate"][
        "pixel_recall_retention"
    ] >= 0.999
    assert detail["synthetic_stress_guards"][
        "observed_minimum_pixel_recall_retention"
    ] < detail["synthetic_stress_guards"][
        "required_minimum_pixel_recall_retention"
    ]
    assert detail["synthetic_stress_guards"][
        "conditions_below_pixel_recall_guard"
    ] == ["underexposure", "blur", "low_resolution", "viewpoint"]
    assert detail["efficiency"]["candidate_latency_ms_median"] > detail[
        "efficiency"
    ]["selected_latency_ms_median"]
    for artifact in detail["artifacts"].values():
        artifact_path = ROOT / artifact["path"]
        assert artifact_path.stat().st_size == artifact["bytes"]
        assert (
            hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            == artifact["sha256"]
        )


def test_native_speed_diagnostic_cannot_silently_replace_recall_profile():
    selection = json.loads(
        (ROOT / "runs" / "SELECTED_DEVELOPMENT_MODEL.json").read_text(
            encoding="utf-8"
        )
    )
    selected_profile = json.loads(
        (ROOT / selection["artifacts"]["development_profile"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    tradeoff = json.loads(
        (
            ROOT
            / "runs"
            / "pilot_cc0_dinov2_4layer_role_split"
            / "SPEED_RECALL_TRADEOFF.json"
        ).read_text(encoding="utf-8")
    )
    assert tradeoff["status"] == "development_diagnostic_not_selected"
    assert selected_profile["triage_scales"] == [1.0, 1.5]
    assert tradeoff["native_diagnostic_profile"]["triage_scales"] == [1.0]
    assert tradeoff["synthetic_stress_case_detection"][
        "native_conditions_at_97_of_98"
    ] == 3
    assert tradeoff["synthetic_stress_case_detection"][
        "selected_multiscale_conditions_at_98_of_98"
    ] == 11
    assert tradeoff["native_benchmark"]["latency_ms_median"] < tradeoff[
        "selected_multiscale_benchmark"
    ]["latency_ms_median"]
    for artifact_name in (
        "lineage_report",
        "native_robustness_report",
    ):
        artifact = tradeoff["evidence_transfer"][artifact_name]
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    for artifact_name in (
        "native_diagnostic_profile",
        "native_benchmark",
        "selected_multiscale_benchmark",
    ):
        artifact = tradeoff[artifact_name]
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
