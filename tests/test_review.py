import csv

import pytest

from vehicle_damage.review import (
    REQUIRED_COLUMNS,
    apply_adjudication,
    exact_agreements,
    load_review_csv,
)


def _write_review(path, reviewer, *, nuisance="glare", decision="clean"):
    row = {
        "sample_id": "a" * 64,
        "image": "car.jpg",
        "group_id": "vehicle-1",
        "reviewer_id": reviewer,
        "decision": decision,
        "nuisance_tags": nuisance,
        "distance": "full_car",
        "angle": "oblique",
        "lighting": "daylight",
        "cleanliness": "clean",
        "notes": "",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerow(row)


def test_exact_double_review_agreement(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_review(first, "reviewer-a")
    _write_review(second, "reviewer-b")
    reviewer_a, records_a = load_review_csv(first)
    reviewer_b, records_b = load_review_csv(second)

    agreed, disagreements = exact_agreements(reviewer_a, records_a, reviewer_b, records_b)

    assert list(agreed) == ["a" * 64]
    assert disagreements == []
    assert agreed["a" * 64].manifest_tags == (
        "glare", "distance:full_car", "angle:oblique", "lighting:daylight", "cleanliness:clean"
    )


def test_disagreement_is_not_accepted_and_same_reviewer_is_rejected(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_review(first, "reviewer-a", nuisance="glare")
    _write_review(second, "reviewer-b", nuisance="reflection")
    reviewer_a, records_a = load_review_csv(first)
    reviewer_b, records_b = load_review_csv(second)
    agreed, disagreements = exact_agreements(reviewer_a, records_a, reviewer_b, records_b)
    assert agreed == {}
    assert disagreements == ["a" * 64]
    with pytest.raises(ValueError, match="distinct"):
        exact_agreements(reviewer_a, records_a, reviewer_a, records_a)

    adjudication = tmp_path / "adjudication.csv"
    _write_review(adjudication, "automotive-assessor", nuisance="panel_gap")
    adjudicator, adjudicated_records = load_review_csv(adjudication)
    resolved = apply_adjudication(
        agreed,
        disagreements,
        reviewer_a,
        reviewer_b,
        adjudicator,
        adjudicated_records,
    )
    assert resolved["a" * 64].nuisance_tags == ("panel_gap",)
    with pytest.raises(ValueError, match="distinct"):
        apply_adjudication(
            agreed,
            disagreements,
            reviewer_a,
            reviewer_b,
            reviewer_a,
            adjudicated_records,
        )
