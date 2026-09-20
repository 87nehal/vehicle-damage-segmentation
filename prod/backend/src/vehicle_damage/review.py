from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


DECISIONS = {"clean", "damaged", "unusable"}
NUISANCE_TAGS = {
    "glare",
    "dirt",
    "panel_gap",
    "reflection",
    "shadow",
    "wet",
    "water_spots",
    "styling_crease",
    "decal",
    "previous_repair",
}
DISTANCES = {"close_up", "medium", "full_car"}
ANGLES = {"front", "rear", "side", "oblique", "top"}
LIGHTING = {"daylight", "night", "indoor", "mixed"}
CLEANLINESS = {"clean", "dirty", "wet"}
REQUIRED_COLUMNS = {
    "sample_id",
    "image",
    "group_id",
    "reviewer_id",
    "decision",
    "nuisance_tags",
    "distance",
    "angle",
    "lighting",
    "cleanliness",
    "notes",
}


@dataclass(frozen=True)
class ReviewRecord:
    sample_id: str
    image: str
    group_id: str
    reviewer_id: str
    decision: str
    nuisance_tags: tuple[str, ...]
    distance: str
    angle: str
    lighting: str
    cleanliness: str
    notes: str = ""

    @property
    def agreement_key(self) -> tuple[object, ...]:
        return (
            self.decision,
            self.nuisance_tags,
            self.distance,
            self.angle,
            self.lighting,
            self.cleanliness,
        )

    @property
    def manifest_tags(self) -> tuple[str, ...]:
        return (
            *self.nuisance_tags,
            f"distance:{self.distance}",
            f"angle:{self.angle}",
            f"lighting:{self.lighting}",
            f"cleanliness:{self.cleanliness}",
        )


def _choice(value: str, allowed: set[str], field: str, line: int) -> str:
    normalized = value.strip().lower()
    if normalized not in allowed:
        raise ValueError(
            f"review line {line}: {field} must be one of {sorted(allowed)}"
        )
    return normalized


def load_review_csv(path: str | Path) -> tuple[str, dict[str, ReviewRecord]]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"review CSV missing columns: {sorted(missing)}")
        records: dict[str, ReviewRecord] = {}
        reviewers: set[str] = set()
        for line, row in enumerate(reader, 2):
            sample_id = row["sample_id"].strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", sample_id):
                raise ValueError(f"review line {line}: sample_id must be SHA-256")
            if sample_id in records:
                raise ValueError(f"review line {line}: duplicate sample_id {sample_id}")
            reviewer = row["reviewer_id"].strip()
            if not reviewer:
                raise ValueError(f"review line {line}: reviewer_id is required")
            reviewers.add(reviewer)
            raw_tags = row["nuisance_tags"].replace(",", ";")
            tags = tuple(sorted({tag.strip().lower() for tag in raw_tags.split(";") if tag.strip()}))
            unknown = set(tags) - NUISANCE_TAGS
            if unknown:
                raise ValueError(f"review line {line}: unknown nuisance tags {sorted(unknown)}")
            records[sample_id] = ReviewRecord(
                sample_id=sample_id,
                image=row["image"].strip(),
                group_id=row["group_id"].strip(),
                reviewer_id=reviewer,
                decision=_choice(row["decision"], DECISIONS, "decision", line),
                nuisance_tags=tags,
                distance=_choice(row["distance"], DISTANCES, "distance", line),
                angle=_choice(row["angle"], ANGLES, "angle", line),
                lighting=_choice(row["lighting"], LIGHTING, "lighting", line),
                cleanliness=_choice(
                    row["cleanliness"], CLEANLINESS, "cleanliness", line
                ),
                notes=row["notes"].strip(),
            )
    if len(reviewers) != 1:
        raise ValueError("each review CSV must contain exactly one reviewer_id")
    if not records:
        raise ValueError("review CSV contains no records")
    return next(iter(reviewers)), records


def exact_agreements(
    reviewer_a: str,
    records_a: dict[str, ReviewRecord],
    reviewer_b: str,
    records_b: dict[str, ReviewRecord],
) -> tuple[dict[str, ReviewRecord], list[str]]:
    if reviewer_a == reviewer_b:
        raise ValueError("double review requires two distinct reviewer IDs")
    if records_a.keys() != records_b.keys():
        missing_a = sorted(records_b.keys() - records_a.keys())
        missing_b = sorted(records_a.keys() - records_b.keys())
        raise ValueError(
            f"review batches differ; missing from A={missing_a}, missing from B={missing_b}"
        )
    agreed: dict[str, ReviewRecord] = {}
    disagreements: list[str] = []
    for sample_id, first in records_a.items():
        second = records_b[sample_id]
        if first.group_id != second.group_id or first.agreement_key != second.agreement_key:
            disagreements.append(sample_id)
        else:
            agreed[sample_id] = first
    return agreed, disagreements


def apply_adjudication(
    agreed: dict[str, ReviewRecord],
    disagreements: list[str],
    reviewer_a: str,
    reviewer_b: str,
    adjudicator: str,
    adjudicated_records: dict[str, ReviewRecord],
) -> dict[str, ReviewRecord]:
    if adjudicator in {reviewer_a, reviewer_b}:
        raise ValueError("adjudicator must be distinct from both initial reviewers")
    missing = sorted(set(disagreements) - adjudicated_records.keys())
    if missing:
        raise ValueError(f"adjudication is missing disputed samples: {missing}")
    resolved = dict(agreed)
    for sample_id in disagreements:
        resolved[sample_id] = adjudicated_records[sample_id]
    return resolved
