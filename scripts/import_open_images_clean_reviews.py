from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image

from vehicle_damage.manifest import load_manifest, validate_manifest
from vehicle_damage.open_images_intake import load_license_approved_candidates
from vehicle_damage.review import apply_adjudication, exact_agreements, load_review_csv
from vehicle_damage.taxonomy import DEFAULT_CLASSES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create clean-only training rows after per-image rights approval and "
            "two independent exact damage reviews"
        )
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--license-review", required=True)
    parser.add_argument("--review-a", required=True)
    parser.add_argument("--review-b", required=True)
    parser.add_argument("--adjudication")
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()

    output_manifest = Path(args.output_manifest).resolve()
    if output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite {output_manifest}")
    approved, license_report = load_license_approved_candidates(
        args.candidates, args.license_review
    )
    expected = {str(row["image_sha256"]): row for row in approved}
    reviewer_a, records_a = load_review_csv(args.review_a)
    reviewer_b, records_b = load_review_csv(args.review_b)
    if set(records_a) != set(expected) or set(records_b) != set(expected):
        raise ValueError("both damage reviews must contain every license-approved candidate")
    agreed, disagreements = exact_agreements(
        reviewer_a, records_a, reviewer_b, records_b
    )
    adjudicator = None
    resolved = agreed
    if disagreements:
        if not args.adjudication:
            raise ValueError(
                "damage-review disagreements require a distinct third-assessor adjudication"
            )
        adjudicator, adjudication_records = load_review_csv(args.adjudication)
        if set(adjudication_records) != set(disagreements):
            raise ValueError("adjudication must contain exactly the disputed candidates")
        resolved = apply_adjudication(
            agreed,
            disagreements,
            reviewer_a,
            reviewer_b,
            adjudicator,
            adjudication_records,
        )
    elif args.adjudication:
        raise ValueError("adjudication was supplied but there are no disagreements")

    output_root = output_manifest.parent
    output_root.mkdir(parents=True, exist_ok=True)
    mask_dir = output_root / "reviewed_clean_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir = output_root / "orientation_normalized_images"
    review_digest = str(license_report["license_review_sha256"])
    rows = []
    derived_images = []
    for sample_id, candidate in expected.items():
        record = resolved[sample_id]
        expected_group = f'open-images-v7:{candidate["image_id"]}'
        if record.group_id != expected_group:
            raise ValueError(f"group_id mismatch for reviewed sample {sample_id}")
        if record.image and Path(record.image).resolve() != Path(str(candidate["image"])).resolve():
            raise ValueError(f"image path mismatch for reviewed sample {sample_id}")
        if record.decision != "clean":
            continue
        image_path = Path(str(candidate["image"])).resolve()
        if _sha256(image_path) != sample_id:
            raise ValueError(f"image changed after review: {image_path}")
        rotation = int(candidate["rotation_degrees_counterclockwise"])
        training_image_path = image_path
        tags = {
            "open_images_v7",
            "per_image_license_reviewed",
            "reviewed_clean",
            "double_reviewed",
            *record.manifest_tags,
        }
        with Image.open(image_path) as image:
            oriented = image.convert("RGB")
            if rotation:
                transpose = {
                    90: Image.Transpose.ROTATE_90,
                    180: Image.Transpose.ROTATE_180,
                    270: Image.Transpose.ROTATE_270,
                }[rotation]
                oriented = oriented.transpose(transpose)
                normalized_dir.mkdir(parents=True, exist_ok=True)
                training_image_path = normalized_dir / f"{sample_id}.png"
                if training_image_path.exists():
                    raise FileExistsError(f"refusing to overwrite {training_image_path}")
                oriented.save(training_image_path)
                tags.add("orientation_normalized")
                derived_images.append(
                    {
                        "original_image_sha256": sample_id,
                        "rotation_degrees_counterclockwise": rotation,
                        "derived_image": str(training_image_path),
                        "derived_image_sha256": _sha256(training_image_path),
                    }
                )
            empty = Image.new("L", oriented.size, 0)
        mask_path = mask_dir / f"{sample_id}.png"
        if mask_path.exists():
            raise FileExistsError(f"refusing to overwrite {mask_path}")
        empty.save(mask_path)
        rows.append(
            {
                "image": os.path.relpath(training_image_path, output_root).replace("\\", "/"),
                "mask": os.path.relpath(mask_path, output_root).replace("\\", "/"),
                "split": "train",
                "group_id": expected_group,
                "source": "open-images-v7-per-image-reviewed",
                "license_id": f"CC-BY-2.0-per-image-review-{review_digest[:16]}",
                "commercial_use": True,
                "damage_supervised": True,
                "tags": sorted(tags),
            }
        )
    if not rows:
        raise ValueError("no candidate was resolved as clean; refusing to write an empty manifest")
    with output_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    validation = validate_manifest(
        load_manifest(output_manifest), len(DEFAULT_CLASSES), check_files=True
    )
    if not validation["valid"]:
        raise ValueError(f"generated manifest failed validation: {validation['errors']}")
    report = {
        **license_report,
        "status": "commercially_reviewed_double_reviewed_clean_training_manifest",
        "training_eligible": True,
        "review_a": str(Path(args.review_a).resolve()),
        "review_a_sha256": _sha256(Path(args.review_a)),
        "reviewer_a": reviewer_a,
        "review_b": str(Path(args.review_b).resolve()),
        "review_b_sha256": _sha256(Path(args.review_b)),
        "reviewer_b": reviewer_b,
        "disagreements": len(disagreements),
        "adjudication": str(Path(args.adjudication).resolve()) if args.adjudication else None,
        "adjudication_sha256": _sha256(Path(args.adjudication)) if args.adjudication else None,
        "adjudicator": adjudicator,
        "clean_images_imported": len(rows),
        "orientation_normalized_images": derived_images,
        "damaged_images_excluded": sum(row.decision == "damaged" for row in resolved.values()),
        "unusable_images_excluded": sum(row.decision == "unusable" for row in resolved.values()),
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": _sha256(output_manifest),
        "manifest_validation": validation,
    }
    Path(str(output_manifest) + ".reviews.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
