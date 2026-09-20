from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image

from vehicle_damage.review import apply_adjudication, exact_agreements, load_review_csv


PATH_FIELDS = ("image", "mask", "exterior_mask", "hard_negative_mask")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Promote clean decisions resolved by two exact reviews or optional "
            "third-assessor adjudication to explicit empty-mask supervision"
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--review-a", required=True)
    parser.add_argument("--review-b", required=True)
    parser.add_argument(
        "--adjudication",
        help="optional CSV from a third automotive assessor resolving every disagreement",
    )
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()

    output_manifest = Path(args.output_manifest)
    if output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite {output_manifest}")
    reviewer_a, records_a = load_review_csv(args.review_a)
    reviewer_b, records_b = load_review_csv(args.review_b)
    agreed, disagreements = exact_agreements(reviewer_a, records_a, reviewer_b, records_b)
    adjudicator = None
    adjudication_records = None
    resolved = agreed
    if args.adjudication:
        adjudicator, adjudication_records = load_review_csv(args.adjudication)
        resolved = apply_adjudication(
            agreed,
            disagreements,
            reviewer_a,
            reviewer_b,
            adjudicator,
            adjudication_records,
        )

    source_manifest = Path(args.manifest)
    source_root = source_manifest.parent.resolve()
    output_root = output_manifest.parent.resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    mask_dir = output_manifest.parent / "reviewed_clean_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    found: set[str] = set()
    promoted = 0
    for row in rows:
        image_path = Path(row["image"])
        image_path = image_path if image_path.is_absolute() else source_root / image_path
        sample_id = _sha256(image_path)
        record = resolved.get(sample_id)
        if record is not None:
            found.add(sample_id)
            if record.group_id != str(row["group_id"]):
                raise ValueError(f"group_id mismatch for reviewed sample {sample_id}")
            if record.decision == "clean":
                with Image.open(image_path) as image:
                    empty = Image.new("L", image.size, 0)
                mask_path = mask_dir / f"{sample_id}.png"
                empty.save(mask_path)
                row["mask"] = os.path.relpath(mask_path, output_root).replace("\\", "/")
                row["damage_supervised"] = True
                tags = set(str(tag) for tag in row.get("tags", []))
                tags.difference_update({"damage_unverified", "exterior_only"})
                tags.update({"reviewed_clean", "double_reviewed", *record.manifest_tags})
                row["tags"] = sorted(tags)
                promoted += 1
        for field in PATH_FIELDS:
            value = row.get(field)
            if value is None or (field == "mask" and record is not None and record.decision == "clean"):
                continue
            path = Path(value)
            resolved_path = path if path.is_absolute() else source_root / path
            row[field] = os.path.relpath(resolved_path, output_root).replace("\\", "/")

    missing = sorted(set(resolved) - found)
    if missing:
        raise ValueError(f"reviewed images are absent from manifest: {missing}")
    with output_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": _sha256(source_manifest),
        "review_a": str(Path(args.review_a).resolve()),
        "review_a_sha256": _sha256(Path(args.review_a)),
        "reviewer_a": reviewer_a,
        "review_b": str(Path(args.review_b).resolve()),
        "review_b_sha256": _sha256(Path(args.review_b)),
        "reviewer_b": reviewer_b,
        "exact_agreements": len(agreed),
        "disagreements_requiring_adjudication": len(disagreements),
        "disagreement_sample_ids": disagreements,
        "adjudication": str(Path(args.adjudication).resolve()) if args.adjudication else None,
        "adjudication_sha256": _sha256(Path(args.adjudication)) if args.adjudication else None,
        "adjudicator": adjudicator,
        "adjudicated_samples": len(disagreements) if args.adjudication else 0,
        "promoted_clean_images": promoted,
        "resolved_damaged_not_imported": sum(r.decision == "damaged" for r in resolved.values()),
        "resolved_unusable_not_imported": sum(r.decision == "unusable" for r in resolved.values()),
        "output_manifest": str(output_manifest.resolve()),
        "output_manifest_sha256": _sha256(output_manifest),
    }
    Path(str(output_manifest) + ".reviews.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
