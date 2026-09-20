import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from vehicle_damage.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[1]


def _assert_embedded_javascript_parses(path):
    node = shutil.which("node")
    if node is None:
        return
    html = path.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.DOTALL)
    assert scripts
    subprocess.run(
        [node, "--check", "-"],
        input="\n".join(scripts),
        text=True,
        check=True,
        capture_output=True,
    )


def test_review_export_and_two_reviewer_clean_import(tmp_path):
    image = tmp_path / "car.jpg"
    mask = tmp_path / "unknown.png"
    exterior = tmp_path / "exterior.png"
    Image.new("RGB", (32, 24), "gray").save(image)
    Image.new("L", (32, 24), 0).save(mask)
    Image.new("L", (32, 24), 255).save(exterior)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image": image.name,
                "mask": mask.name,
                "exterior_mask": exterior.name,
                "split": "train",
                "group_id": "vehicle-1",
                "source": "consented-first-party",
                "license_id": "internal-release-1",
                "commercial_use": True,
                "damage_supervised": False,
                "tags": ["damage_unverified", "exterior_only"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    batch = tmp_path / "batch"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_review_batch.py"),
            "--manifest", str(manifest),
            "--output-dir", str(batch),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (batch / "review.html").is_file()
    _assert_embedded_javascript_parses(batch / "review.html")

    with (batch / "review-template.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    reviews = []
    for reviewer in ("assessor-a", "assessor-b"):
        path = tmp_path / f"{reviewer}.csv"
        row = dict(rows[0])
        row.update(
            reviewer_id=reviewer,
            decision="clean",
            nuisance_tags="glare;panel_gap",
            distance="full_car",
            angle="oblique",
            lighting="daylight",
            cleanliness="clean",
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
        reviews.append(path)

    output_manifest = tmp_path / "reviewed" / "manifest.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "apply_clean_reviews.py"),
            "--manifest", str(manifest),
            "--review-a", str(reviews[0]),
            "--review-b", str(reviews[1]),
            "--output-manifest", str(output_manifest),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    reviewed = load_manifest(output_manifest)
    assert reviewed[0].damage_supervised
    assert {"reviewed_clean", "double_reviewed", "glare", "panel_gap"} <= set(
        reviewed[0].tags
    )
    assert not np.asarray(Image.open(reviewed[0].mask)).any()
    report = json.loads(Path(str(output_manifest) + ".reviews.json").read_text())
    assert report["promoted_clean_images"] == 1
    assert report["disagreements_requiring_adjudication"] == 0

    disputed = dict(rows[0])
    disputed.update(
        reviewer_id="assessor-b",
        decision="damaged",
        nuisance_tags="glare",
        distance="full_car",
        angle="oblique",
        lighting="daylight",
        cleanliness="clean",
    )
    with reviews[1].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(disputed)
    adjudication = tmp_path / "automotive-assessor.csv"
    adjudicated = dict(rows[0])
    adjudicated.update(
        reviewer_id="automotive-assessor",
        decision="clean",
        nuisance_tags="reflection;panel_gap",
        distance="full_car",
        angle="oblique",
        lighting="daylight",
        cleanliness="clean",
    )
    with adjudication.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(adjudicated)

    adjudicated_manifest = tmp_path / "adjudicated" / "manifest.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "apply_clean_reviews.py"),
            "--manifest", str(manifest),
            "--review-a", str(reviews[0]),
            "--review-b", str(reviews[1]),
            "--adjudication", str(adjudication),
            "--output-manifest", str(adjudicated_manifest),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    adjudicated_rows = load_manifest(adjudicated_manifest)
    assert adjudicated_rows[0].damage_supervised
    assert {"reflection", "panel_gap"} <= set(adjudicated_rows[0].tags)
    adjudication_report = json.loads(
        Path(str(adjudicated_manifest) + ".reviews.json").read_text()
    )
    assert adjudication_report["disagreements_requiring_adjudication"] == 1
    assert adjudication_report["adjudicated_samples"] == 1
    assert adjudication_report["adjudicator"] == "automotive-assessor"


def test_review_export_respects_hash_bound_priority_ranking(tmp_path):
    manifest_rows = []
    digests = {}
    for name in ("lower", "higher"):
        image = tmp_path / f"{name}.jpg"
        mask = tmp_path / f"{name}.png"
        exterior = tmp_path / f"{name}-exterior.png"
        Image.new("RGB", (24, 16), name == "higher" and "white" or "gray").save(image)
        Image.new("L", (24, 16), 0).save(mask)
        Image.new("L", (24, 16), 255).save(exterior)
        digests[name] = hashlib.sha256(image.read_bytes()).hexdigest()
        manifest_rows.append(
            {
                "image": image.name,
                "mask": mask.name,
                "exterior_mask": exterior.name,
                "split": "train",
                "group_id": name,
                "source": "cc0-test",
                "license_id": "CC0-1.0",
                "commercial_use": True,
                "damage_supervised": False,
                "tags": ["damage_unverified", "exterior_only"],
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8"
    )
    ranking = tmp_path / "ranking.json"
    ranking.write_text(
        json.dumps(
            {
                "status": "review_priority_only_not_ground_truth",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "selection_tag": "damage_unverified",
                "ranking_method": "test priority",
                "samples": [
                    {"image_sha256": digests["higher"], "rank": 1},
                    {"image_sha256": digests["lower"], "rank": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    batch = tmp_path / "ranked-batch"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_review_batch.py"),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(batch),
            "--ranking",
            str(ranking),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    with (batch / "review-template.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["group_id"] for row in rows] == ["higher", "lower"]
    provenance = json.loads((batch / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert provenance["ranking"]["status"] == "review_priority_only_not_ground_truth"
    assert provenance["ranking"]["order"] == "highest"
    assert provenance["ranking"]["matched_samples"] == 2

    lowest_batch = tmp_path / "lowest-ranked-batch"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_review_batch.py"),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(lowest_batch),
            "--ranking",
            str(ranking),
            "--ranking-order",
            "lowest",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    with (lowest_batch / "review-template.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        lowest_rows = list(csv.DictReader(handle))
    assert [row["group_id"] for row in lowest_rows] == ["lower", "higher"]
