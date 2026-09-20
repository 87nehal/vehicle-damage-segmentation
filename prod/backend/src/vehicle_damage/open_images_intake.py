from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from PIL import Image


CC_BY_2_URL = "https://creativecommons.org/licenses/by/2.0/"
ALLOWED_BOX_SOURCES = frozenset({"xclick"})
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
METADATA_COLUMNS = frozenset(
    {
        "ImageID", "Subset", "OriginalURL", "OriginalLandingURL", "License",
        "AuthorProfileURL", "Author", "Title", "OriginalSize", "OriginalMD5", "Rotation",
    }
)
BOX_COLUMNS = frozenset(
    {
        "ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax",
        "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
        "IsDepiction", "IsInside",
    }
)
CLASS_COLUMNS = frozenset({"LabelName", "DisplayName"})
LICENSE_REVIEW_FIELDS = (
    "candidate_id", "image_id", "image_sha256", "original_landing_url",
    "declared_license_url", "declared_author", "declared_author_profile_url",
    "reviewer_id", "source_page_checked_at_utc", "decision",
    "observed_license_url", "observed_author",
    "commercial_ml_training_rights_confirmed", "notes",
)


@dataclass(frozen=True)
class CarBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    source: str
    is_occluded: bool
    is_truncated: bool


@dataclass(frozen=True)
class OpenImagesCandidate:
    candidate_id: str
    image_id: str
    image: str
    image_sha256: str
    image_md5_hex: str
    official_original_md5: str
    original_size_bytes: int
    subset: str
    original_url: str
    original_landing_url: str
    declared_license_url: str
    declared_author_profile_url: str
    declared_author: str
    declared_title: str
    rotation_degrees_counterclockwise: int
    car_boxes: tuple[CarBox, ...]
    source: str = "open-images-v7"
    intake_status: str = "quarantined_pending_per_image_license_review"
    commercial_use_approved: bool = False
    training_eligible: bool = False
    damage_label_status: str = "unreviewed"

    def to_json(self) -> dict[str, object]:
        row = asdict(self)
        row["car_boxes"] = [asdict(box) for box in self.car_boxes]
        return row


def _require_columns(reader: csv.DictReader, required: frozenset[str], source: Path) -> None:
    missing = required - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")


def _flag_is(value: str, expected: int) -> bool:
    try:
        return float(value.strip()) == float(expected)
    except (TypeError, ValueError):
        return False


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _load_car_label_ids(path: Path) -> set[str]:
    labels: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader, CLASS_COLUMNS, path)
        for row in reader:
            if row["DisplayName"].strip().casefold() == "car":
                label = row["LabelName"].strip()
                if label:
                    labels.add(label)
    if not labels:
        raise ValueError(f"{path}: no exact Car class found")
    return labels


def _parse_box(row: dict[str, str]) -> CarBox | None:
    source = row["Source"].strip().casefold()
    if source not in ALLOWED_BOX_SOURCES:
        return None
    if not _flag_is(row["Confidence"], 1):
        return None
    if (
        not _flag_is(row["IsGroupOf"], 0)
        or not _flag_is(row["IsDepiction"], 0)
        or not _flag_is(row["IsInside"], 0)
    ):
        return None
    try:
        xmin, xmax = float(row["XMin"]), float(row["XMax"])
        ymin, ymax = float(row["YMin"]), float(row["YMax"])
    except (TypeError, ValueError):
        return None
    if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
        return None
    return CarBox(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        source=source,
        is_occluded=_flag_is(row.get("IsOccluded", "0"), 1),
        is_truncated=_flag_is(row.get("IsTruncated", "0"), 1),
    )


def _load_car_boxes(path: Path, car_labels: set[str]) -> dict[str, tuple[CarBox, ...]]:
    boxes: dict[str, list[CarBox]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader, BOX_COLUMNS, path)
        for row in reader:
            if row["LabelName"].strip() not in car_labels:
                continue
            image_id = row["ImageID"].strip()
            box = _parse_box(row)
            if image_id and box is not None:
                boxes[image_id].append(box)
    if not boxes:
        raise ValueError(
            f"{path}: no eligible Car boxes; intake only accepts xclick, confidence=1, "
            "non-group, non-depiction, exterior-view annotations"
        )
    return {image_id: tuple(rows) for image_id, rows in boxes.items()}


def _decode_official_md5(value: str) -> bytes | None:
    raw = value.strip()
    if len(raw) == 32:
        try:
            decoded = bytes.fromhex(raw)
        except ValueError:
            decoded = b""
        if len(decoded) == 16:
            return decoded
    try:
        decoded = base64.b64decode(raw, validate=True)
    except (ValueError, base64.binascii.Error):
        return None
    return decoded if len(decoded) == 16 else None


def _resolve_image(images_dir: Path, image_id: str) -> Path | None:
    matches = [
        candidate
        for suffix in IMAGE_SUFFIXES
        for candidate in (
            images_dir / f"{image_id}{suffix}",
            images_dir / f"{image_id}{suffix.upper()}",
        )
        if candidate.is_file()
    ]
    unique = list(dict.fromkeys(path.resolve() for path in matches))
    return unique[0] if len(unique) == 1 else None


def _valid_rotation(value: str) -> int | None:
    try:
        rotation = int(float(value.strip() or "0"))
    except ValueError:
        return None
    return rotation if rotation in {0, 90, 180, 270} else None


def _metadata_candidates(
    metadata_path: Path,
    images_dir: Path,
    car_boxes: dict[str, tuple[CarBox, ...]],
) -> tuple[list[OpenImagesCandidate], Counter[str]]:
    accepted: dict[str, OpenImagesCandidate] = {}
    rejected: Counter[str] = Counter()
    seen: set[str] = set()
    duplicates: set[str] = set()
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader, METADATA_COLUMNS, metadata_path)
        for row in reader:
            image_id = row["ImageID"].strip()
            if image_id not in car_boxes:
                continue
            if image_id in seen:
                rejected["duplicate_metadata_row"] += 1
                duplicates.add(image_id)
                accepted.pop(image_id, None)
                continue
            seen.add(image_id)
            if row["License"].strip() != CC_BY_2_URL:
                rejected["license_not_exact_cc_by_2"] += 1
                continue
            required_values = {
                "subset": row["Subset"], "original_url": row["OriginalURL"],
                "original_landing_url": row["OriginalLandingURL"],
                "author_profile_url": row["AuthorProfileURL"], "author": row["Author"],
                "title": row["Title"], "original_md5": row["OriginalMD5"],
                "original_size": row["OriginalSize"],
            }
            if any(not value.strip() for value in required_values.values()):
                rejected["missing_attribution_or_provenance"] += 1
                continue
            if not all(
                _is_http_url(row[field])
                for field in (
                    "OriginalURL", "OriginalLandingURL", "AuthorProfileURL", "License"
                )
            ):
                rejected["invalid_provenance_url"] += 1
                continue
            expected_md5 = _decode_official_md5(row["OriginalMD5"])
            if expected_md5 is None:
                rejected["invalid_official_md5"] += 1
                continue
            try:
                original_size = int(row["OriginalSize"])
            except ValueError:
                original_size = 0
            if original_size <= 0:
                rejected["invalid_original_size"] += 1
                continue
            rotation = _valid_rotation(row["Rotation"])
            if rotation is None:
                rejected["invalid_rotation"] += 1
                continue
            image_path = _resolve_image(images_dir, image_id)
            if image_path is None:
                rejected["missing_or_ambiguous_local_image"] += 1
                continue
            image_bytes = image_path.read_bytes()
            if len(image_bytes) != original_size:
                rejected["original_size_mismatch"] += 1
                continue
            actual_md5 = hashlib.md5(image_bytes, usedforsecurity=False).digest()
            if actual_md5 != expected_md5:
                rejected["original_md5_mismatch"] += 1
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, SyntaxError):
                rejected["unreadable_image"] += 1
                continue
            image_sha256 = hashlib.sha256(image_bytes).hexdigest()
            candidate_id = hashlib.sha256(
                f"open-images-v7:{image_id}:{image_sha256}".encode("utf-8")
            ).hexdigest()
            if image_id not in duplicates:
                accepted[image_id] = OpenImagesCandidate(
                    candidate_id=candidate_id, image_id=image_id, image=str(image_path),
                    image_sha256=image_sha256, image_md5_hex=actual_md5.hex(),
                    official_original_md5=row["OriginalMD5"].strip(),
                    original_size_bytes=original_size,
                    subset=row["Subset"].strip(), original_url=row["OriginalURL"].strip(),
                    original_landing_url=row["OriginalLandingURL"].strip(),
                    declared_license_url=row["License"].strip(),
                    declared_author_profile_url=row["AuthorProfileURL"].strip(),
                    declared_author=row["Author"].strip(),
                    declared_title=row["Title"].strip(),
                    rotation_degrees_counterclockwise=rotation, car_boxes=car_boxes[image_id],
                )
    rejected["eligible_box_images_missing_from_metadata"] = len(set(car_boxes) - seen)
    return list(accepted.values()), rejected


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_select(
    candidates: Iterable[OpenImagesCandidate], *, limit: int, seed: int
) -> list[OpenImagesCandidate]:
    ordered = sorted(
        candidates,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{row.image_id}".encode("utf-8")).hexdigest(),
            row.image_id,
        ),
    )
    return ordered[:limit] if limit else ordered


def _write_license_review(
    output_dir: Path,
    candidates: list[OpenImagesCandidate],
    *,
    inventory_sha256: str,
) -> None:
    rows = [
        {
            "candidate_id": row.candidate_id,
            "image_id": row.image_id,
            "image_sha256": row.image_sha256,
            "original_landing_url": row.original_landing_url,
            "declared_license_url": row.declared_license_url,
            "declared_author": row.declared_author,
            "declared_author_profile_url": row.declared_author_profile_url,
            "reviewer_id": "",
            "source_page_checked_at_utc": "",
            "decision": "",
            "observed_license_url": "",
            "observed_author": "",
            "commercial_ml_training_rights_confirmed": "",
            "notes": "",
        }
        for row in candidates
    ]
    with (output_dir / "LICENSE_REVIEW_TEMPLATE.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=LICENSE_REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    cards = []
    for index, row in enumerate(candidates):
        relative_image = Path(
            os.path.relpath(Path(row.image), output_dir.resolve())
        ).as_posix()
        cards.append(
            f'<section class="card" data-index="{index}"><h2>{index + 1}. '
            f'{html.escape(row.image_id)}</h2>'
            f'<img loading="lazy" src="{html.escape(relative_image)}" '
            f'style="transform:rotate(-{row.rotation_degrees_counterclockwise}deg)">'
            f'<p>Title: {html.escape(row.declared_title)}<br>Declared author: '
            f'{html.escape(row.declared_author)}<br>Declared license: '
            f'<a target="_blank" rel="noopener" href="{html.escape(row.declared_license_url)}">'
            f'CC BY 2.0</a><br><a target="_blank" rel="noopener" '
            f'href="{html.escape(row.original_landing_url)}">Open original landing page</a> | '
            f'<a target="_blank" rel="noopener" '
            f'href="{html.escape(row.declared_author_profile_url)}">Author profile</a></p>'
            '<button type="button" data-check>Mark source page checked now</button> '
            '<span data-checked>not checked</span><br>'
            '<label>Decision <select data-field="decision"><option value="">select</option>'
            '<option value="approve">approve</option><option value="reject">reject</option>'
            '</select></label><br>'
            '<label>Observed license URL <input size="55" data-field="observed_license_url"></label> '
            '<button type="button" data-copy-license>Copy declared license</button><br>'
            '<label>Observed author <input size="45" data-field="observed_author"></label> '
            '<button type="button" data-copy-author>Copy declared author</button><br>'
            '<label><input type="checkbox" data-rights> I confirm commercial ML training '
            'rights for this image</label><br>'
            '<label>Notes <input size="70" data-field="notes"></label></section>'
        )
    safe_rows = json.dumps(rows).replace("</", "<\\/")
    page = f"""<!doctype html><meta charset="utf-8"><title>Open Images license review</title>
<style>body{{font:14px system-ui;margin:20px;background:#f4f5f7}}header{{position:sticky;top:0;background:white;padding:14px;z-index:2}}section{{background:white;padding:14px;margin:14px 0;border-radius:8px}}img{{max-width:700px;max-height:520px;display:block;margin:10px}}label{{display:inline-block;margin:6px 4px}}input,select{{margin-left:4px}}[data-checked]{{font-weight:600}}</style>
<header><h1>Quarantined Open Images candidates</h1><p>These files are not approved for training and have no damage labels. A qualified rights reviewer must open each original landing page, verify the current image license and attribution, and make an explicit decision. Approval here still does not establish that a vehicle is damage-free; a separate double visual review is required. Draft CSVs are backups only and fail closed in the importer until every row is complete.</p><label>Rights reviewer ID <input id="reviewer"></label> <strong id="progress">0/{len(rows)} decided</strong> <button id="draft">Download draft CSV</button> <button id="backup">Download state backup</button> <label>Restore state <input id="restore" type="file" accept=".json,application/json"></label> <button id="download">Download completed CSV</button> <button id="reset">Reset local form</button></header>
{''.join(cards)}
<script>
const base={safe_rows}; const inventorySha256={json.dumps(inventory_sha256)}; const backupSchema='open-images-license-review-state-v1'; const key='open-images-license-review-'+location.pathname;
const saved=JSON.parse(localStorage.getItem(key)||'{{}}');
const reviewer=document.querySelector('#reviewer'); reviewer.value=saved.reviewer_id||'';
function cardState(card){{return saved[card.dataset.index]||(saved[card.dataset.index]={{}})}}
function renderProgress(){{const decided=document.querySelectorAll('.card select[data-field="decision"]').length-[...document.querySelectorAll('.card select[data-field="decision"]')].filter(x=>!x.value).length;document.querySelector('#progress').textContent=`${{decided}}/${{base.length}} decided`;}}
function save(){{saved.reviewer_id=reviewer.value;document.querySelectorAll('.card').forEach(card=>{{const state=cardState(card);card.querySelectorAll('[data-field]').forEach(el=>state[el.dataset.field]=el.value);state.commercial_ml_training_rights_confirmed=card.querySelector('[data-rights]').checked?'yes':'';}});localStorage.setItem(key,JSON.stringify(saved));renderProgress();}}
reviewer.oninput=save;
document.querySelectorAll('.card').forEach(card=>{{const i=Number(card.dataset.index),state=cardState(card),declared=base[i];card.querySelectorAll('[data-field]').forEach(el=>{{el.value=state[el.dataset.field]||'';el.onchange=save;el.oninput=save}});const rights=card.querySelector('[data-rights]');rights.checked=state.commercial_ml_training_rights_confirmed==='yes';rights.onchange=save;const checked=card.querySelector('[data-checked]');checked.textContent=state.source_page_checked_at_utc||'not checked';card.querySelector('[data-check]').onclick=()=>{{state.source_page_checked_at_utc=new Date().toISOString();checked.textContent=state.source_page_checked_at_utc;save()}};card.querySelector('[data-copy-license]').onclick=()=>{{card.querySelector('[data-field="observed_license_url"]').value=declared.declared_license_url;save()}};card.querySelector('[data-copy-author]').onclick=()=>{{card.querySelector('[data-field="observed_author"]').value=declared.declared_author;save()}};}});renderProgress();
function esc(v){{v=String(v??'');return /[\",\\n]/.test(v)?'\"'+v.replaceAll('\"','\"\"')+'\"':v}}
document.querySelector('#reset').onclick=()=>{{if(confirm('Clear all locally saved rights-review answers?')){{localStorage.removeItem(key);location.reload()}}}};
document.querySelector('#draft').onclick=()=>{{save();const reviewerId=reviewer.value.trim();if(!reviewerId){{alert('Rights reviewer ID is required');return}}const output=[{json.dumps(list(LICENSE_REVIEW_FIELDS))}.join(',')];for(let i=0;i<base.length;i++){{const state=cardState(document.querySelector(`.card[data-index="${{i}}"]`)),row={{...base[i],...state,reviewer_id:reviewerId}};output.push({json.dumps(list(LICENSE_REVIEW_FIELDS))}.map(field=>esc(row[field])).join(','));}}const blob=new Blob([output.join('\\n')],{{type:'text/csv'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='draft-license-review.csv';a.click();URL.revokeObjectURL(a.href)}};
document.querySelector('#backup').onclick=()=>{{save();const payload={{schema:backupSchema,inventory_sha256:inventorySha256,saved}},blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='license-review-state.json';a.click();URL.revokeObjectURL(a.href)}};
document.querySelector('#restore').onchange=async event=>{{const file=event.target.files[0];if(!file)return;try{{const payload=JSON.parse(await file.text());if(payload.schema!==backupSchema||payload.inventory_sha256!==inventorySha256||!payload.saved||typeof payload.saved!=='object'||Array.isArray(payload.saved))throw new Error('backup does not match this candidate inventory');const allowed=new Set(['reviewer_id',...base.map((_,i)=>String(i))]);if(Object.keys(payload.saved).some(name=>!allowed.has(name)))throw new Error('backup contains unknown rows');for(const name of allowed){{if(name==='reviewer_id'||payload.saved[name]===undefined)continue;if(!payload.saved[name]||typeof payload.saved[name]!=='object'||Array.isArray(payload.saved[name]))throw new Error('backup row is invalid')}}localStorage.setItem(key,JSON.stringify(payload.saved));location.reload()}}catch(error){{alert('Restore rejected: '+error.message);event.target.value=''}}}};
document.querySelector('#download').onclick=()=>{{save();const reviewerId=reviewer.value.trim();if(!reviewerId){{alert('Rights reviewer ID is required');return}}const output=[{json.dumps(list(LICENSE_REVIEW_FIELDS))}.join(',')];for(let i=0;i<base.length;i++){{const state=cardState(document.querySelector(`.card[data-index="${{i}}"]`)),decision=(state.decision||'').trim();if(!state.source_page_checked_at_utc){{alert(`Mark source page checked for image ${{i+1}}`);return}}if(!decision){{alert(`Select a decision for image ${{i+1}}`);return}}if(decision==='approve'){{if((state.observed_license_url||'').trim()!==base[i].declared_license_url){{alert(`Approved license must match declared CC BY 2.0 for image ${{i+1}}`);return}}if((state.observed_author||'').trim()!==base[i].declared_author){{alert(`Approved author must match attribution for image ${{i+1}}`);return}}if(state.commercial_ml_training_rights_confirmed!=='yes'){{alert(`Confirm commercial ML training rights for image ${{i+1}}`);return}}}}const row={{...base[i],...state,reviewer_id:reviewerId}};output.push({json.dumps(list(LICENSE_REVIEW_FIELDS))}.map(field=>esc(row[field])).join(','));}}const blob=new Blob([output.join('\\n')],{{type:'text/csv'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='completed-license-review.csv';a.click();URL.revokeObjectURL(a.href)}};
</script>"""
    (output_dir / "license-review.html").write_text(page, encoding="utf-8")


def prepare_open_images_candidates(
    *,
    metadata: str | Path,
    boxes: str | Path,
    class_descriptions: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    limit: int = 0,
    seed: int = 101,
) -> dict[str, object]:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    metadata_path = Path(metadata).resolve()
    boxes_path = Path(boxes).resolve()
    classes_path = Path(class_descriptions).resolve()
    local_images = Path(images_dir).resolve()
    for path in (metadata_path, boxes_path, classes_path):
        if not path.is_file():
            raise ValueError(f"missing input file: {path}")
    if not local_images.is_dir():
        raise ValueError(f"missing images directory: {local_images}")

    car_labels = _load_car_label_ids(classes_path)
    car_boxes = _load_car_boxes(boxes_path, car_labels)
    candidates, rejected = _metadata_candidates(metadata_path, local_images, car_boxes)
    selected = _stable_select(candidates, limit=limit, seed=seed)
    if not selected:
        raise ValueError("no candidates passed the fail-closed intake checks")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory = output / "CANDIDATES.jsonl"
    inventory.write_text(
        "".join(json.dumps(row.to_json(), sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    with (output / "ATTRIBUTION.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "image_id", "title", "author", "author_profile_url", "license_url",
            "original_landing_url", "original_url",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "image_id": row.image_id,
                    "title": row.declared_title,
                    "author": row.declared_author,
                    "author_profile_url": row.declared_author_profile_url,
                    "license_url": row.declared_license_url,
                    "original_landing_url": row.original_landing_url,
                    "original_url": row.original_url,
                }
            )
    inventory_sha256 = _sha256(inventory)
    _write_license_review(
        output,
        selected,
        inventory_sha256=inventory_sha256,
    )
    provenance: dict[str, object] = {
        "status": "quarantined_review_candidates_not_training_data",
        "training_eligible": False,
        "commercial_use_approved": False,
        "damage_labels_present": False,
        "policy": {
            "required_image_license": CC_BY_2_URL,
            "allowed_box_sources": sorted(ALLOWED_BOX_SOURCES),
            "requires_original_md5_match": True,
            "requires_per_image_landing_page_license_review": True,
            "requires_separate_double_damage_review": True,
        },
        "inputs": {
            "metadata": {"path": str(metadata_path), "sha256": _sha256(metadata_path)},
            "boxes": {"path": str(boxes_path), "sha256": _sha256(boxes_path)},
            "class_descriptions": {
                "path": str(classes_path), "sha256": _sha256(classes_path),
            },
            "images_dir": str(local_images),
        },
        "selection": {
            "seed": seed,
            "limit": limit,
            "eligible_before_limit": len(candidates),
            "selected_candidates": len(selected),
            "eligible_car_box_images": len(car_boxes),
            "rejections": dict(sorted(rejected.items())),
        },
        "inventory": {"path": str(inventory), "sha256": inventory_sha256},
    }
    (output / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return provenance


def _load_candidate_inventory(path: Path) -> dict[str, dict[str, object]]:
    provenance_path = path.parent / "PROVENANCE.json"
    if not provenance_path.is_file():
        raise ValueError(f"missing intake provenance: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    recorded_inventory = provenance.get("inventory", {})
    if (
        provenance.get("status") != "quarantined_review_candidates_not_training_data"
        or provenance.get("training_eligible") is not False
        or provenance.get("commercial_use_approved") is not False
        or not isinstance(recorded_inventory, dict)
        or recorded_inventory.get("sha256") != _sha256(path)
    ):
        raise ValueError("candidate inventory does not match its fail-closed provenance")
    required = {
        "candidate_id", "image_id", "image", "image_sha256", "original_landing_url",
        "declared_license_url", "declared_author_profile_url", "declared_author",
        "intake_status", "commercial_use_approved", "training_eligible",
        "damage_label_status", "car_boxes", "official_original_md5", "image_md5_hex",
        "original_url", "declared_title", "original_size_bytes",
        "rotation_degrees_counterclockwise",
    }
    rows: dict[str, dict[str, object]] = {}
    image_hashes: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
        candidate_id = str(row["candidate_id"])
        image_id = str(row["image_id"])
        image_sha256 = str(row["image_sha256"])
        if candidate_id in rows:
            raise ValueError(f"{path}:{line_number}: duplicate candidate_id")
        if image_sha256 in image_hashes:
            raise ValueError(f"{path}:{line_number}: duplicate image content")
        if (
            row["intake_status"] != "quarantined_pending_per_image_license_review"
            or row["commercial_use_approved"] is not False
            or row["training_eligible"] is not False
            or row["damage_label_status"] != "unreviewed"
        ):
            raise ValueError(f"{path}:{line_number}: candidate is not in the required quarantine state")
        if row["declared_license_url"] != CC_BY_2_URL:
            raise ValueError(f"{path}:{line_number}: declared license is not exact CC BY 2.0")
        if not all(
            _is_http_url(str(row[field]))
            for field in (
                "original_landing_url", "declared_license_url", "declared_author_profile_url"
            )
        ):
            raise ValueError(f"{path}:{line_number}: invalid provenance URL")
        image_path = Path(str(row["image"])).resolve()
        if not image_path.is_file():
            raise ValueError(f"{path}:{line_number}: missing image {image_path}")
        image_bytes = image_path.read_bytes()
        if len(image_bytes) != int(row["original_size_bytes"]):
            raise ValueError(f"{path}:{line_number}: original size evidence mismatch")
        actual_sha256 = hashlib.sha256(image_bytes).hexdigest()
        if actual_sha256 != image_sha256:
            raise ValueError(f"{path}:{line_number}: image SHA-256 mismatch")
        actual_md5 = hashlib.md5(image_bytes, usedforsecurity=False).digest()
        if (
            _decode_official_md5(str(row["official_original_md5"])) != actual_md5
            or str(row["image_md5_hex"]) != actual_md5.hex()
        ):
            raise ValueError(f"{path}:{line_number}: original MD5 evidence mismatch")
        expected_candidate_id = hashlib.sha256(
            f"open-images-v7:{image_id}:{image_sha256}".encode("utf-8")
        ).hexdigest()
        if candidate_id != expected_candidate_id:
            raise ValueError(f"{path}:{line_number}: candidate_id mismatch")
        if row["rotation_degrees_counterclockwise"] not in {0, 90, 180, 270}:
            raise ValueError(f"{path}:{line_number}: invalid rotation metadata")
        boxes = row["car_boxes"]
        if not isinstance(boxes, list) or not boxes:
            raise ValueError(f"{path}:{line_number}: eligible Car box is required")
        if any(not isinstance(box, dict) or box.get("source") not in ALLOWED_BOX_SOURCES for box in boxes):
            raise ValueError(f"{path}:{line_number}: invalid Car box provenance")
        row["image"] = str(image_path)
        rows[candidate_id] = row
        image_hashes.add(image_sha256)
    if not rows:
        raise ValueError(f"candidate inventory is empty: {path}")
    return rows


def _parse_utc_timestamp(value: str, *, line_number: int) -> str:
    raw = value.strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"license review line {line_number}: invalid source_page_checked_at_utc"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"license review line {line_number}: timestamp must be UTC")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_license_approved_candidates(
    inventory: str | Path, license_review: str | Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    inventory_path = Path(inventory).resolve()
    review_path = Path(license_review).resolve()
    candidates = _load_candidate_inventory(inventory_path)
    if not review_path.is_file():
        raise ValueError(f"missing license review: {review_path}")
    decisions: dict[str, dict[str, str]] = {}
    reviewers: set[str] = set()
    with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader, frozenset(LICENSE_REVIEW_FIELDS), review_path)
        for line_number, review in enumerate(reader, 2):
            candidate_id = review["candidate_id"].strip()
            candidate = candidates.get(candidate_id)
            if candidate is None:
                raise ValueError(f"license review line {line_number}: unknown candidate_id")
            if candidate_id in decisions:
                raise ValueError(f"license review line {line_number}: duplicate candidate_id")
            reviewer = review["reviewer_id"].strip()
            if not reviewer:
                raise ValueError(f"license review line {line_number}: reviewer_id is required")
            reviewers.add(reviewer)
            static_pairs = {
                "image_id": "image_id",
                "image_sha256": "image_sha256",
                "original_landing_url": "original_landing_url",
                "declared_license_url": "declared_license_url",
                "declared_author": "declared_author",
                "declared_author_profile_url": "declared_author_profile_url",
            }
            for review_field, candidate_field in static_pairs.items():
                if review[review_field].strip() != str(candidate[candidate_field]):
                    raise ValueError(
                        f"license review line {line_number}: {review_field} does not match inventory"
                    )
            checked_at = _parse_utc_timestamp(
                review["source_page_checked_at_utc"], line_number=line_number
            )
            decision = review["decision"].strip().casefold()
            if decision not in {"approve", "reject"}:
                raise ValueError(
                    f"license review line {line_number}: decision must be approve or reject"
                )
            if decision == "approve":
                if review["observed_license_url"].strip() != CC_BY_2_URL:
                    raise ValueError(
                        f"license review line {line_number}: approved license must be exact CC BY 2.0"
                    )
                if review["observed_author"].strip() != str(candidate["declared_author"]):
                    raise ValueError(
                        f"license review line {line_number}: observed author must match attribution"
                    )
                if review["commercial_ml_training_rights_confirmed"].strip().casefold() != "yes":
                    raise ValueError(
                        f"license review line {line_number}: commercial ML training rights must be confirmed"
                    )
            decisions[candidate_id] = {
                **review,
                "reviewer_id": reviewer,
                "decision": decision,
                "source_page_checked_at_utc": checked_at,
            }
    missing = sorted(candidates.keys() - decisions.keys())
    if missing:
        raise ValueError(f"license review is missing candidates: {missing}")
    if len(reviewers) != 1:
        raise ValueError("license review must contain exactly one reviewer_id")
    approved = [candidates[key] for key, value in decisions.items() if value["decision"] == "approve"]
    if not approved:
        raise ValueError("license review approved no candidates")
    report: dict[str, object] = {
        "status": "per_image_license_review_completed_damage_unreviewed",
        "training_eligible": False,
        "inventory": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
        "intake_provenance": str(inventory_path.parent / "PROVENANCE.json"),
        "intake_provenance_sha256": _sha256(inventory_path.parent / "PROVENANCE.json"),
        "license_review": str(review_path),
        "license_review_sha256": _sha256(review_path),
        "license_reviewer": next(iter(reviewers)),
        "reviewed_candidates": len(decisions),
        "approved_candidates": len(approved),
        "rejected_candidates": len(decisions) - len(approved),
    }
    return approved, report
