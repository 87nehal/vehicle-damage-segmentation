from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import random
from pathlib import Path

from vehicle_damage.manifest import load_manifest
from vehicle_damage.review import (
    ANGLES,
    CLEANLINESS,
    DISTANCES,
    LIGHTING,
    NUISANCE_TAGS,
)


FIELDS = [
    "sample_id", "image", "group_id", "reviewer_id", "decision",
    "nuisance_tags", "distance", "angle", "lighting", "cleanliness", "notes",
]


def _select(value: str, choices: set[str]) -> str:
    options = '<option value="">select</option>' + "".join(
        f'<option value="{html.escape(choice)}">{html.escape(choice)}</option>'
        for choice in sorted(choices)
    )
    return f'<select data-field="{value}">{options}</select>'


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tamper-evident vehicle review batch")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default="damage_unverified")
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--ranking",
        help="Optional hash-bound output from rank_review_candidates.py",
    )
    parser.add_argument(
        "--ranking-order",
        choices=("highest", "lowest"),
        default="highest",
        help="Review the strongest predictions first or mine likely false negatives",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")

    samples = [
        sample for sample in load_manifest(args.manifest, require_commercial=True)
        if args.tag in sample.tags and (args.split is None or sample.split == args.split)
    ]
    random.Random(args.seed).shuffle(samples)
    ranking_provenance = None
    if args.ranking:
        ranking_path = Path(args.ranking)
        ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
        if ranking.get("status") != "review_priority_only_not_ground_truth":
            raise ValueError("ranking is not a review-priority-only artifact")
        manifest_path = Path(args.manifest)
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if ranking.get("manifest_sha256") != manifest_sha256:
            raise ValueError("ranking source manifest hash does not match --manifest")
        if ranking.get("selection_tag") != args.tag:
            raise ValueError("ranking selection tag does not match --tag")
        ranked_rows = ranking.get("samples")
        if not isinstance(ranked_rows, list) or not ranked_rows:
            raise ValueError("ranking contains no sample records")
        rank_by_digest = {}
        for row in ranked_rows:
            digest = str(row.get("image_sha256", ""))
            rank = row.get("rank")
            if not digest or digest in rank_by_digest or not isinstance(rank, int):
                raise ValueError("ranking requires unique image hashes and integer ranks")
            rank_by_digest[digest] = rank
        sample_digests = {
            sample.image: hashlib.sha256(sample.image.read_bytes()).hexdigest()
            for sample in samples
        }
        matched = sum(digest in rank_by_digest for digest in sample_digests.values())
        if matched == 0:
            raise ValueError("ranking does not match any selected manifest image")
        def ranking_key(sample):
            digest = sample_digests[sample.image]
            rank = rank_by_digest.get(digest)
            if rank is None:
                return (1, 0, digest)
            ordered_rank = rank if args.ranking_order == "highest" else -rank
            return (0, ordered_rank, digest)

        samples.sort(key=ranking_key)
        ranking_provenance = {
            "path": str(ranking_path.resolve()),
            "sha256": hashlib.sha256(ranking_path.read_bytes()).hexdigest(),
            "status": ranking["status"],
            "ranking_method": ranking.get("ranking_method"),
            "order": args.ranking_order,
            "matched_samples": matched,
        }
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("no manifest samples match the requested review selection")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    cards = []
    for index, sample in enumerate(samples):
        digest = hashlib.sha256(sample.image.read_bytes()).hexdigest()
        row = {
            "sample_id": digest,
            "image": str(sample.image.resolve()),
            "group_id": sample.group_id,
            "reviewer_id": "",
            "decision": "",
            "nuisance_tags": "",
            "distance": "",
            "angle": "",
            "lighting": "",
            "cleanliness": "",
            "notes": "",
        }
        rows.append(row)
        relative_image = Path(os.path.relpath(sample.image.resolve(), output.resolve())).as_posix()
        nuisance = "".join(
            f'<label><input type="checkbox" data-tag="{html.escape(tag)}">{html.escape(tag)}</label>'
            for tag in sorted(NUISANCE_TAGS)
        )
        cards.append(
            f'<section class="card" data-index="{index}"><h2>{index + 1}. '
            f'{html.escape(sample.group_id)}</h2><img loading="lazy" src="{html.escape(relative_image)}">'
            '<label>Decision<select data-field="decision"><option value="">select</option>'
            '<option value="clean">clean</option><option value="damaged">damaged</option>'
            '<option value="unusable">unusable</option></select></label>'
            f'<label>Distance{_select("distance", DISTANCES)}</label>'
            f'<label>Angle{_select("angle", ANGLES)}</label>'
            f'<label>Lighting{_select("lighting", LIGHTING)}</label>'
            f'<label>Cleanliness{_select("cleanliness", CLEANLINESS)}</label>'
            f'<div class="tags">Nuisance tags: {nuisance}</div>'
            '<label>Notes<input data-field="notes"></label></section>'
        )

    csv_path = output / "review-template.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    safe_rows = json.dumps(rows).replace("</", "<\\/")
    page = f"""<!doctype html>
<meta charset="utf-8"><title>Vehicle clean/damage review</title>
<style>
body{{font:14px system-ui;margin:20px;background:#f4f5f7}}header{{position:sticky;top:0;background:white;padding:12px;z-index:2}}
.card{{background:white;margin:16px 0;padding:14px;border-radius:8px}}img{{max-width:700px;max-height:520px;display:block;margin:8px 0}}
label{{margin:6px 12px 6px 0;display:inline-block}}.tags label{{display:inline-block}}select,input{{margin-left:5px}}
</style><header><strong>Double-review batch</strong> Reviewer ID <input id="reviewer"> <button id="download">Download completed CSV</button> <button id="reset">Reset local form</button>
<p>Mark visible damage as damaged. Glare, dirt, gaps, reflections and shadows are not damage. Complete every field.</p></header>
{''.join(cards)}
<script>
const base={safe_rows}; const key='vehicle-review-'+location.pathname;
const saved=JSON.parse(localStorage.getItem(key)||'{{}}');
document.querySelectorAll('.card').forEach(card=>{{const i=card.dataset.index; const state=saved[i]||{{}};
 card.querySelectorAll('[data-field]').forEach(el=>{{el.value=state[el.dataset.field]||''; el.onchange=save; el.oninput=save}});
 card.querySelectorAll('[data-tag]').forEach(el=>{{el.checked=(state.nuisance_tags||[]).includes(el.dataset.tag); el.onchange=save}});
}}); document.querySelector('#reviewer').value=saved.reviewer_id||''; document.querySelector('#reviewer').oninput=save;
function save(){{const state={{reviewer_id:document.querySelector('#reviewer').value}}; document.querySelectorAll('.card').forEach(card=>{{const row={{}};
 card.querySelectorAll('[data-field]').forEach(el=>row[el.dataset.field]=el.value); row.nuisance_tags=[...card.querySelectorAll('[data-tag]:checked')].map(x=>x.dataset.tag); state[card.dataset.index]=row;}}); localStorage.setItem(key,JSON.stringify(state));}}
function esc(v){{v=String(v??''); return /[\",\\n]/.test(v)?'"'+v.replaceAll('"','""')+'"':v}}
document.querySelector('#reset').onclick=()=>{{if(confirm('Clear all locally saved answers for this batch?')){{localStorage.removeItem(key);location.reload()}}}};
document.querySelector('#download').onclick=()=>{{save(); const state=JSON.parse(localStorage.getItem(key)); if(!state.reviewer_id.trim()){{alert('Reviewer ID is required');return}}
 const fields={json.dumps(FIELDS)}; const out=[fields.join(',')]; for(let i=0;i<base.length;i++){{const r=state[i]||{{}};
 for(const f of ['decision','distance','angle','lighting','cleanliness'])if(!(r[f]||'').trim()){{alert(`Complete ${{f}} for image ${{i+1}}`);return}}
 const row={{...base[i],...r,reviewer_id:state.reviewer_id,nuisance_tags:(r.nuisance_tags||[]).sort().join(';')}}; out.push(fields.map(f=>esc(row[f])).join(','));}}
 const blob=new Blob([out.join('\\n')],{{type:'text/csv'}}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='review-'+state.reviewer_id+'.csv'; a.click(); URL.revokeObjectURL(a.href);}};
</script>"""
    (output / "review.html").write_text(page, encoding="utf-8")
    provenance = {
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "selection_tag": args.tag,
        "split": args.split,
        "seed": args.seed,
        "samples": len(rows),
        "sample_ids": [row["sample_id"] for row in rows],
        "ranking": ranking_provenance,
    }
    (output / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(rows), "html": str(output / 'review.html'), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
