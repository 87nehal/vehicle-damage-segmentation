from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path

from vehicle_damage.open_images_intake import load_license_approved_candidates
from vehicle_damage.review import ANGLES, CLEANLINESS, DISTANCES, LIGHTING, NUISANCE_TAGS


FIELDS = [
    "sample_id", "image", "group_id", "reviewer_id", "decision",
    "nuisance_tags", "distance", "angle", "lighting", "cleanliness", "notes",
]


def _select(field: str, choices: set[str]) -> str:
    options = '<option value="">select</option>' + "".join(
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>'
        for value in sorted(choices)
    )
    return f'<select data-field="{field}">{options}</select>'


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a double damage-review batch after per-image license approval"
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--license-review", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    approved, license_report = load_license_approved_candidates(
        args.candidates, args.license_review
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    cards = []
    for index, candidate in enumerate(approved):
        image_path = Path(str(candidate["image"]))
        sample_id = str(candidate["image_sha256"])
        group_id = f'open-images-v7:{candidate["image_id"]}'
        rows.append(
            {
                "sample_id": sample_id,
                "image": str(image_path),
                "group_id": group_id,
                "reviewer_id": "",
                "decision": "",
                "nuisance_tags": "",
                "distance": "",
                "angle": "",
                "lighting": "",
                "cleanliness": "",
                "notes": "",
            }
        )
        relative_image = Path(os.path.relpath(image_path, output)).as_posix()
        rotation = int(candidate["rotation_degrees_counterclockwise"])
        nuisance = "".join(
            f'<label><input type="checkbox" data-tag="{html.escape(tag)}">'
            f'{html.escape(tag)}</label>'
            for tag in sorted(NUISANCE_TAGS)
        )
        cards.append(
            f'<section class="card" data-index="{index}"><h2>{index + 1}. '
            f'{html.escape(group_id)}</h2><img loading="lazy" '
            f'src="{html.escape(relative_image)}" style="transform:rotate(-{rotation}deg)">'
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

    with (output / "review-template.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    safe_rows = json.dumps(rows).replace("</", "<\\/")
    page = f"""<!doctype html><meta charset="utf-8"><title>Open Images damage review</title>
<style>body{{font:14px system-ui;margin:20px;background:#f4f5f7}}header{{position:sticky;top:0;background:white;padding:12px;z-index:2}}.card{{background:white;margin:16px 0;padding:14px;border-radius:8px}}img{{max-width:700px;max-height:520px;display:block;margin:8px 0}}label{{margin:6px 12px 6px 0;display:inline-block}}select,input{{margin-left:5px}}</style>
<header><strong>Independent vehicle damage review</strong> Reviewer ID <input id="reviewer"> <button id="download">Download completed CSV</button><p>Visible exterior damage means damaged. Glare, dirt, panel gaps, reflections, shadows and styling creases are not damage. Complete every field without consulting another reviewer.</p></header>{''.join(cards)}
<script>const base={safe_rows};const key='open-images-damage-review-'+location.pathname;const saved=JSON.parse(localStorage.getItem(key)||'{{}}');document.querySelectorAll('.card').forEach(card=>{{const i=card.dataset.index,state=saved[i]||{{}};card.querySelectorAll('[data-field]').forEach(el=>{{el.value=state[el.dataset.field]||'';el.onchange=save;el.oninput=save}});card.querySelectorAll('[data-tag]').forEach(el=>{{el.checked=(state.nuisance_tags||[]).includes(el.dataset.tag);el.onchange=save}})}});document.querySelector('#reviewer').value=saved.reviewer_id||'';document.querySelector('#reviewer').oninput=save;function save(){{const state={{reviewer_id:document.querySelector('#reviewer').value}};document.querySelectorAll('.card').forEach(card=>{{const row={{}};card.querySelectorAll('[data-field]').forEach(el=>row[el.dataset.field]=el.value);row.nuisance_tags=[...card.querySelectorAll('[data-tag]:checked')].map(x=>x.dataset.tag);state[card.dataset.index]=row}});localStorage.setItem(key,JSON.stringify(state))}}function esc(v){{v=String(v??'');return /[\",\\n]/.test(v)?'\"'+v.replaceAll('\"','\"\"')+'\"':v}}document.querySelector('#download').onclick=()=>{{save();const state=JSON.parse(localStorage.getItem(key));if(!state.reviewer_id.trim()){{alert('Reviewer ID is required');return}}const fields={json.dumps(FIELDS)},out=[fields.join(',')];for(let i=0;i<base.length;i++){{const r=state[i]||{{}};for(const f of ['decision','distance','angle','lighting','cleanliness'])if(!(r[f]||'').trim()){{alert(`Complete ${{f}} for image ${{i+1}}`);return}}const row={{...base[i],...r,reviewer_id:state.reviewer_id,nuisance_tags:(r.nuisance_tags||[]).sort().join(';')}};out.push(fields.map(f=>esc(row[f])).join(','))}}const blob=new Blob([out.join('\\n')],{{type:'text/csv'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='review-'+state.reviewer_id+'.csv';a.click();URL.revokeObjectURL(a.href)}};</script>"""
    (output / "review.html").write_text(page, encoding="utf-8")
    provenance = {
        **license_report,
        "status": "license_approved_candidates_awaiting_double_damage_review",
        "training_eligible": False,
        "samples": len(rows),
        "sample_ids": [row["sample_id"] for row in rows],
    }
    (output / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps({"samples": len(rows), "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
