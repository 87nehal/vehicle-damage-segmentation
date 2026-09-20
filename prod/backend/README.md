# Vehicle damage segmentation

This repository is a commercially cautious, recall-oriented baseline for visible exterior vehicle damage. It implements training, manifest/license validation, robust nuisance augmentation, tiled inference, calibration, evaluation, and review flags. A local CC0 pilot checkpoint is included in the workspace, but no production-reliability claim is made without representative first-party data.

A CC0 bootstrap checkpoint has now been trained locally; see
[PILOT_RESULTS.md](PILOT_RESULTS.md). It detected 81/81 cases on its
damage-only test split (95% Wilson lower bound 95.47%), but pixel precision is
only 14.08%, clean false alerts are unmeasurable, and the fail-closed audit
passes only 3/17 gates. It is not production-approved.

The leading untested development candidate is a shared-backbone role-split
DINOv2-small model. It preserves the previous selected final-layer heads for
native+1.5x 0.45 triage and uses four-last-layer heads for a native 0.55 mask
with a 144-pixel component guard. On consumed development validation it detects
98/98 cases; the mask reaches 50.22% pixel precision, 73.34% pixel recall,
6.13% background FPR, and 35.38%/33.41% connected-region precision/recall. It
also detects 98/98 cases in every one of 11 deterministic synthetic stress
conditions without losing any per-type hit or matched region relative to the
previous selection. Branch-selective execution skips unused segmentation heads
at the 1.5x triage-only scale without changing any output and runs at 335.0 ms
median on an RTX 3060
Laptop GPU. The previous 97.2 ms native-scale profile remains a faster,
lower-robustness alternative that requires its previous checkpoint. There are still no verified clean evaluation groups,
and only 4/17 release gates pass, so this is not a production result.
The hash-pinned machine-readable selection is
`runs/SELECTED_DEVELOPMENT_MODEL.json`.

The current wired development candidate additionally consumes 535 COCO 2017
full-scene car negatives, 141 augmented copies of the first intact-car
regression image, and 162 augmented handle/fuel-cap and side-view negatives.
Its frozen profile uses a 1,200-pixel connected-component guard. On the
held-out mixed validation it detects 91/98 damage cases (92.86%) and alerts on
1/100 clean COCO cases (1.0%). All three supplied clean regression images
produce no damage mask or detections; low-resolution examples are routed to
recapture. This is a measured improvement for the reported grille/body
contour/handle failure, but the recall trade-off means it remains
development-only and is not a production accuracy claim.

The standalone four-layer heads improved binary localization but lost two
crack-type triage hits. The promoted role split restores the exact parent
triage path after tile/TTA fusion while keeping the improved mask. Severe low
resolution still adds 10 false regions, and the two-scale path remains slower
than the 97.2 ms native-only fallback; exact evidence is documented in
[PILOT_RESULTS.md](PILOT_RESULTS.md).

## Why this design

- Semantic masks fit irregular dents, scratches, cracks, and breakage better than image classification or boxes.
- Separate presence and type heads protect high-recall localization from fine-grained class imbalance.
- Opt-in single-layer, four-layer, split-fusion, and dual-type DINOv2 decoders
  share the same commercially reviewed standard Apache-2.0 backbone; no other
  DINOv2 weight variants are admitted by the model factory.
- An auxiliary exterior head suppresses predictions off the inspected vehicle.
- Hard-negative masks explicitly teach glare, dirt, panel gaps, styling creases, reflections, and shadows as non-damage.
- Random scale/crop and overlapping full-resolution tiles cover close-ups and full-car photos.
- Versioned augmentation profiles preserve exact experiment replay.
  `robust_v2` adds label-aligned viewpoint warps, cast shadows, and sensor
  noise; `robust_v3` also mixes full-scene views and detail loss so tiny damage
  is not always enlarged by a positive crop.
- Asymmetric Tversky loss and held-out threshold calibration prioritize recall without hiding the resulting false-positive cost.
- Poor-quality or uncertain photos are routed to review/recapture; they are not forced into an unsafe answer.

Screen a newly captured directory before annotation. This creates hash-bound
JSON and CSV inventories, flags duplicate content, routes localized glare to
manual review, and routes unreadable, low-resolution, badly exposed, or blurred
captures to recapture. It does not grant licensing or training approval:

```powershell
python scripts/audit_capture_quality.py --images-dir path/to/new-captures --output audits/capture-quality.json
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Run the development model in the frontend

The inspection page is wired to the hash-pinned checkpoint and frozen profile
declared in `runs/SELECTED_DEVELOPMENT_MODEL.json`. The API loads the model once,
rejects artifact hash mismatches, and does not expose threshold overrides. The
selected FP16 profile currently requires a CUDA GPU.

Install the serving dependencies once, then start the integrated development
launcher:

```powershell
pip install -e ".[serve]"
cd frontend
npm install
npm run dev
```

The launcher starts (or reuses) the hash-verified model API on port 8001,
waits for the selected checkpoint to load, and then starts Next.js. Use
`npm run dev:web` only when running `vehicle-damage-api` separately.

Open `http://localhost:3000`, then upload a JPEG, PNG, or WebP image. The UI
shows the precise segmentation overlay, recall-only review regions, image
quality warnings, and the fail-closed automated/manual-review/recapture
decision. This remains a development-only interface, not production approval.

Prepare masks using [the annotation protocol](docs/ANNOTATION.md) and the
[first-party data plan](docs/FIRST_PARTY_DATA_PLAN.md), copy and edit
`data/manifest.example.jsonl`, then validate provenance and split isolation:

```powershell
vehicle-damage validate-manifest --manifest data/manifest.jsonl
vehicle-damage train --manifest data/manifest.jsonl --config configs/base.yaml --output-dir runs/baseline
```

For a legally permissive bootstrap set, the preparation script selectively fetches the polygon-annotated damage subset from the 3.1 GB Humans in the Loop archive and records its CC0 provenance. `--limit 0` fetches the complete damage subset; use a small limit only to smoke-test the pipeline:

```powershell
python scripts/prepare_hitl_cc0.py --output data/hitl_cc0 --limit 0
vehicle-damage validate-manifest --manifest data/hitl_cc0/manifest.jsonl
```

This public set is damage-enriched and is not a production test set. Do not reinterpret its separate part-annotation images as verified clean negatives.

### Double-review clean-data intake

Unverified or newly captured images can be exported to a self-contained browser
review page. Each row is bound to the image SHA-256. Two different reviewers
must agree exactly on the decision, nuisance tags, distance, angle, lighting,
and cleanliness before a clean image receives an explicit empty damage mask:

```powershell
python scripts/create_review_batch.py --manifest data/first_party/manifest.jsonl --output-dir review_batches/first_party_001 --tag damage_unverified --limit 100
# Each reviewer opens review.html and downloads their completed CSV.
python scripts/apply_clean_reviews.py --manifest data/first_party/manifest.jsonl --review-a review-a.csv --review-b review-b.csv --output-manifest data/first_party_reviewed/manifest.jsonl
vehicle-damage validate-manifest --manifest data/first_party_reviewed/manifest.jsonl
```

Same-reviewer submissions, altered image hashes, incomplete metadata, unknown
tags, and unresolved reviewer disagreements fail closed. A third automotive
assessor can resolve all disputed samples by adding
`--adjudication assessor.csv`; their reviewer ID must differ from both initial
reviewers. Resolved damaged images are not treated as clean; they still require
damage masks. A ready development batch is available at
`review_batches/cc0_unverified_001/review.html`, but those public auxiliary
images do not replace deployment-representative first-party testing.

Open Images V7 can only enter a pre-training quarantine. Given official image
metadata, box annotations, class descriptions, and already-downloaded originals,
the following command verifies exact CC BY 2.0 metadata, attribution fields,
original MD5 values, and human-drawn non-depiction Car boxes:

```powershell
python scripts/download_open_images_originals.py --metadata path/to/image-metadata.csv --boxes path/to/boxes.csv --class-descriptions path/to/class-descriptions-boxable.csv --output-dir data/open_images_originals --limit 200 --max-attempts 800
python scripts/prepare_open_images_candidates.py --metadata path/to/image-metadata.csv --boxes path/to/boxes.csv --class-descriptions path/to/class-descriptions-boxable.csv --images-dir path/to/images --output-dir review_batches/open_images_license_001 --limit 200
```

The output is intentionally not a training manifest. Every candidate is marked
not commercially approved, not training eligible, and damage-unreviewed. Use
the generated license review page and attribution CSV for per-image source
verification. Once a qualified rights reviewer completes its CSV, create the
independent visual-review batch and import only exact clean agreements:

```powershell
python scripts/create_open_images_damage_review_batch.py --candidates review_batches/open_images_license_001/CANDIDATES.jsonl --license-review completed-license-review.csv --output-dir review_batches/open_images_damage_001
python scripts/import_open_images_clean_reviews.py --candidates review_batches/open_images_license_001/CANDIDATES.jsonl --license-review completed-license-review.csv --review-a vehicle-review-a.csv --review-b vehicle-review-b.csv --output-manifest data/open_images_reviewed_clean/manifest.jsonl
```

Disagreements require a distinct third assessor with `--adjudication`; damaged
and unusable rows never become negative masks. See `docs/DATA_AND_LICENSES.md`
for the complete policy and official source links.

A 109-image original-file batch is already staged at
`review_batches/open_images_license_validation_001/license-review.html`. It is
still quarantined and must not be used for training until the generated rights
review and independent visual-review stages are completed.

The selected model can rank damage-unverified, commercially permitted images
to make review more efficient without creating pseudo-labels:

```powershell
python scripts/rank_review_candidates.py --checkpoint runs/pilot_cc0_dinov2_4layer_role_split/best.pt --calibration runs/pilot_cc0_dinov2_4layer_role_split/development-profile-fp16-triage0450-scales1-1p5-mask0550-scale1-comp144.json --manifest data/hitl_parts_cc0/manifest.jsonl --output runs/pilot_cc0_dinov2_4layer_role_split/review-priority-hitl-parts-all.json --device cuda
python scripts/create_review_batch.py --manifest data/hitl_parts_cc0/manifest.jsonl --output-dir review_batches/cc0_model_priority_001 --ranking runs/pilot_cc0_dinov2_4layer_role_split/review-priority-hitl-parts-all.json --limit 100
python scripts/create_review_batch.py --manifest data/hitl_parts_cc0/manifest.jsonl --output-dir review_batches/cc0_low_response_001 --ranking runs/pilot_cc0_dinov2_4layer_role_split/review-priority-hitl-parts-all.json --ranking-order lowest --limit 100
```

The high-response queue targets possible false positives if reviewers confirm
an image is clean. The low-response queue targets false negatives if reviewers
confirm damage. Rankings are bound to image, manifest, checkpoint, and profile
hashes and are explicitly marked as review priority only. Two independent
reviewers and adjudication are still required; damaged decisions need dense
mask annotation before training.

For dense correction of likely false negatives, export the lowest-response
deduplicated training images to CVAT's Segmentation Mask 1.1 format:

```powershell
python scripts/export_cvat_annotation_batch.py --checkpoint runs/pilot_cc0_dinov2_4layer_role_split/best.pt --calibration runs/pilot_cc0_dinov2_4layer_role_split/development-profile-fp16-triage0450-scales1-1p5-mask0550-scale1-comp144.json --manifest data/hitl_combined/manifest.jsonl --ranking runs/pilot_cc0_dinov2_4layer_role_split/review-priority-hitl-combined-unique-auxiliary.json --output-dir annotation_batches/cc0_unique_low_response_001 --device cuda --ranking-order lowest --limit 25
```

Create a CVAT task from `images.zip`, then upload
`preannotations-segmentation-mask-1.1.zip` using the **Segmentation Mask
1.1** format. An annotator must correct every mask and a different reviewer
must inspect it. Export the corrected task in the same format and import it:

```powershell
python scripts/import_cvat_annotation_batch.py --batch-manifest annotation_batches/cc0_unique_low_response_001/BATCH_MANIFEST.json --corrected-annotations corrected-segmentation-mask-1.1.zip --output-manifest data/hitl_dense_round1/manifest.jsonl --annotator-id annotator-a --reviewer-id reviewer-b
```

The importer verifies source and mask hashes, dimensions, class IDs, positive
damage pixels, and commercial provenance; it replaces only matching
damage-unsupervised training rows and validates the complete resulting
manifest. Empty corrected masks are rejected because clean vehicles require
the separate double-review workflow.

The separate 998-image car-part subset can safely supervise only the exterior
head. The preparation script records `damage_supervised: false`, so these
images never enter the damage presence/type loss and are never counted as
clean negatives:

```powershell
python scripts/prepare_hitl_parts_cc0.py --output data/hitl_parts_cc0 --limit 0
python scripts/combine_manifests.py --merge-auxiliary --output data/hitl_combined/manifest.jsonl data/hitl_cc0/manifest.jsonl data/hitl_parts_cc0/manifest.jsonl
vehicle-damage validate-manifest --manifest data/hitl_combined/manifest.jsonl
```

When auxiliary exterior-only rows are present, `damage_sample_fraction`
controls the expected share of damage-supervised training samples. Missing
exterior masks are treated as unknown and excluded from the exterior loss;
they are not silently converted to all-vehicle masks.

The source archive reuses some exact images across its two annotation subsets.
`--merge-auxiliary` attaches exterior polygons only to matching damage rows in
the training split, drops auxiliary copies of held-out images, and excludes
near-duplicate auxiliary images from held-out splits. Manifest validation then
rechecks exact hashes, perceptual hashes, and group isolation.

Full-checkpoint warm starts and backbone-only warm starts are both supported,
but require explicit commercial-use and provenance fields in the config. They
are mutually exclusive. This prevents an unreviewed external checkpoint from
silently entering the training lineage.

The manifest requires `train`, `validation`, `calibration`, and `test` groups for a complete experiment. All views of one vehicle/claim/burst belong to one `group_id`. Training refuses rows not explicitly approved for commercial use.

After training, calibrate only on the dedicated calibration split. The threshold is the highest observed value meeting the requested damage-case recall, where detection must cover at least 5% of that case's annotated damage pixels:

```powershell
vehicle-damage calibrate --checkpoint runs/baseline/best.pt --manifest data/manifest.jsonl --output runs/baseline/calibration.json --target-recall 0.97
vehicle-damage evaluate --checkpoint runs/baseline/best.pt --calibration runs/baseline/calibration.json --manifest data/manifest.jsonl --output runs/baseline/test-report.json
```

Production approval must use the frozen case/slice report and the gates in [VALIDATION.md](docs/VALIDATION.md), not the training history.

Run tiled inference after calibration:

```powershell
vehicle-damage infer --checkpoint runs/baseline/best.pt --calibration runs/baseline/calibration.json --image example.jpg --output prediction.png
```

`prediction.png` is an indexed class mask. The adjacent JSON includes image-quality warnings and a manual-review flag.

Benchmark the complete tiled/TTA path on deployment-class hardware and retain
the JSON with the release evidence:

```powershell
python scripts/benchmark_inference.py --checkpoint runs/pilot_cc0_hardneg/best.pt --image example.jpg --output runs/benchmark.json --device cuda
```

CUDA FP16 must have its own calibration artifact; never enable it while reusing
an FP32 threshold. The included development pair is:

```powershell
vehicle-damage infer --checkpoint runs/pilot_cc0_hardneg/best.pt --calibration runs/pilot_cc0_hardneg/calibration-fp16.json --image example.jpg --output prediction.png --device cuda
python scripts/benchmark_inference.py --checkpoint runs/pilot_cc0_hardneg/best.pt --image example.jpg --output runs/benchmark-fp16.json --device cuda --mixed-precision
```

The current DINOv2 development-only pair is:

```powershell
vehicle-damage infer --checkpoint runs/pilot_cc0_dinov2_4layer_role_split/best.pt --calibration runs/pilot_cc0_dinov2_4layer_role_split/development-profile-fp16-triage0450-scales1-1p5-mask0550-scale1-comp144.json --image example.jpg --output prediction.png --device cuda
python scripts/benchmark_inference.py --checkpoint runs/pilot_cc0_dinov2_4layer_role_split/best.pt --calibration runs/pilot_cc0_dinov2_4layer_role_split/development-profile-fp16-triage0450-scales1-1p5-mask0550-scale1-comp144.json --image example.jpg --output runs/dinov2-benchmark.json --device cuda
```

For the lower-latency development alternative, use the previous
`pilot_cc0_dinov2_fullscene_soup25/best.pt` checkpoint together with its
`development-dual-profile-fp16-triage0450-mask0690.json` profile.

A native-only diagnostic of the current role-split checkpoint measures
151.4 ms median and 6.61 images/s on the same RTX 3060 protocol, versus
335.0 ms and 2.98 images/s for the selected multiscale path. It is deliberately
not selected: native triage loses one case each under shadow, severe low
resolution, and simulated distance (97/98 in those conditions), while the
selected path retains 98/98 in all 11 conditions. The pinned comparison is
`runs/pilot_cc0_dinov2_4layer_role_split/SPEED_RECALL_TRADEOFF.json`.

The primary PNG is the native-scale 0.55 high-confidence mask after its
separately frozen 144-pixel component filter. Triage is unfiltered. The adjacent
`.review.png` contains lower-confidence evidence from the averaged
1.0x/1.5x triage path; any such evidence forces `manual_review`. Thresholds and
scales were selected on development validation. They must be recalibrated on
fresh data and confirmed on a new untouched test; the filenames and reports
deliberately identify them as development profiles.

The adjacent `.disagreement.png` localizes all branch conflicts: triage-only
damage, segmentation-only damage, and pixels where both branches detect damage
but assign different types. It is a reviewer aid and is never a substitute for
the primary high-confidence mask.

The adjacent JSON is the authoritative disposition, not the presence or
absence of pixels in the primary PNG. Its `decision` is one of
`no_damage_detected`, `damage_detected`, `manual_review_required`, or
`recapture_required`. Low resolution, severe exposure/glare, or blur requires
recapture. Localized glare, excessive threshold ambiguity, triage-only
evidence, segmentation-only evidence, or damage-type disagreement requires
manual review. `automated_decision_allowed` is false for either fail-closed
outcome; the legacy `manual_review` flag also remains true for both so older
consumers cannot accidentally accept a failed capture.

Calibration stores `exterior_floor`, tile size, overlap, flip-TTA, and numeric
precision as one frozen scoring path. Separate optional minimum-component
areas for triage and the high-confidence mask are stored in the same artifact,
so mask cleanup cannot silently erase recall evidence or differ between
evaluation and deployment. A floor of 1 disables exterior
suppression. Compare settings only on calibration data; inference and
evaluation read them from the artifact and reject conflicting overrides so a
threshold cannot silently be reused with a different predictor.

`evaluate` also reports one-to-one connected-region precision/recall and
false-positive components on clean views. Its development defaults are
`--min-component-pixels 16 --region-iou 0.10`; these are resolution-dependent
and must be predeclared and frozen for a production camera workflow. The
current leading validation result remains poor (35.38% region precision and
33.41% region recall), despite 100% permissive case recall, so the bundled
checkpoint must not be represented as a production-reliable segmenter.

Run the deterministic synthetic stress diagnostic only on a consumed
development split. It measures degradation under lighting, glare, dirt,
shadow, sensor noise, blur, detail loss, viewpoint, and simulated distance,
but is not release evidence:

```powershell
python scripts/benchmark_robustness.py --checkpoint runs/pilot_cc0_dinov2_4layer_role_split/best.pt --calibration runs/pilot_cc0_dinov2_4layer_role_split/development-profile-fp16-triage0450-scales1-1p5-mask0550-scale1-comp144.json --manifest data/hitl_hardneg/manifest.jsonl --split validation --output runs/pilot_cc0_dinov2_4layer_role_split/validation-robustness-development-mask0550-comp144.json
```

## Data plan that matters most

Start with consented first-party inspection images and clean vehicles from the exact deployment workflow. A useful pilot normally needs thousands of distinct vehicles, not augmented copies; every rare damage/slice needs enough independent cases for a meaningful confidence interval. Oversample subtle real dents and small cracks, but keep the test distribution natural. Run active-learning rounds by reviewing high-uncertainty predictions and all false negatives/false positives, then add panel gaps, reflections, glare, grime, shadows, decals, and previous repairs as explicit hard negatives.

Generate exact independent-group floors and statistically buffered targets from
the latest frozen evaluation:

```powershell
python scripts/plan_release_collection.py --report runs/baseline/test-report.json --output runs/baseline/release-data-plan.json
```

The plan keeps missing evidence separate from measured model failures; adding
test samples can close coverage gaps but cannot repair failed precision or
recall.

Known-background pixels in damage-annotated training images can be mined before
first-party clean data arrives. The miner protects a guard band around polygon
boundaries, caps each image's selected fraction, records checkpoint provenance,
and writes a new manifest rather than mutating source labels:

```powershell
python scripts/mine_hard_negatives.py --checkpoint runs/pilot_cc0_exterior/best.pt --calibration runs/pilot_cc0_exterior/calibration.json --manifest data/hitl_combined/manifest.jsonl --output-manifest data/hitl_hardneg/manifest.jsonl --minimum-score 0.2453135252 --maximum-fraction 0.10 --guard-radius 5
vehicle-damage validate-manifest --manifest data/hitl_hardneg/manifest.jsonl
```

Mined masks are model-generated proposals. Review them for missed subtle
damage before using them in any production training lineage; the included CC0
hard-negative run is explicitly a development experiment.

The DINOv2 follow-up re-mined and merged 27,015,889 hard-negative pixels, then
tested both full-model and frozen-backbone continuation. Both produced cleaner
masks but detected only 97/98 damage cases across the investigated recall-first
thresholds, so neither replaces the current 98/98 development leader. Exact
results and checkpoint hashes are in [PILOT_RESULTS.md](PILOT_RESULTS.md).

A separate `robust_v2` retraining experiment and its checkpoint soups were
rejected because they required substantially noisier thresholds. The later
`robust_v3` warm-start deliberately included full-scene tiny damage and severe
detail loss. A 75% original / 25% robust-v3 checkpoint interpolation became
the leading development checkpoint: at matched mask recall it improves
precision and background FPR. Its native-scale fast profile reduces synthetic
stress case misses from seven to three; the selected multiscale triage profile
recovers those last three cases while leaving the native-scale visible mask
unchanged. Synthetic nuisance performance remains diagnostic and does not
replace first-party clean/nuisance evaluation.

Damage-type thresholding, relative type-score biasing, and type-head-only
fine-tuning were also tested. Each either caused absent-type false alerts,
regressed at least one class or nuisance slice, or produced no improvement, so
the selected checkpoint and unbiased type assignment remain unchanged. See
[PILOT_RESULTS.md](PILOT_RESULTS.md) for the exact no-regression comparisons.

The completed CC0 mining run selected 17,350,905 pixels across 570 training
images. Its epoch-5 development checkpoint lowered calibration background FPR
from 40.40% to 35.88% at 98.46% measured case recall. It has not been scored
on the consumed pilot test and is not production-approved; see
[PILOT_RESULTS.md](PILOT_RESULTS.md).

See [DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md) before downloading a public dataset or checkpoint. In particular, CarDD is not cleared here for commercial training or even commercial testing without owner authorization.
