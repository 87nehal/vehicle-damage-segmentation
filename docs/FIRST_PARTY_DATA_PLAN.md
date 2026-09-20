# First-party data required for a releasable model

The CC0 checkpoint is a bootstrap labeler, not evidence that the deployment
camera population is safe. Capture data through the real inspection workflow
and group every photo from one vehicle, claim, burst, or incident under one
`group_id`. No group, exact image, or near duplicate may cross splits.

## Capture and annotation

Collect clean and damaged vehicles across daylight, night, indoor, outdoor,
wet, dirty, dark/light/metallic paint, glare, shadow, blur, compression,
front/rear/side/oblique views, and close-up/medium/full-car distances. Keep the
original pixels and camera metadata where policy permits. Record device family,
site, date window, distance, angle, weather/lighting, paint, cleanliness, and
the exact nuisance tags used by the release audit: `glare`, `dirt`,
`panel_gap`, `reflection`, and `shadow`.

Create an explicit indexed damage mask for every image, including an all-zero
mask for reviewed clean images. Also annotate the inspectable exterior and
confusing non-damage regions when possible. Damage masks should be independently
reviewed twice, with disagreements adjudicated by an automotive assessor.
Automatically mined hard-negative masks are proposals only: a reviewer must
remove missed scratches, dents, chips, and prior repairs before they are used
for a production training run.

### Review workflow

Create a browser batch from commercially approved manifest rows tagged
`damage_unverified`:

```powershell
python scripts/create_review_batch.py `
  --manifest data/first_party/manifest.jsonl `
  --output-dir review_batches/first_party_001 `
  --tag damage_unverified `
  --limit 100
```

Two independent reviewers open `review.html`, enter distinct reviewer IDs,
complete every item, and download their CSV files. The page records clean,
damaged, or unusable; the five release nuisance tags plus wet/water spots,
styling creases, decals, and prior repairs; and distance, angle, lighting, and
cleanliness. A reset button clears local browser state before another reviewer
uses the same workstation.

Import only exact agreements:

```powershell
python scripts/apply_clean_reviews.py `
  --manifest data/first_party/manifest.jsonl `
  --review-a review-assessor-a.csv `
  --review-b review-assessor-b.csv `
  --output-manifest data/first_party_reviewed/manifest.jsonl

vehicle-damage validate-manifest `
  --manifest data/first_party_reviewed/manifest.jsonl
```

If the two reviews disagree, a third automotive assessor completes a review
CSV for the disputed samples and the import is rerun with
`--adjudication assessor.csv`. The assessor ID must differ from both initial
reviewer IDs and every disputed SHA-256 must be resolved.

The importer verifies image SHA-256 identities, requires two distinct reviewer
IDs, refuses to overwrite an existing output, creates explicit empty masks only
for exact clean agreements, preserves split/group/license provenance, and
writes hashes and disagreement IDs to a review provenance report. Exact
damaged agreements still require pixel masks. Any disagreement must be resolved
by an automotive assessor through `--adjudication`; it is never silently
accepted.

## Isolation and minimum evaluation coverage

- Training/validation: target thousands of distinct vehicles, with clean
  examples at least as common as they are in deployment and deliberate
  oversampling of subtle scratches, paint damage, and low-light cases.
- New calibration: keep vehicle, site, and preferably later time windows
  separate from training; use it once to freeze the operating threshold.
- Untouched test: at least 73 damage cases overall, at least 30 independent
  cases for every damage class, and at least 100 independently reviewed clean
  vehicles.
- Hard-negative test slices: at least 40 independent clean examples for each
  of glare, dirt, panel gaps/contours, reflections, and shadows. One image may
  carry multiple truthful tags, but repeated views of one vehicle still count
  as one independent group when interpreting confidence.
- Shadow deployment set: collect later and from another site/device mix to
  measure drift, review rate, latency, and failure modes before automation.

These are minimum audit floors, not recommended stopping points. More samples
are needed whenever a confidence bound misses its gate or a material camera,
paint, geography, weather, or severity slice is underrepresented.

Generate exact floors and buffered quotas from the latest frozen evaluation:

```powershell
python scripts/plan_release_collection.py `
  --report runs/pilot_cc0_dinov2_4layer_role_split/validation-report-development-mask0550-comp144.json `
  --output runs/pilot_cc0_dinov2_4layer_role_split/RELEASE_DATA_COLLECTION_PLAN.json
```

For the current gates, the bare minimum of 73 damage groups passes the 95%
Wilson lower bound only with zero false negatives. A buffered target of 142
permits two. The 100-clean-group floor likewise permits zero false alerts;
142 clean groups permits two. Each 40-group clean nuisance slice permits zero
alerts, while 53 permits one. These are independent vehicle/claim groups, and
the per-class floor remains 30 for each of the five damage types. The generated
plan separately identifies missing evidence and measured model failures:
collecting more test data cannot repair the current class or region errors.

## Promotion sequence

1. Ingest rows with explicit commercial approval and immutable provenance.
2. Validate the manifest and resolve every leakage or mask error.
3. Fine-tune using train only; choose checkpoints on validation only.
4. Calibrate once on the new calibration split.
5. Evaluate once on the untouched test and run `audit-release`.
6. Keep human review/recapture enabled until every gate, slice review, and
   deployment-hardware benchmark passes.
