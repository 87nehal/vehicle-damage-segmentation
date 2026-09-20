# CC0 exterior-supervised pilot results

## Verdict

**Not approved for production.** The selected checkpoint passes the overall
damage-case recall gates on this small damage-only test, but fails 11 of 14
automated release gates. After adding fail-closed region coverage, precision,
and recall gates, the refreshed audit fails 14 of 17 gates. It is suitable
only for bootstrap labeling,
hard-negative mining, and further development.

## Data and isolation

- Humans in the Loop Car Parts and Car Damages, CC0 1.0.
- 814 damage-polygon images: 570 train / 98 validation / 65 calibration / 81 test.
- 998 car-part-polygon rows were prepared as exterior-only supervision with
  `damage_supervised: false`; they were never treated as clean negatives.
- The two archive subsets reuse 441 exact image files. The merge attached 301
  exterior masks to matching damage-training rows, discarded 140 auxiliary
  copies of held-out images, and retained 557 unique exterior-only training
  images.
- Final combined manifest: 1,371 rows, including 858 real exterior masks.
- Exact SHA-256, 64-bit difference-hash, and group audits pass across splits.
- Every damage test image contains damage. There are no verified clean images
  or dedicated glare/dirt/panel-gap/reflection/shadow negative slices.

### Latest clean-negative expansion

For the reported grille/body-contour false positives, a separate development
manifest was built at `data/hitl_coco_user_clean_hardneg/manifest.jsonl`:

- 535 COCO 2017 validation images containing cars, with zero damage masks and
  car bounding-box exterior masks (235 train / 100 validation / 100
  calibration / 100 test in the source clean set);
- the first supplied intact-car image plus 140 deterministic lighting, blur,
  colour, crop, rotation, and flip variants, all marked as explicit hard
  negatives;
- 162 additional handle/fuel-cap and side-view variants from the two latest
  supplied images, also marked as hard negatives;
- 1,652 total rows after combining the 814 supervised damage images, the COCO
  clean negatives, and all three user regression families.

The refined candidate reduces the held-out clean false-alert rate to 1/100 and
produces no damage mask on any of the three supplied intact-car regressions. It
detects 91/98 held-out damage cases (92.86%), so it remains a development
candidate rather than a production model. See
`runs/pilot_cc0_dinov2_parts_hardneg_long/` for the checkpoint, frozen profile,
and validation report.

## Selected model

- DeepLabV3/ResNet-50 with damage-presence, damage-type, and exterior heads.
- No external pretrained weights; lineage is self-trained CC0 data only.
- Class-aware crops, nuisance augmentation, focal presence loss,
  recall-oriented Tversky loss, and horizontal-flip tiled inference.
- Checkpoint: `runs/pilot_cc0_exterior/best.pt` (epoch 7).
- Size: 287,933,475 bytes.
- SHA-256: `80E3C36FFDAA6162D3A8B9FAFA3F9EED9FDAF68EEAB57D23C32B8BF3BA00C87B`.

## Calibration

The 65-case calibration split selected threshold `0.2453135252` for a target
97% case recall and minimum 5% damage coverage. Exterior floor values 0, 0.5,
and 1 were compared only on calibration data; full exterior weighting
(`exterior_floor: 0`) had the lowest negative-pixel FPR and was frozen.

- calibration case recall: 98.46% (64/65);
- calibration non-damage-pixel FPR: 40.40%.

## Pilot test (now consumed)

The 81-image damage-only test was evaluated for the exterior candidate and
one precision-focused follow-up. It is therefore consumed for future model
selection and cannot serve as the untouched production-approval test. The
metrics below remain useful pilot evidence but must be confirmed on a new
first-party holdout.

- damage-case recall: **100%** (81/81);
- 95% Wilson lower bound: **95.47%**;
- damage-pixel precision: **14.08%**;
- damage-pixel recall: **64.02%**;
- background-pixel false-positive rate: **41.13%**;
- predicted damage fraction: **43.31%** of test pixels;
- dent type case recall: 73.33% (44/60);
- scratch type case recall: 43.14% (22/51);
- crack/breakage type case recall: 93.85% (61/65);
- paint-damage type case recall: 5.41% (2/37);
- deformation/detachment type case recall: 47.73% (21/44);
- quality-review rate: 55.56% (45/81);
- clean-image false-alert rate: not measurable.

The refreshed fail-closed audit passes 3/17 gates: overall case recall, its confidence
bound, and crack/breakage recall. It fails four class-recall gates, clean-test
coverage, clean false-alert rate, all five required hard-negative slices, and
all three region gates because the legacy frozen test report has no region
measurements.
See `runs/pilot_cc0_exterior/release-audit.json`.

## Rejected experiment

Increasing background loss produced a precision-focused epoch with a slightly
better calibration FPR (39.95%), but its frozen test FPR worsened to 42.22%
and pixel precision fell to 13.86%. It was rejected; artifacts remain under
`runs/pilot_cc0_precision` for traceability.

A moderate background weight of 0.2 improved validation F2 to 0.5556, but at
the required calibration recall its negative-pixel FPR rose to 44.11%. It was
rejected on calibration without another test evaluation; artifacts remain
under `runs/pilot_cc0_balanced_presence`.

Increasing mined hard-negative weight from 3 to 10 raised validation pixel
precision to 24.73% but reduced recall at the default operating point. After
case-recall calibration, background FPR was 37.40%, worse than the 35.88%
moderate candidate, so it was rejected without test evaluation. Artifacts
remain under `runs/pilot_cc0_hardneg_strong`.

Independent per-type pixel thresholds were tested because the ordinary argmax
suppresses rare classes. At 90% type-case recall, every class falsely alerted
on 100% of validation images where that class was absent, so this approach was
rejected (`runs/pilot_cc0_hardneg/type-threshold-diagnostic.json`).

Two independent multi-label case heads were also tested. The average-pooling
head achieved only 0.45–0.67 validation AUROC by class; at 90% recall its
absent-class false-alert rates were 55–94%. A frozen-backbone average-plus-max
head achieved only 0.50–0.58 AUROC and 77–91% false-alert rates. The joint
model's segmentation calibration FPR was 37.84%, also worse than 35.88%.
These experiments were rejected without test evaluation; artifacts remain in
`runs/pilot_cc0_multilabel` and `runs/pilot_cc0_multilabel_balanced`. Future
case-head checkpoint selection now uses macro AUROC rather than prevalence-
inflated F2.

## Hard-negative development candidate

The canonical checkpoint mined false-positive pixels only from annotated
background in the 570 damage-supervised training images. Mining used its
frozen threshold, a five-pixel damage-boundary guard, and a 10% per-image cap.
The resulting manifest contains 570 hard-negative masks and 17,350,905 mined
pixels (9.70% of annotated background before the guard). Source labels were
not modified, and checkpoint lineage is recorded in
`data/hitl_hardneg/PROVENANCE.json`.

A six-epoch warm-started fine-tune selected epoch 5 on validation F2. Relative
to epoch 1, validation F2 rose from 0.5350 to 0.5444 and mean damage IoU from
0.0598 to 0.0666. On the dedicated calibration split, threshold
`0.2465689778` retained 98.46% case recall (64/65) while reducing
non-damage-pixel FPR from 40.40% to 35.88%, an 11.2% relative reduction.

- candidate checkpoint: `runs/pilot_cc0_hardneg/best.pt`;
- SHA-256: `9364FF2F5AC180F7BD7DDC95FAE1A8B70756A67337E62587F098C23BA8CF12A8`;
- FP32 reference calibration: `runs/pilot_cc0_hardneg/calibration.json`;
- recommended CUDA development calibration: `runs/pilot_cc0_hardneg/calibration-fp16.json`.

On an RTX 3060 Laptop GPU, the complete 768-pixel tiled plus horizontal-flip
TTA path took 686.9 ms median and 771.9 ms p95 for a 798×599 image (20 measured
runs after warm-up), or 1.46 images/s, with 933,906,432 bytes peak allocated
GPU memory. The reproducible report is
`runs/pilot_cc0_hardneg/benchmark-rtx3060-798x599.json`. This is asynchronous
triage performance, not real-time video performance.

CUDA FP16 autocast was separately calibrated rather than reusing the FP32
threshold. It retained 98.46% calibration case recall and produced 35.8759%
background FPR versus 35.8793% for FP32. On the same hardware/image protocol,
median latency fell to 349.1 ms, p95 to 358.2 ms, throughput rose to 2.86
images/s, and peak allocated memory fell to 829,297,152 bytes. Validation case
recall remained 100%; pixel precision, recall, and FPR changed by less than
0.011 percentage points. See
`runs/pilot_cc0_hardneg/benchmark-rtx3060-798x599-fp16.json`. FP16 remains
development-only and requires CUDA; CPU inference must use the FP32 artifact.

Connected-component scoring makes the localization limitation explicit. On
the 98 damage-only validation images, using the predeclared development
defaults of at least 16 pixels per component and IoU at least 0.10, the FP16
profile produced 47 matched regions, 606 false-positive regions, and 402
missed regions: 7.20% region precision and 10.47% region recall. The split has
no clean views, so false-positive components per clean view remain
unmeasurable. The full artifact is
`runs/pilot_cc0_hardneg/validation-report-fp16-regions.json`. This result shows
why 100% case recall under the permissive 5%-coverage triage rule is not a
claim of accurate damage segmentation.

A calibration-only connected-component diagnostic then tested output-area
cutoffs of 0, 16, 64, 256, 1,024, and 4,096 pixels. The 256-pixel setting was
the lowest-background-FPR option that retained the 97% calibration case-recall
target (64/65). On validation it still detected 98/98 cases and reduced false
regions from 606 to 320; region precision rose from 7.20% to 12.57%, while
matched regions fell from 47 to 46 and region recall from 10.47% to 10.24%.
Pixel FPR moved from 37.3249% to 37.2432%. Because the cutoff is
resolution-dependent, removes one matched region, and remains far below the
80% precision/90% recall release gates, it is retained as an optional
development profile rather than replacing the recall-first default. Artifacts:
`component-filter-diagnostic-fp16.json`,
`calibration-fp16-component256.json`, and
`validation-report-fp16-component256.json` under
`runs/pilot_cc0_hardneg`.

The complete filtered path, including mask conversion and connected-component
post-processing, measured 337.1 ms median, 340.7 ms p95, 2.97 images/s, and
829,297,152 bytes peak allocated GPU memory in a separate 20-run measurement.
Timing variation means this is not evidence that filtering speeds up the
network; it only shows that the full path remains in the same latency class.

Faster inference variants were measured but were not promoted. Disabling TTA
reached 185.1 ms median (5.40 images/s), but calibration background FPR rose
from 35.8759% to 36.0146%. A 640/160 tile profile reached 263.1 ms median and
25.75% calibration FPR, but validation case recall fell to 95/98 (96.94%);
forcing 100% calibration recall raised FPR to 43.26%. A 704/176 profile reached
275.8 ms median but calibration FPR rose to 39.34%, so it was rejected without
validation use. The 768/192 FP16-plus-TTA profile remains the development
default because it best preserves the recall objective among the measured
profiles, not because it meets production gates.

This is the leading **development candidate**, not a production-selected
ResNet model, but it has been superseded by the DINOv2 development candidate
below. It has deliberately not been compared on the consumed 81-image test.
Because the original calibration split has now compared development
candidates, release requires both a new calibration split and a new untouched,
clean-inclusive, slice-labelled test set.

## Apache-2.0 DINOv2 development candidate

The standard Meta DINOv2-small backbone was added behind the same commercial-
provenance gate. Official code and model-weight terms are Apache 2.0; XRay and
Cell-DINO variants with noncommercial terms are rejected. Exact upstream
revision, weight hash, and URLs are recorded in
`runs/pilot_cc0_dinov2/PRETRAINED_PROVENANCE.json` and
`docs/DATA_AND_LICENSES.md`.

Eight epochs on the same CC0 hard-negative manifest selected epoch 7 by
validation any-damage F2. Before tiled threshold calibration, validation F2
was 0.7112 and mean damage IoU was 0.1679, versus 0.5444 and 0.0666 for the
ResNet hard-negative candidate.

- checkpoint: `runs/pilot_cc0_dinov2/best.pt` (epoch 7, 98,920,636 bytes);
- SHA-256: `1ED94AD2CC0CE1F0A964AE1CF9A7301A76B4E1C67D0D7C8C5E9BFE3A8BAC930C`;
- architecture: DINOv2 ViT-S/14 plus separate presence, type, and exterior heads;
- inference: CUDA FP16, 518-pixel tiles, 126-pixel overlap, horizontal-flip TTA.

A strict 97%-target calibration threshold of 0.85995 retained 64/65 calibration
cases at 2.50% background FPR, but generalized to only 92/98 validation cases.
A 100%-calibration-recall threshold still reached only 93/98 validation cases.
Because this score shift failed the recall goal, a transparent validation
threshold sweep selected 0.525 as the highest-recall, best-pixel-F2 development
profile among settings that detected all 98 validation cases. This selection
consumes validation and is not an independent calibration or release result.

At that development operating point:

- case recall: 100% (98/98), Wilson 95% lower bound 96.23%;
- pixel precision: 39.24%;
- pixel recall: 85.47%;
- background-pixel FPR: 11.16%;
- region precision: 24.38% (127 matched, 394 false regions);
- region recall: 28.29% (322 missed regions);
- type-case recall: dent 70.83%, scratch 63.49%, crack/breakage 92.31%, paint damage 55.00%, deformation/detachment 77.55%;
- fail-closed development audit: 4/17 gates pass.

Compared with the prior FP16 ResNet profile on the same validation data, DINOv2
raises pixel precision from 13.27% to 39.24%, pixel recall from 67.73% to
85.47%, and region precision/recall from 7.20%/10.47% to 24.38%/28.29%, while
reducing background FPR from 37.32% to 11.16%.

On the RTX 3060 Laptop protocol, the complete DINOv2 scoring and mask path took
94.2 ms median and 115.5 ms p95, delivered 10.62 images/s, and used 280,725,504
bytes peak allocated GPU memory. The prior FP16 ResNet path took 349.1 ms and
829,297,152 bytes. See
`runs/pilot_cc0_dinov2/benchmark-rtx3060-798x599-fp16-threshold0525.json`.

### Dual-threshold development output

Because one threshold could not simultaneously preserve the case-level recall
goal and produce a reasonably precise mask, inference and evaluation now
support two explicit roles. The 0.525 triage threshold produces a review signal
and preserves 98/98 development cases; a separately selected 0.70 threshold
produces the high-confidence segmentation. Pixels present only in the triage
mask are saved in an adjacent `.review.png` and force manual review rather than
being silently discarded.

At the dual development profile:

- triage case recall: 100% (98/98), Wilson 95% lower bound 96.23%;
- high-confidence pixel precision: 48.70%;
- high-confidence pixel recall: 72.06%;
- high-confidence background-pixel FPR: 6.40%;
- high-confidence region precision: 31.89% (147 matched, 314 false regions);
- high-confidence region recall: 32.74% (302 missed regions);
- fail-closed development audit: 4/17 gates pass.

The complete dual path measured 101.9 ms median, 119.6 ms p95, 9.81 images/s,
and 280,725,504 bytes peak allocated GPU memory on the RTX 3060 Laptop. See
`development-dual-profile-fp16-triage0525-mask0700.json`,
`validation-report-fp16-dual-triage0525-mask0700.json`, and
`benchmark-rtx3060-798x599-fp16-dual-triage0525-mask0700.json` under
`runs/pilot_cc0_dinov2/`.

Both thresholds were selected after inspecting validation, so this is a safer
development output contract, not independent calibration or production proof.

This was the leading **development** candidate before the full-scene follow-up
below; it remains the first parent and rollback checkpoint.
It has not touched the consumed test split. Region quality, four damage types,
and every clean/hard-negative gate still fail or remain unmeasurable.

## DINOv2 self-mined hard-negative follow-up (rejected)

The leading DINOv2 model was used to mine additional high-score background
regions from the training split. Unioning them with the earlier masks produced
27,015,889 supervised hard-negative pixels on 570 images (12,853,652 pixels
were newly selected before overlap removal). The manifest validates with zero
errors; provenance is recorded in `data/hitl_hardneg_dinov2/PROVENANCE.json`.

Two continuation strategies were evaluated without touching the test split:

- The four-epoch full-model run selected epoch 1. Uncalibrated validation mean
  damage IoU improved to 0.1885, but F2 fell to 0.7072. At its 0.72445
  calibration threshold it detected 97/98 validation cases, with 60.62% pixel
  precision, 55.87% pixel recall, 3.06% background FPR, 36.29% region
  precision, and 30.07% region recall. The same case remained missed even at a
  0.20 threshold. Checkpoint SHA-256:
  `98FCCBAACA81DFE2B1B15740B02DB4C781BA481D6261955D9B92D9952488B2AF`.
- The two-epoch recall-guard run froze the backbone, used a 1e-5 head learning
  rate, and reduced hard-negative weighting to 1.5. Its selected epoch improved
  uncalibrated F2 to 0.7179 and mean damage IoU to 0.1806. Strict calibration
  detected 92/98 validation cases; the recall-first sweep recovered only 97/98
  throughout thresholds 0.40--0.60. Checkpoint SHA-256:
  `BF8E47B6060A39D2D1FFE28F2CA77451371206D51C5BAAED4F88C605B76AE615`.

Both variants are rejected as primary detectors because they lose a damage case
that the 0.525 leading profile detects. Their cleaner-mask result supports the
implemented dual-threshold review design, but it is not a release claim and
does not compensate for absent verified clean data.

## Robustness-augmentation and checkpoint-soup follow-up (rejected)

The training pipeline now has explicit augmentation profiles. `baseline_v1`
preserves the exact earlier behavior. `robust_v2` additionally applies aligned
mild projective/affine viewpoint changes, soft cast shadows, and phone-like
sensor noise, while reverting any geometric transform that erases the rare
class retained by a positive crop. The experiment is reproducible with
`configs/pilot_cc0_dinov2_robustaug.yaml`; unknown profile names fail closed.

Eight fresh `robust_v2` epochs produced two candidates:

- The epoch-7 recall-F2 checkpoint reached uncalibrated F2 0.7176 and mean
  damage IoU 0.1527. Its strict calibration generalized to 95/98 validation
  cases. Reaching 98/98 required threshold 0.35 and 17.39% background FPR. Its
  best region-F2 mask at 0.75 reached 53.15% pixel precision, 67.77% pixel
  recall, 5.04% background FPR, and 33.25%/31.18% region precision/recall.
  SHA-256: `09B53E1FF43C07AD115ED75AA5510944A370939D3FDA283945C78281BEB83CC1`.
- The epoch-8 best-IoU checkpoint reached uncalibrated F2 0.7142 and mean
  damage IoU 0.1657. Its 98/98 threshold was at most 0.35, with at least 15.08%
  background FPR. At its best region-F2 threshold of 0.75 it reached 56.50%
  pixel precision, 63.84% pixel recall, 4.15% background FPR, and
  37.08%/31.63% region precision/recall. SHA-256:
  `2E87B7CD850AE702C8477685A9A508CEDED538C0D95975B899A9F584917023E4`.

Compatibility-checked linear checkpoint soups were also tested to obtain those
nuisance-trained weights without a second inference pass. Exact parent hashes
and mixture weights are embedded beside each checkpoint:

- 50% original / 50% robust-IoU: 98/98 only at threshold 0.35, with 15.26%
  background FPR; best region F2 remained below the leader. SHA-256:
  `11CCCCF001DD3D0574DF8033199DF08A5252BFA86A6650BFB5008B5AFEAD9ED6`.
- 75% original / 25% robust-IoU: 98/98 at threshold 0.40, with 14.31%
  background FPR. Its 0.70 mask modestly improved precision/FPR but reduced
  pixel and region recall, so it did not dominate the leader. SHA-256:
  `087D8AC2C4627B79E24CCC5B80AC6B2B3DAD0AC0B24A28DF5BA8248502EAD6A7`.

All four candidates are rejected for primary use. `robust_v2` remains available
for retraining when real nuisance-labelled data arrives, but synthetic
augmentation is not evidence of real glare, dirt, shadow, distance, or angle
robustness.

## Full-scene robustness follow-up and previous development leader

A deterministic `synthetic_stress_v1` diagnostic was added for clean input,
under/overexposure, localized glare, dirt, cast shadow, sensor noise, blur,
five-fold detail loss, label-aligned viewpoint changes, and simulated distance.
It runs the frozen dual-threshold path, records missed groups and threshold
sweeps, and clearly marks its output as consumed-development diagnostics rather
than production evidence. On the prior leader, clean/glare/overexposure/noise/
blur retained 98/98 case recall, but low resolution fell to 94/98 and
underexposure, dirt, shadow, viewpoint, and distance each reached 97/98.

Inspection showed that the five unique missed cases had damage occupying only
0.36%--0.84% of the image. The versioned `robust_v3` profile therefore mixes
30% label-safe full-scene training views with the existing positive crops and
adds seeded 20%--55% detail degradation. A four-epoch warm-start from the
original leader selected epoch 4 (uncalibrated validation F2 0.7141, mean
damage IoU 0.1845). Its standalone checkpoint recovered 98/98 cases in all 11
stress conditions, but a high-confidence 0.80 mask reduced clean pixel recall
to 55.96%, so it was not promoted directly.

- standalone checkpoint: `runs/pilot_cc0_dinov2_fullscene/best.pt`;
- standalone SHA-256: `289161FC16043FD7DE4B6207F32418EDF00C2B335EE6FD647351227822EFFA5F`;
- strict calibration: 64/65 cases at threshold 0.84773 and 2.18% background
  FPR, but only 94/98 validation cases, confirming calibration-to-validation
  score shift;
- 50/50 original/full-scene soup: rejected because 98/98 required threshold
  0.40 with 12.89% background FPR; SHA-256
  `A8330FC1F3BA39D2F76D9DA07B9B8D9A3CA6512CD2022AFAE40C01C7D5779E77`.

A compatibility-checked 75% original / 25% full-scene interpolation gave the
best measured single-model trade-off and became the previous development
leader:

- checkpoint: `runs/pilot_cc0_dinov2_fullscene_soup25/best.pt`;
- SHA-256: `B9AB98C544EEA4625D9DE6D1610DC424BAF2DB28AAE27DFC26EAB4BF088D4D84`;
- selected recall-rescue profile:
  `development-recall-rescue-profile-fp16-triage0450-scales1-1p5-mask0690-scale1-comp24.json`;
- inference scales: native plus 1.5x averaged for the 0.45 triage signal;
  native only for the 0.69 high-confidence mask;
- component filtering: none for triage; components below 24 pixels removed
  only from the high-confidence mask;
- triage case recall: 98/98, Wilson 95% lower 96.23%;
- high-confidence pixel precision/recall: 49.65% / 72.06%;
- high-confidence background-pixel FPR: 6.16%;
- high-confidence region precision/recall: 33.33% / 32.52%
  (146 matched, 292 false, 303 missed regions);
- fail-closed development audit: 4/17 gates pass;
- RTX 3060 Laptop: 333.8 ms median, 356.9 ms p95, 3.00 images/s, and
  315,065,856 bytes peak allocated GPU memory.

Relative to the previous leader at matched clean pixel recall, the promoted
candidate improves clean pixel precision from 48.70% to 49.65%, lowers mask
background FPR from 6.40% to 6.16%, and raises region precision from 31.89% to
33.33%, with one fewer matched region. Its native-scale fast profile improves
low-resolution case recall from 94/98 to 97/98 and reduces aggregate
condition-level misses from seven to three. Averaging native and 1.5x scores
for triage then recovers `hitl_00587` under shadow, `hitl_00584` under severe
resolution loss, and `hitl_00660` at simulated distance: all 98/98 cases are
detected in every one of the 11 synthetic conditions. The native-scale visible
mask is bit-identical to the fast profile. Clean transformed-background triage
FPR rises from 12.40% to 13.14%.

The precise-mask component filter was selected separately so it cannot erase
triage evidence. Calibration allowed at most 32 pixels without losing a case,
a matched region, or more than 0.1% relative pixel recall. The 32-pixel option
then lost two matched regions under deterministic sensor noise and was
rejected. A 24-pixel guard preserves every matched region in clean validation
and all 11 stress conditions while reducing false regions in every condition;
on the unmodified validation images they fall from 302 to 292. This selection
also consumes development data and is not evidence about real clean vehicles.

The native-scale fast profile remains available as
`development-dual-profile-fp16-triage0450-mask0690.json`. It runs at 97.2 ms
median, 114.9 ms p95, 10.28 images/s, and 280,725,504 bytes peak allocated GPU
memory, but retains the three synthetic misses above. The recall-rescue profile
is selected because the project explicitly prioritizes false-negative
reduction; deployments with a hard latency limit must revalidate the fast
profile rather than silently changing the scoring path.

Artifacts are under `runs/pilot_cc0_dinov2_fullscene_soup25/`, including the
canonical validation report, fail-closed audit, fine threshold sweep,
`benchmark-rtx3060-fp16-recall-rescue-comp24.json`, and
`validation-robustness-recall-rescue-comp24.json`. The hash-pinned selection is
`runs/SELECTED_DEVELOPMENT_MODEL.json`. All operating points, inference scales,
and checkpoint mixtures were selected after inspecting validation. Neither the
stress result nor the public damage-only split supplies verified clean
vehicles, real nuisance false-alert rates, or independent production evidence.

### Rejected damage-type calibration and refinement

Three targeted attempts to improve damage-type recall were evaluated without
changing the selected binary-presence operating point:

- Direct per-type thresholds calibrated near 90% recall produced 71%--100%
  absent-class image false-alert rates and were rejected.
- Relative type-probability multipliers applied only inside already accepted
  damage pixels left binary presence and masks unchanged. The mild guarded
  vector `[1.324854, 1.199215, 1.1, 1.273490, 1.456857]` improved clean dent
  hits from 57 to 58 with no clean class loss, but lost one scratch hit under
  underexposure and viewpoint, one crack hit under sensor noise, and one paint
  hit under severe resolution loss. It was rejected under the no-regression
  rule.
- An eight-epoch type-head-only continuation from the selected checkpoint was
  verified to change exactly the five `type_head` tensors while all other 184
  tensors remained bit-identical. Epoch 1 changed clean hit counts from
  `57/42/75/24/41` to `62/45/73/23/41` for dent/scratch/crack/paint/
  deformation; epoch 7 reached `61/43/74/22/41`. Weight interpolations at
  10%, 15%, 20%, and 25% each lost at least one class. The 12.5% interpolation
  exactly tied all five baseline hit counts and supplied no benefit.

The experimental multiplier profile and type-head checkpoints remain as
diagnostic artifacts, but the hash-pinned selected checkpoint and unbiased
component-filtered profile are unchanged. These results indicate that neither
post-hoc reweighting nor a small type-head-only update resolves the type
confusions without new representative labels and stronger dense features.

### Four-layer DINOv2 role-split decoder (new development leader)

The original DINOv2 decoder used only the final transformer block. Following
the official DINOv2 four-layer semantic-segmentation pattern, a new
`dinov2_4layer` architecture concatenates the final four normalized patch-token
grids and learns a 1x1 projection before the existing dense heads. The
projection is initialized as an exact identity on the final layer, so the
warm-started model is numerically identical before optimization. Unit tests
verify this equivalence on non-patch-multiple input sizes.

Six epochs were trained on the commercially permitted merged CC0 and
DINOv2-mined hard-negative manifest with the backbone frozen. Epoch 3 was
selected by resized any-damage F2. Its checkpoint is
`runs/pilot_cc0_dinov2_4layer_refine/best.pt`, SHA-256
`91F903C8035FBD9C9039C86EC3B7139AF4A3E73AB3D910FF45153DB23F958650`.
The final development profile uses the existing native+1.5x 0.45 triage path,
a native 0.55 precise mask, and a 144-pixel mask-only component filter. The
threshold was recall-guarded on calibration; the filter is explicitly selected
on consumed validation/stress. A 160-pixel filter was rejected because it lost
one matched region under simulated distance.

Against the previous hash-pinned selection on clean consumed validation:

| Metric | Previous selection | Role-split selection |
|---|---:|---:|
| Triage cases | 98/98 | 98/98 |
| Pixel precision | 49.65% | 50.22% |
| Pixel recall | 72.06% | 73.34% |
| Background FPR | 6.16% | 6.13% |
| Region precision | 33.33% | 35.38% |
| Region recall | 32.52% | 33.41% |
| Region TP / FP / FN | 146 / 292 / 303 | 150 / 274 / 299 |
| Dent / scratch / crack / paint / deformation triage hits | 57 / 42 / 75 / 24 / 41 | 57 / 42 / 75 / 24 / 41 |

The standalone four-layer checkpoint initially changed clean type hits to
`62/44/73/25/41`; global thresholds, parameter soups, split type heads, and
probability fusion could not recover crack without another type regression.
The selected role split solves this by accumulating the previous parent heads
for triage and the four-layer heads for segmentation after tile/TTA fusion,
sharing the same bit-identical backbone. A real-image equivalence check gave
maximum absolute error 0.0 against each source path.

The complete role-split 11-condition stress run retains 98/98 cases everywhere
and preserves all five per-type hit counts exactly in every condition. Relative
to the previous selected model, pixel recall improves by 0.73--5.99 percentage points
in every condition and matched regions never decrease. False regions decrease
in 10 conditions but increase by 10 under severe low resolution; pixel
precision also decreases under blur, low resolution, and simulated distance.
Branch-selective execution omits the unused four-layer projection and
segmentation heads at the 1.5x triage-only scale. Its outputs are exactly equal
to exhaustive execution on the checked real image (maximum absolute
difference 0.0 for both roles and exterior maps). The complete RTX 3060 Laptop
path measures 335.0 ms median, 369.6 ms p95, 2.98 images/s, and 374,984,192
bytes peak allocated memory, versus 333.8 ms, 356.9 ms, 3.00 images/s, and
315,065,856 bytes for the previous selection.

The parent triage compatibility projection is itself a frozen identity that
selects the already-computed final DINO feature level. Bypassing that redundant
convolution changes FP16 pre-head features by at most 0.00775 on the checked
tile, but all triage logits and all complete tiled/TTA/multiscale probabilities
remain exactly equal (maximum absolute difference 0.0). The evidence is pinned
in `DIRECT_FINAL_EQUIVALENCE.json`; the measured full-path timing change from
335.4 ms to 335.0 ms is too small to claim as a material speedup.

The selected checkpoint was also benchmarked with an explicitly non-selected
native-only triage diagnostic. It measures 151.4 ms median, 166.7 ms p95,
6.61 images/s, and 374,984,192 bytes peak allocated memory: a 54.8% latency
reduction and 2.21x throughput relative to the selected role-split path.
However, the tensor-identical parent native triage evidence retains only 97/98
cases under shadow (`hitl_00587`), severe low resolution
(`hitl_00584`), and simulated distance (`hitl_00660`). The 1.0x/1.5x
selected path retains 98/98 in every condition. Threshold reductions recover
the native misses only with higher transformed-background FPR, so the native
profile is not promoted. The hash-pinned decision is
`runs/pilot_cc0_dinov2_4layer_role_split/SPEED_RECALL_TRADEOFF.json`.

The role-split checkpoint is now the hash-pinned development selection at
`runs/pilot_cc0_dinov2_4layer_role_split/best.pt`, SHA-256
`03F64E322347ADAA5062B1699566693BC25142F3E28080E514F26A2D178A00D8`.
Its fail-closed audit remains only 4/17. Severe low resolution adds 10 false
regions, several degraded conditions lose pixel precision, the operating point
consumes validation/stress data, and no independently reviewed clean vehicles
exist; promotion is not a production-reliability claim.

### Rejected quarter-resolution RGB detail refiner

A zero-initialized 60,967-parameter RGB residual was added only to the precise
mask branch, providing quarter-resolution detail while the DINOv2 backbone,
coarse mask heads, and triage heads remained frozen. The warm start was exactly
equal across all six output tensors, and all 206 parent tensors remained
bit-identical after eight training epochs.

Calibration selected a 0.565 fallback threshold under the 99.9% pixel-recall
guard. On clean consumed validation it preserved 98/98 cases and all type hits,
retained 99.940% of selected pixel recall, improved pixel precision from
50.22% to 50.40%, reduced background FPR from 6.130% to 6.082%, and changed
region TP/FP/FN from 150/274/299 to 151/274/298.

The candidate was nevertheless rejected. Across the full 11-condition stress
diagnostic, recall retention fell to 98.36% under underexposure and below
99.9% under blur, low resolution, and viewpoint. Dirt lost two matched regions,
low resolution lost three, glare added five false regions, and shadow added
one. Median RTX 3060 Laptop latency also increased from 335.0 ms to 355.0 ms.
The complete hash-bound evidence is
`runs/pilot_cc0_dinov2_detail_refine/DETAIL_REFINER_EXPERIMENT.json`;
the selected role-split checkpoint is unchanged.

### Damage-unverified CC0 active-review queues

The selected checkpoint scored all 998 commercially permitted exterior-part
images that lack exhaustive damage labels. The hash-bound output is
`review-priority-hitl-parts-all.json`, SHA-256
`D6E80A8E396FE7245DC716542F461F7F285E0A74F7D64A9D35A496EDE1F105DA`.
It is explicitly review-priority metadata, not ground truth.

Every image produced some triage and precise-mask response; 995/998 had a
precise mask covering at least 1% of the image and 858/998 at least 5%. Median
precise-mask area was 10.95%, with 90th/95th/99th percentiles of
22.48%/27.00%/36.16%. Visual inspection of both ranking extremes found obvious
collision damage among the strongest responses and real but smaller,
peripheral, or partially out-of-view damage among weak responses. The source
pool must therefore not be reinterpreted as clean negatives.

Two blinded, disjoint, 100-image browser batches were exported:

- `review_batches/cc0_model_priority_001` presents the strongest responses,
  prioritizing false-positive discovery if an image is independently confirmed
  clean.
- `review_batches/cc0_low_response_001` presents the weakest responses,
  prioritizing subtle/missed-damage discovery if an image is confirmed damaged.

The HTML does not expose model scores. Each batch is bound to source image,
manifest, ranking, checkpoint, and profile hashes. No row enters damage
supervision until two distinct reviewers agree (or an automotive assessor
adjudicates); confirmed damage still requires a dense mask.

The combined manifest removes images already represented by the damage
dataset, leaving 557 genuinely new damage-unsupervised training candidates.
Their separate ranking is
`review-priority-hitl-combined-unique-auxiliary.json`, SHA-256
`5C082BB66A45213262BE09A48E9F15E124C14DB50CB14E991A61252B68D928EF`.
All 557 receive some precise-mask response, 556 cover at least 1%, and 481
cover at least 5%; median precise area is 11.11%. The 25 weakest responses were
exported for recall-focused correction under
`annotation_batches/cc0_unique_low_response_001`.

The CVAT batch contains 25 source images, 25 indexed six-class masks, 25
component object masks, a contiguous label map, and a hash-bound manifest. The
image archive SHA-256 is
`1341A40AB841942FC24661954DEE14277142AF5B97F852AA90CFC33F7ACC57C8`;
the Segmentation Mask 1.1 preannotation archive SHA-256 is
`DA0A743846945219ECB2093043550E71159B5CD7CF715DEF1D7FA371528EDD2B`.
Model masks remain suggestions. The correction importer requires distinct
annotator/reviewer identities, rejects empty or invalid masks, updates only
matching unsupervised training rows, and validates the complete resulting
manifest before it can enter training.

## Required before production

A public-source clean-candidate queue now reduces, but does not close, this
gap. `review_batches/open_images_license_validation_001` contains 109 original
Open Images vehicle files whose official byte lengths and MD5 values were
verified. Every row remains `training_eligible: false`, rights-unapproved, and
damage-unreviewed. Expansion paused after the source host returned HTTP 429;
the downloader now records exact rejection causes and stops after three
consecutive rate-limit responses. These candidates count as zero clean groups
until a qualified rights review and two independent automotive reviews are
completed. They also cannot replace first-party deployment data.

Acquire consented first-party images from the deployment workflow: clean
vehicles and pixel-marked glare, reflections, dirt/water, seams, contours,
shadows, decals, and previous repairs. Add subtle real scratches and paint
damage, dark/metallic paint, wet/night scenes, full-car/close-up pairs, and
vehicle/claim group IDs. Retrain, calibrate once, and pass every gate on a
geographically and temporally separate test set.
