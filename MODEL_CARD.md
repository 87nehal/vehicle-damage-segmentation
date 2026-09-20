# Vehicle damage segmentation model card

## Status

Bootstrap pilot only; not production-approved. The frozen, already-tested CC0
checkpoint is `runs/pilot_cc0_exterior/best.pt`. The leading development
candidate is `runs/pilot_cc0_dinov2_4layer_role_split/best.pt`. It shares one
standard Apache-2.0 DINOv2-small backbone, uses the previous selected
final-layer heads for recall-first triage, and uses frozen-backbone four-layer
heads for the precise mask. Its validation-selected profile averages native
and 1.5x parent-head scores for a 0.45 review threshold and reaches 100% case
recall; its native-scale 0.55 four-layer mask uses a 144-pixel component guard
and reaches 50.22% pixel precision, 73.34% pixel recall, and 6.13% background
FPR, but only 35.38% region precision and 33.41% region recall.
The validation and calibration splits have compared candidates and operating
points, and the model has not been evaluated on a new untouched test. Results,
checksums, provenance, and blockers are documented in
`PILOT_RESULTS.md`.

The currently wired development candidate is the hard-negative refinement in
`runs/pilot_cc0_dinov2_parts_hardneg_long/best.pt`. It was trained with the
original CC0 damage set plus 535 COCO 2017 full-scene car negatives, 141
lighting/blur/flip augmentations of the first intact-car regression image, and
162 handle/fuel-cap and side-view augmentations. Its profile applies a
1,200-pixel connected-component guard. On the held-out mixed validation it
detects 91/98 damage cases (92.86%) and raises alerts on 1/100 clean COCO
cases (1.0%). All three supplied clean regression images yield no damage
mask/detections; low-resolution examples are routed to recapture. The
candidate is explicitly not production-approved because recall is below the
requested release bar.

For CUDA development inference, the selected profile freezes FP16, tile size,
overlap, TTA, both thresholds, 1.0x/1.5x triage scales, a native-only
segmentation scale, separate parent/four-layer heads, and zero-pixel
triage/144-pixel segmentation component filters. The independently calibrated strict profile did not
transfer at the required recall, so the leading thresholds and scales were
selected on consumed validation and are development-only. They are not a
substitute for fresh calibration or the missing clean/hard-negative release
test.

## Intended use

Visible exterior vehicle damage triage from RGB photographs: dents, scratches, crack/breakage, paint damage, and deformation/detachment. Outputs are segmentation masks plus a manual-review decision. It is not a repair-cost estimator and cannot detect hidden structural/mechanical damage.

## Architecture

The repository supports a DeepLabV3/ResNet-50 baseline and several DINOv2
ViT-S/14 dense decoders. The leading role-split model shares one backbone but
has separate parent triage and four-layer segmentation head sets; each set has
damage-presence, damage-type, and inspectable-exterior heads. Decoupling presence from type
prevents background imbalance from suppressing rare damage classes. Training
combines focal loss, positive-pixel type loss, recall-weighted Tversky loss,
hard-negative upweighting, and nuisance augmentation. Inference uses
overlapping tiles and horizontal-flip test-time augmentation so small damage in
full-car images is not collapsed by a single resize.

## Known limitations

Inference now emits a fail-closed four-way decision and separate triage-only
and all-branch-disagreement masks. Resolution, severe exposure/glare, or blur
requires recapture; localized glare, near-threshold ambiguity, or branch
disagreement requires human review. The selected-manifest smoke case pins
these semantics, but this routing is a safety mechanism rather than evidence
of production accuracy.

The public pilot test split is damage-only, cannot measure clean-image false
alerts, and is now consumed because it was used to compare two candidates.
Reported damage-case recall is 100% (81/81; 95% Wilson lower bound
95.47%), but this comes with only 14.08% pixel precision and a 41.13%
background-pixel FPR. Paint type case recall is 5.41%, scratch is 43.14%, and
the refreshed automated release audit fails 14/17 gates. Glare, grime, seams,
dark/metallic paint, subtle damage, glass, occlusion, and unseen camera
pipelines require first-party held-out testing.

The leading DINOv2 profile passes only 4/17 development release gates. It fails four
of five class-recall gates, both region-quality gates, clean-test coverage,
clean false-alert measurement, and all required glare/dirt/panel-gap/
reflection/shadow slices. Its 0.45 threshold was chosen after inspecting
validation results and therefore cannot be treated as independently validated.

The leading DINOv2 development candidate's damage-only recall-rescue validation
result has only 35.38% connected-region precision and 33.41% connected-region recall at
the documented 16-pixel/0.10-IoU development rule. Its 100% validation
case-recall result means that at least 5% of annotated damage was covered in
each case by the lower review threshold; it does not mean that damage was
localized accurately by the high-confidence mask. There are no
verified clean validation views from which to estimate false-positive regions
per clean vehicle.

Two DINOv2 continuation runs using 27,015,889 merged self-mined hard-negative
pixels produced cleaner masks but topped out at 97/98 validation cases over the
reported recall-first threshold ranges. They are retained as rejected
development artifacts and are not the leading checkpoint.

The versioned `robust_v2` augmentation experiment adds perspective/affine
viewpoint perturbations, cast shadows, and sensor noise. Its checkpoints and
two weight-averaged soups improved selected precision measures but needed
higher-background-FPR thresholds than the leader to recover 98/98 validation
cases. They are rejected candidates; synthetic corruptions do not validate
real nuisance robustness.

The later `robust_v3` warm-start adds full-scene training views and severe
detail loss. Its standalone checkpoint recovered 98/98 cases under every
synthetic stress condition but sacrificed too much high-confidence mask
recall. A 75% original / 25% robust-v3 interpolation preserves clean pixel
recall and improves mask precision/FPR. Its native-only fast profile reduces
aggregate stress-case misses from seven to three; the selected 1.0x/1.5x
triage average recovers the remaining shadow, severe-low-resolution, and
simulated-distance cases, reaching 98/98 in all 11 conditions. This increases
clean transformed-background triage FPR from 12.40% to 13.14% and median RTX
3060 Laptop latency from 97.2 ms to 333.8 ms for the finalized profile. These are deterministic
synthetic diagnostics on consumed validation—not proof of real-world nuisance
performance or a substitute for independently reviewed clean vehicles.

A mask-only component filter was selected with a calibration recall guard. A
32-pixel candidate removed more false regions but lost two matched regions
under synthetic sensor noise and was rejected. The finalized 24-pixel filter
preserves all matched regions in clean validation and all 11 synthetic stress
conditions while reducing false regions in every condition (302 to 292 on the
unmodified validation images). Triage remains unfiltered, so small evidence is
still routed to review. This is consumed-development evidence, not an
independent estimate of deployment false positives.

A prior ResNet calibration-selected 256-pixel component filter reduced validation false
regions from 606 to 320 without changing 98/98 permissive case recall, but it
also lost one matched region and still achieved only 12.57% region precision
and 10.24% region recall. It is therefore an optional development diagnostic,
not a production fix or the recall-first default.

Experimental multi-label case heads did not reliably distinguish class
presence from absence on the public data (roughly 0.45–0.67 validation AUROC)
and are not part of the selected candidate. Localized specular highlights are
conservatively routed to manual review, but this safety heuristic is not a
substitute for real clean glare/reflection training and test images.

Post-hoc damage-type calibration and type-head-only refinement were also
rejected. Independent per-type thresholds needed 71%--100% false alerts on
images where the corresponding type was absent to approach the requested
recall. A milder relative type bias preserved binary masks and improved clean
dent recall by one case, but lost one type hit in four synthetic nuisance
slices. An eight-epoch type-head-only run was verified to change only the five
`type_head` tensors (all other 184 checkpoint tensors remained bit-identical),
yet its best checkpoints traded crack/paint recall for dent/scratch recall.

A quarter-resolution RGB detail residual produced a small clean-validation
localization gain but was also rejected. It retained only 98.36% of selected
pixel recall under synthetic underexposure, lost matched regions under dirt
and low resolution, added false regions under glare and shadow, and increased
median latency by 5.95%. The selected model does not include this branch.
Checkpoint interpolations from 10% through 75% either regressed a class or,
at 12.5%, exactly tied all five clean-validation type hit counts. None is part
of the selected profile.

An experimental four-last-layer DINOv2 decoder was then warm-started from the
previous selected checkpoint and trained with the backbone frozen. Its stress-guarded
0.55/144-pixel profile is a binary-localization improvement on consumed clean
validation: pixel precision/recall are 50.22%/73.34%, background FPR is 6.13%,
and region precision/recall are 35.38%/33.41% (150 matched, 274 false, 299
missed). It retains 98/98 cases in all 11 synthetic conditions, improves pixel
recall in every condition, and never loses a matched region relative to the
selected model. However, false regions increase by 10 under severe resolution
loss, precision decreases under blur/low-resolution/distance, and its own
triage heads reduce clean crack hits from 75 to 73. Parameter soups, crack
multipliers, and probability fusion did not remove that regression. The
promoted role-split architecture instead runs the previous final-layer heads
only for triage and the four-layer heads only for segmentation in one shared
backbone pass. It preserves every clean and stress-condition type hit while
retaining the four-layer mask improvement. Branch-selective execution skips
the unused segmentation heads at the 1.5x triage-only scale and is
tensor-identical to exhaustive execution on the checked real image. The
complete path measures 335.0 ms median and 375 MB peak allocated GPU memory.

A native-only role-split diagnostic measures 151.4 ms median (6.61 images/s),
but loses one development case under each of shadow, severe low resolution,
and simulated distance. Lowering its threshold to recover those cases raises
transformed-background FPR. Because the intended use prioritizes false-negative
reduction, this profile is explicitly rejected for selection; the
native-plus-1.5x path remains the frozen development default.

The clean-data pipeline currently has 109 hash-bound Open Images vehicle
candidates staged for human review. Original byte lengths and MD5 values match
the official metadata, but Open Images disclaims per-image license assurance;
therefore every candidate remains rights-unapproved, damage-unreviewed, and
ineligible for training. The queue contributes no release-test evidence and
does not change the selected checkpoint. Qualified per-image rights review,
two independent clean/damage reviews, and fresh first-party calibration/test
data are still required.
