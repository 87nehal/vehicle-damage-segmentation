# Annotation protocol

Use indexed, single-channel PNG masks with the ids in `configs/base.yaml`. A pixel is damage only when there is visible physical or coating change. Reflections, glare, dirt, water, panel seams, design creases, shadows, decals, and normal part boundaries are background.

Annotate three aligned masks when possible:

1. `mask`: damage class. Use `crack_or_breakage` for fractured glass/plastic or a visibly broken lamp; use `deformation_or_detachment` for displaced, torn, smashed, or missing exterior pieces.
2. `exterior_mask`: inspectable vehicle exterior. Exclude sky, road, people, open interior, and background vehicles.
3. `hard_negative_mask`: confusing but non-damaged regions such as glare, grime, water spots, deep panel gaps, styling lines, strong reflections, shadows, stickers, and prior-repair texture.

Every damage image should be double-reviewed. Resolve disagreement with an automotive assessor. Keep an explicit empty damage mask for clean vehicles and hard negatives—do not omit them.

Model-assisted masks are proposals, not annotations. For CVAT, import the
repository export using **Segmentation Mask 1.1**, correct all boundaries and
class assignments, and require a second person to review the completed masks.
The annotator and reviewer IDs must differ. Import corrected masks with
`scripts/import_cvat_annotation_batch.py`; it rejects empty damage masks,
unknown colors/class IDs, changed source images, and dimension mismatches.
Clean decisions must instead pass the independent classification-review and
adjudication workflow.

Group all photos of the same vehicle, incident, burst, or near-duplicate source under the same `group_id`. A group may occur in only one split. Before splitting, remove exact and perceptual duplicates.

Required capture slices include daylight/night, indoor/outdoor, clean/dirty/wet, light/dark paint, metallic/solid paint, strong glare, close-up/medium/full-car, front/rear/side/oblique angle, phone types, blur/compression, and each damage class and severity.
