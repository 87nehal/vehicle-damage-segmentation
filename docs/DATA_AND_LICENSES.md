# Commercial-use provenance policy

This repository fails closed: every manifest row must state `commercial_use: true` and a non-empty `license_id`. Preserve the agreement, source URL, acquisition date, consent/release, attribution requirements, and a cryptographic inventory outside the training manifest. Legal counsel should approve the inventory before release.

Current source decisions:

| Source | Default status | Reason |
|---|---|---|
| First-party/company-captured data with releases | Allowed after internal approval | Best match to deployment and cleanest provenance |
| Licensed repair/insurance partner data | Allowed after contract review | Contract must permit ML training, derivatives, and deployment |
| Humans in the Loop Car Parts and Car Damages | Allowed as bootstrap, CC0 1.0 | Publisher explicitly dedicates the 1,812-image polygon dataset to the public domain; retain provenance |
| CarDD | Blocked | Official terms say non-commercial research/education; commercial testing or use needs prior authorization |
| VehiDE | Blocked | The primary publication restricts the Flickr/Shutterstock-derived images to non-commercial research and education; a conflicting Kaggle Apache-2.0 label is not sufficient |
| CDDM | Blocked pending signed terms | The official repository requires a separate dataset licensing form; obtain explicit commercial training, derivative-model, and deployment rights before use |
| Grec et al. 2026 vehicle-damage dataset | Blocked pending license | The official repository describes 2,290 annotated logistics images but publishes no explicit dataset license |
| Kaggle Undamaged Vehicle Image Dataset | Blocked | The data card says images were assembled from automotive resale websites; an uploader-applied Apache-2.0 label does not establish rights to those photographs |
| CAD 100K | Blocked pending release and license review | The supplied paper describes data but does not provide commercial-use terms or a verifiable release |
| Open Images V7 vehicle candidates | Blocked from training pending per-image verification | Google lists images as CC BY 2.0 and annotations as CC BY 4.0, but explicitly disclaims assurance of each image's license status; preserve author/landing-page metadata and verify every image before commercial approval |
| Supplied 2023 web-collected classification paper dataset | Blocked | The paper says its images were collected from the web and does not provide dataset release or commercial-use terms |
| Web-scraped images | Blocked | Public visibility is not a commercial training license |
| Synthetic images | Allowed only when generator, input, and output rights are recorded | Keep synthetic data out of the final real-only test set |

The additional public-source decisions above were rechecked on 2026-09-19:

- VehiDE primary publication and stated restriction:
  https://www.tandfonline.com/doi/full/10.1080/24751839.2024.2367387
- CDDM official repository and licensing-form requirement:
  https://github.com/SCUT-CCNL/CDDM
- Grec et al. official dataset repository:
  https://github.com/marcellogrec/vehicle-damage-detection
- Undamaged Vehicle Image Dataset source description:
  https://www.kaggle.com/datasets/garystafford/undamaged-vehicle-image-dataset
- Open Images V7 official description and license warning:
  https://storage.googleapis.com/openimages/web/factsfigures_v7.html
- Open Images V7 official download and metadata format:
  https://storage.googleapis.com/openimages/web/download_v7.html

Open Images candidates can be staged without silently approving them:

```bash
python scripts/download_open_images_originals.py \
  --metadata path/to/train-images-boxable-with-rotation.csv \
  --boxes path/to/train-annotations-bbox.csv \
  --class-descriptions path/to/class-descriptions-boxable.csv \
  --output-dir data/open_images_originals \
  --limit 200 --max-attempts 800

python scripts/prepare_open_images_candidates.py \
  --metadata path/to/train-images-boxable-with-rotation.csv \
  --boxes path/to/train-annotations-bbox.csv \
  --class-descriptions path/to/class-descriptions-boxable.csv \
  --images-dir path/to/original-images \
  --output-dir review_batches/open_images_license_001 \
  --limit 200
```

This offline intake accepts only exact CC BY 2.0 metadata rows with complete
attribution, an exact original byte-length and MD5 match, and `xclick` Car boxes
that are neither group annotations, depictions, nor interior views. The bounded
downloader uses only `OriginalURL`; thumbnails and transformed dataset copies
cannot substitute for the publisher-recorded original. The intake writes a quarantined candidate inventory,
attribution sheet, provenance hashes, and a per-image license review page. It
deliberately does **not** write a training manifest: every record remains
`training_eligible: false`, `commercial_use_approved: false`, and damage
unreviewed. A qualified rights reviewer must visit each original landing page,
complete `LICENSE_REVIEW_TEMPLATE.csv`, and explicitly confirm commercial ML
training rights. Approved rows can then enter the separate double visual review:

```bash
python scripts/create_open_images_damage_review_batch.py \
  --candidates review_batches/open_images_license_001/CANDIDATES.jsonl \
  --license-review completed-license-review.csv \
  --output-dir review_batches/open_images_damage_001

# Two automotive reviewers independently complete review.html.
python scripts/import_open_images_clean_reviews.py \
  --candidates review_batches/open_images_license_001/CANDIDATES.jsonl \
  --license-review completed-license-review.csv \
  --review-a vehicle-review-a.csv \
  --review-b vehicle-review-b.csv \
  --output-manifest data/open_images_reviewed_clean/manifest.jsonl
```

Only exact clean agreements become all-zero damage masks. Damaged and unusable
rows are excluded, and any disagreement requires a third, distinct automotive
assessor via `--adjudication`. Candidate IDs, image SHA-256 values, attribution
fields, license evidence, reviewer IDs, and all input-file hashes are verified
and retained in the output audit sidecar. Non-zero official rotation metadata
is honored in both review pages; an approved training row uses a lossless
orientation-normalized PNG whose derivative SHA-256 and transformation are
recorded while the original remains untouched. This workflow records reviewer
attestations; it is not a substitute for counsel or for deployment-specific
first-party test data.

An initial real review batch is staged at
`review_batches/open_images_license_validation_001/license-review.html`. It has
109 original-source files (273,967,770 bytes total) that passed the official
size and MD5 checks. Expansion paused when the source host began returning
HTTP 429; rate-limited URLs were not mislabeled as missing. These images remain
unapproved, damage-unreviewed, and ineligible for training. Source URLs, hashes,
and lineage are recorded in
`data/open_images_v7_source/SOURCE_PROVENANCE.json`. Although Open Images calls
the source pool `validation`, this project treats it only as a candidate pool;
none of these images is assigned to this model's validation or release test.
The review page saves progress locally, exports an incomplete draft CSV for
inspection, and can export and restore a JSON browser-state backup bound to the
candidate-inventory SHA-256. Cross-batch state restores are rejected. Draft
CSVs remain intentionally invalid input to the promotion importer; only the
separately validated completed export can advance.

The immutable pre-review capture-quality audit covers all 109 originals: 24
pass the automatic quality screen, 76 require manual quality review (primarily
localized specular highlights), and 9 trigger recapture rules; no byte-identical
duplicates were found. The JSON and CSV reports are stored beside the review
page and hash-pinned in `SOURCE_PROVENANCE.json`. Quality screening does not
approve image rights or establish a clean damage label.
A bounded retry on 2026-09-19 again received three consecutive HTTP 429
responses and stopped through the downloader circuit breaker. Do not repeatedly
probe the host; resume expansion only after an operator-selected cooldown or by
using an officially supported bulk-download route.

PyTorch and TorchVision source code use permissive terms, but their own documentation warns that individual pretrained weights may inherit dataset/model terms. Therefore `pretrained_backbone` is false by default. Turning it on requires a separate checkpoint and upstream-data review; it is not a legal conclusion.

The opt-in DINOv2 pilot is limited to Meta's standard
`vit_small_patch14_dinov2.lvd142m` weights. Meta's official DINOv2 README and
model card identify the standard code and model weights as Apache License 2.0:

- https://github.com/facebookresearch/dinov2#license
- https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
- https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m

Cached snapshot `4610ca143709d58a633b6397a74412c2c3842454`, acquired
2026-09-19, contains an 88,240,510-byte `model.safetensors` file with SHA-256
`04D27F3400D059FC0CFD7D17DD1909A75BF3EA8FB3EEB48B97CB99E57EE20081`.
The configuration records `commercial_use: true` and the Apache-2.0 provenance
identifier, but final distribution still requires counsel to review notices,
patents, and upstream training-data risk. DINOv2 XRay and Cell-DINO variants
use separate noncommercial terms and are explicitly rejected by the model
factory.

The HITL archive's part-annotation and damage-annotation directories reuse
exact image files. In the prepared copy, 441 binaries occur in both subsets.
Never concatenate the manifests naively: use `--merge-auxiliary` so exterior
labels are merged only into training rows and auxiliary copies of held-out
images are discarded. Part-annotated images are not exhaustively reviewed for
damage and must keep `damage_supervised: false`.
