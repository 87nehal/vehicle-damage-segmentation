# Reliability and release gates

Do not approve a checkpoint from pixel IoU alone. Freeze a claim/vehicle-grouped test set before tuning, then report 95% confidence intervals overall and for every required slice.

Minimum proposed gates for a human-review product:

- damage-case recall at least 97%, with the 95% Wilson lower bound at least 95%;
- per-class damage-case recall at least 90% on at least 30 independent cases per class;
- connected-region recall at least 90% and precision at least 80% across at least 100 annotated damage components;
- at least 100 independently reviewed clean test vehicles, with both the clean-image false-alert point estimate and its 95% Wilson upper bound at most 5%;
- at least 40 independent clean examples for each of glare, dirt, seams/body contours, reflections, and shadows, with both the slice false-alert point estimate and its 95% Wilson upper bound at most 10%;
- no material slice more than 5 percentage points below overall recall;
- calibration drift and review rate measured on a later, geographically separate shadow set;
- latency and memory measured on the actual deployment hardware.

Count a case as detected when a predicted component intersects the assessor mask at the predeclared overlap rule. Aggregate all views sharing a vehicle/claim `group_id` before computing case recall, false-alert rates, slice counts, or confidence intervals; images from one vehicle are not independent trials. Also publish pixel precision/recall, region precision/recall, false-positive components per clean image, mask IoU, per-class confusion, and the manual-review rate. Thresholds must be selected only on the calibration split and then frozen.

The evaluator's development defaults count 8-connected damage components with
at least 16 output pixels and match them one-to-one at IoU at least 0.10. The
minimum area is resolution-dependent: production must predeclare a physically
meaningful area at the fixed capture/output resolution, freeze both the area
and IoU rules before examining the test set, and report results at that exact
operating point. Region metrics are per view; case and false-alert confidence
intervals remain grouped by vehicle/claim.

Low-quality or uncertain images must go to manual review or trigger recapture. Never market image-only inspection as detecting hidden structural, mechanical, or flood damage.

Deployment consumers must use the JSON `decision`, not infer a decision from
an empty output mask. `recapture_required` covers resolution, severe
exposure/glare, and blur failures. `manual_review_required` covers localized
glare, excessive near-threshold area, one-sided triage/segmentation evidence,
and branch disagreement on damage type. Only `damage_detected` and
`no_damage_detected` set `automated_decision_allowed=true`. These routes are
safety guards, not evidence that the development checkpoint meets release
quality gates.

Deterministic synthetic corruptions are useful for regression testing and for
finding sensitivity to glare, lighting, blur, resolution, viewpoint, or scale.
They do not count toward clean coverage, nuisance-slice coverage, confidence
intervals, calibration, or release gates. Those require independently captured
and reviewed deployment-representative images.

Run the fail-closed audit after producing the frozen report. Missing clean or
hard-negative slices fail rather than being interpreted as zero false alerts:

```powershell
vehicle-damage audit-release --report runs/baseline/test-report.json --output runs/baseline/release-audit.json
```
