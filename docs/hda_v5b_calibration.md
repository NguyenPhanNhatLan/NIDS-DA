# Frozen V5b asymmetric calibration

The selected seed-42 V5b asymmetric checkpoint is pinned in
`configs/hda_v5b_calibration_frozen.json`, along with its config hash and historical
development reference ROC-AUC=0.8098, AP=0.5851. Training code and weights are not
changed by calibration.

## Fit and freeze

```bash
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5b_calibration --stage fit
```

This command reads labels only from UNSW validation. It computes frozen source
Normal/Attack median logit margins and chooses the highest-recall operating point
whose source-validation FPR is at most 0.01. Ties prefer lower FPR, then higher
threshold. Selection sorts scores once and handles tied scores as groups; an
all-negative operating point is included when necessary. Predictions use `>=`.

The original frozen V2 teacher selects CICIDS adaptation-train pseudo pools with
the unchanged rules `margin <= q02` and `q95 <= margin < q98`. Target labels are
not decoded or used. Frozen V5b scores those selected features. Only two affine
parameters are fitted from medians:

`a = (source_attack - source_normal) / (target_attack - target_normal)`

`b = source_normal - a * target_normal`

The fit requires ordered, non-collapsed anchors and finite `a > 0`. Invalid
anchors stop the run; the code never reverses ranking or uses development labels
to find a fallback. Medians use `torch.median` (lower middle value for even counts).

The artifact `results/hda_v5b/asymmetric/calibration/seed42.json` freezes a, b,
anchors, source operating threshold, policy, model dependencies, code hashes and
calibration-data hashes. The command refuses to overwrite an existing artifact.
Model parameters have gradients disabled and remain in eval mode throughout.
The target development loader is never created during fitting.

## Report after freezing

```bash
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5b_calibration --stage report
```

Report verifies the artifact digest, code, reference, model dependencies and
calibration-data hashes before reading development labels. It then applies the
fixed transform `a * margin + b` and the saved source-margin threshold. It does
not fit parameters, select a target threshold, or compute an oracle threshold.
F1, Recall, Precision, FPR, ROC-AUC and PR-AUC (average precision) are saved to
`results/hda_v5b/asymmetric/development/v5b_calibrated_seed42.json`. Existing reports
are not overwritten.

The source FPR constraint does not guarantee the same target FPR. Positive affine
calibration preserves margin ordering, so it changes the operating point, not the
underlying ranking. Metrics use margins to avoid sigmoid saturation; raw-margin
ROC-AUC/AP are included for comparison. Tiny differences from historical metrics
computed from float32 softmax scores can arise from score ties. The helper
`calibrated_probability` provides `sigmoid(a * margin + b)` for inference.

This procedure does not use target labels to fit calibration. The target split
has already informed earlier development/model selection, however, so results
remain development performance, not untouched final-test performance. Do not
retune a, b or the threshold policy after opening this report.
