# Frozen V2 with residual target correction

This experiment preserves the UNSW seed-42 BaselineMLP and the same-adaptation-seed
V2 target adapter. Only `ResidualTargetCorrection` is trained:

`h0 = V2_adapter(x)`

`h = h0 + alpha * correction_block(h0)`

The exact correction sequence is `LayerNorm(256) → Linear(256,128) → GELU →
Dropout(0.1) → Linear(128,256)`. The corrected hidden vector passes through the
frozen source `fc2 → BN2 → ReLU` and classifier. Latent width remains 168.
All source layers and V2 adapter parameters and BN buffers stay frozen in eval
mode. Only the correction is in train mode.

`alpha` is a trainable scalar initialized to one. The final residual Linear layer
has zero-initialized weight and bias, so initial predictions still match V2 exactly.
On the first backward pass, the data gradient reaches the final Linear layer;
alpha and earlier correction layers initially have zero data gradients. Earlier
layers can learn once the final layer moves away from zero. Alpha is unconstrained.

## First experiment

```bash
PYTHONPATH=src .venv/bin/python -u -m training.hda_residual --mode base --seed 42
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_residual --mode base --seed 42
```

Base loss: `hidden MMD + 0.05 Normal MMD + 0.05 Attack MMD`, equivalent to
`hidden MMD + 0.10 * conditional MMD` with the V5a class-average definition.
No new source pretraining or target warmup is required; the original V2 checkpoint
is the starting point. V6a/b/c remain separate experiments.

## Optional follow-up

V5b is a separate HDAV1Model adapter experiment, not this residual architecture.
Use `training.hda_v5b` and `evaluation.hda_v5b` as documented in `docs/hda_v5b.md`.

After comparing the base run with V5a, run the separate rank variant:

```bash
PYTHONPATH=src .venv/bin/python -u -m training.hda_residual --mode rank --seed 42
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_residual --mode rank --seed 42
```

Rank loss: `hidden MMD + 0.05 Normal MMD + 0.02 Attack MMD + 0.01 rank loss`.
Teacher and student margins use the same natural target batch. Teacher margins
are detached. Rank loss is `1 - Pearson correlation`; a constant teacher margin
has undefined correlation, so that batch contributes zero rank loss. Student
norm is clamped for numerical stability. This measures linear association of
margins; it does not guarantee unchanged ROC-AUC. This variant changes both the
Attack weight and the rank term, so its difference from base is not a rank-only
ablation. Both modes start from V2 with alpha one and a zero residual output,
not from each other's output.

## Data and outputs

The original protocol supplies 10 epochs, batch size 256, class batch size 64,
Adam lr 0.001 and weight decay 0.0001. Single-RBF MMD and pseudo pools are reused:
Normal `margin <= q02`, Attack `q95 <= margin < q98`, computed on adaptation-train
features by the frozen V2 teacher. No target labels are loaded during training.
The batch loader uses the optimized V6 Arrow conversion.

Evaluation is development-only, on the split explicitly declared by the original
protocol. Source-validation threshold is copied from UNSW baseline results into
the training checkpoint and reused during evaluation. Oracle threshold/F1 print
only as diagnostics. Main results include V2 teacher metrics and deltas against
the fixed V4 reference; matching V5a lambda-0.10 results are compared when present.
No ROC-AUC target is guaranteed.

Config: `configs/hda_residual.json`.
Checkpoints: `models/hda_residual/{base,rank}_seed42.pt`.
Results: `results/hda_residual/development/{base,rank}_seed42.json`.
Checkpoints store the correction and hashes of its frozen dependencies. Evaluation
rejects changed source/teacher files or incompatible config/code. Existing
checkpoints are not overwritten. Use seed 43 or 44 to select the matching V2 teacher;
the source remains seed 42.
