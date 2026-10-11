# Funnel16 ablation

Run from the project root with the virtualenv activated and `PYTHONPATH="$PWD/src"`.
No changes to the existing canonical model suite or 5D Funnel artifacts are required.
Training requires an open development revision; do not remove an existing development lock.

The encoder is input → 64 → 32 → 16. The classifier consumes raw 16D features.
Alignment consumes L2-normalized features. Source SupCon also computes cosine similarity
on normalized features, without changing what the classifier receives.

Stages:

- `source-only`: weighted source CE.
- `source-supcon`: same random initialization and settings, source CE + SupCon.
- `ce-control`: continue the selected source SupCon checkpoint with CE + SupCon, λ = 0.
- `class-mkmmd`: same continuation, plus class-aware three-scale MK-MMD and linear λ warmup.
- `evaluate`: replay all four checkpoints on source validation and then target validation.

Both continuation stages freeze BatchNorm running statistics. Alignment requires at least
two source and two confident target samples for each class; otherwise the whole alignment
term is skipped. No target labels are read during training. Pseudo-label stability is
measured on a fixed, ordered target-training probe at each epoch (including the initial
checkpoint as the first comparison). This is a consistency diagnostic, not proof of accuracy.

Config: `configs/proposal_funnel16.json`. Alpha = 0.1, temperature = 0.1,
confidence = 0.9, maximum λ = 0.01, warmup = 10 epochs are experimental defaults.
Source-AP early stopping can stop before the full λ is reached; inspect history.

```bash
for direction in unsw_to_cicids cicids_to_unsw; do
  for seed in 42 43 44 45 46; do
    for stage in source-only source-supcon ce-control class-mkmmd evaluate; do
      python -m experiments.proposal_funnel16 \
        --direction "$direction" --seed "$seed" --stage "$stage" || break 3
    done
  done
done
```

Artifacts are under `models/<revision>/funnel_16d` and `results/<revision>/funnel_16d`.
Results are not automatically included in the original suite's aggregate/export/final-test stages.
For evaluation replay, use `--evaluation-name evaluation_v2` to create a new output
directory rather than overwriting an earlier report.

Evaluation includes AP, ROC-AUC, confusion counts, source-selected F1 threshold,
score quantiles/PR curves, and thresholds meeting source-validation FPR limits 1%/5%.
Target FPR may differ at those thresholds. Labeled target recall at fixed target FPR
is separately marked as oracle diagnostic only, never used to select a deployed threshold.
Compare paired target AP changes between each ablation, especially alignment minus
continued control. These diagnostics do not evaluate target test or compute latent MMD/AUC.
