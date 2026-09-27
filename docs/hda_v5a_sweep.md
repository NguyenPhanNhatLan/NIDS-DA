# V5a: conditional MMD weight development sweep

Run from the repository root:

```bash
PYTHONPATH=src .venv/bin/python -u -m training.hda_v5a
```

The pinned manifest is `configs/hda_v5a_sweep.json`. This is a new development
experiment, not an evaluation-only revision of the old training run. It uses
adaptation seed 42 and source seed 42. Each of 0.10, 0.25, 0.50, and 1.00 calls the
unchanged `train_hda_v4` function for 10 epochs, starting from the identical V2
teacher checkpoint with the same seed reset and pool construction order.
The sole training hyperparameter change is `lambda_conditional`:

`loss = hidden_mmd + lambda_conditional * (normal_mmd + attack_mmd) / 2`

Architecture, single-RBF MMD, optimizer, batch sizes, pseudo-pool quantile rules,
and BN handling are inherited from V4. No CE loss is added. Target training labels
are not loaded. Development labels are read after training for source-threshold
metrics and console-only oracle diagnostics.

Checkpoints, per-weight JSON, and a final `comparison.csv` go to
`results/thesis/hda_v5a_conditional_weight_seed42/`. The original V4 checkpoint and
results are preserved. Lambda 1.00 is a reproducibility control run; it does not
replace the fixed reference AP=0.5861, ROC-AUC=0.6828, F1=0.5375, Recall=0.3821,
FPR=0.0081 in `configs/hda_v4_development_reference.json`. Each JSON reports metric
deltas against that exact rounded reference. Oracle F1=0.5960 remains diagnostic.

The runner verifies the original training code through the pinned evaluation
revision and additionally pins the new runner and supporting code. It records the
parent training protocol, sweep, teacher, source, threshold, and reference hashes.
It rejects reusing an output directory with different provenance and skips
completed per-weight results on restart. An interrupted weight without a result
JSON trains again from the teacher. This sweep supports development selection
only; it does not claim untouched final-test performance or multi-seed robustness.
