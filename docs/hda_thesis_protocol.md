# HDA thesis protocol v1

The current CICIDS test set was used to inspect labels and choose the pseudo-label
bands. Existing v2/v4 results, including 0.5861, are development/exploratory results.
Features-only adaptation code does not undo this experiment-design leakage.
Repartitioning already inspected records cannot create an untouched holdout.

## Frozen experiment

The executable settings are in `configs/hda_thesis_protocol.json`. V2 uses hidden
256D marginal MMD. V4 starts from the same-seed v2 adapter and adds the average of
Normal and Attack conditional latent 168D MMD, weighted by 1.0. No further tuning
is allowed under this protocol version after final labels are opened.

Both methods use 10 epochs, batch size 256, Adam lr 0.001 and weight decay 0.0001,
one RBF kernel with median bandwidth, a Linear/BN/ReLU adapter without dropout,
and a frozen source/shared tail/classifier. Only natural CICIDS batches update
adapter BN running buffers; conditional batches use adapter BN in eval mode.
V4 uses 64 samples per class/domain, fixed teacher pseudo-labels from adaptation
train quantiles (bottom 2% Normal; 95–98% Attack), and the last training epoch.

Seed 42 remains the development seed. Planned final seeds are 42, 43, 44. This
does not launch extra training. All final seeds use the identical data split,
preprocessing, quantile rules, and hyperparameters; only the training RNG changes.
Each seed needs its own UNSW checkpoint/validation threshold and its own v2 teacher.

## Data roles

- Adaptation train: current `cicids_train`, features only for v2/v4 and teacher
  quantiles. This path remains provisional until the final holdout is identified.
- Development: current inspected `cicids_test`, labels permitted for diagnostics.
  `cicids_val` is not automatically untouched: target supervised training uses its
  labels for checkpoint selection.
- Final test: pending an independently held-out or new data source whose labels
  and outcomes did not inform development. Merely changing the split seed is
  insufficient. If no untouched data remains, report a prospective/repeated-split
  evaluation honestly instead of calling it untouched.

Before final evaluation, record holdout provenance, verify feature-duplicate and
collection/flow grouping separation from train/development, and transform it with
the frozen train-fitted preprocessing. Do not use final features for scaling fit,
MMD, pseudo-label quantiles, checkpoint selection, threshold tuning, or diagnostics
during development. If preparation changes adaptation train, refit preprocessing
on that train only and retrain both v2 and v4 under a new, finalized protocol hash.

## Commands

Retraining uses a separate checkpoint namespace; legacy checkpoints are not final
protocol checkpoints. Do not run until the data roles have been finalized.

```bash
PYTHONPATH=src .venv/bin/python -m training.hda_v2 --protocol configs/hda_thesis_protocol.json --seed 42
PYTHONPATH=src .venv/bin/python -m training.hda_v4 --protocol configs/hda_thesis_protocol.json --seed 42
PYTHONPATH=src .venv/bin/python -m evaluation.hda --protocol configs/hda_thesis_protocol.json --phase development --version v4 --seed 42
```

Final evaluation checks the holdout readiness flags and requires a checkpoint
carrying the same protocol hash. The final path is deliberately unset; final
evaluation currently stops before reading any target labels.

After protocol/data freeze, train/evaluate v2 and v4 for all three planned seeds
with `--phase final`. Preserve each result before aggregating AP, ROC-AUC, F1,
Recall and FPR as mean ± sample standard deviation. Report margin ranking and
geometry as explanatory post-training diagnostics; do not use them to tune the
architecture after opening the final test. Any subsequent tuning requires a new
protocol and fresh final holdout.
