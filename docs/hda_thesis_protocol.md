# HDA thesis protocol v2: fixed source, multiple adaptation seeds

The current CICIDS test set was used to inspect labels and choose the pseudo-label
bands. Existing v2/v4 results, including 0.5861, are development/exploratory results.
Features-only adaptation code does not undo this experiment-design leakage.
Repartitioning already inspected records cannot create an untouched holdout.

## Frozen experiment

The executable settings are in `configs/hda_thesis_protocol.json`. V2 uses hidden
256D marginal MMD. V4 starts from the same-adaptation-seed v2 adapter and adds the average of
Normal and Attack conditional latent 168D MMD, weighted by 1.0. No further tuning
is allowed under this protocol version after final labels are opened.

Both methods use 10 epochs, batch size 256, Adam lr 0.001 and weight decay 0.0001,
one RBF kernel with median bandwidth, a Linear/BN/ReLU adapter without dropout,
and a frozen source/shared tail/classifier. Only natural CICIDS batches update
adapter BN running buffers; conditional batches use adapter BN in eval mode.
V4 uses 64 samples per class/domain, fixed teacher pseudo-labels from adaptation
train quantiles (bottom 2% Normal; 95–98% Attack), and the last training epoch.

Source pretraining is fixed: every run loads `unsw_seed42.pt` and uses the threshold
from `results/baseline/unsw_seed42.json`. Adaptation seeds are 42, 43, 44 for the
current internal/development evaluation. Only the adaptation initialization,
data shuffling, and pool sampling RNG varies. The data split, preprocessing,
source weights, quantile rules, and hyperparameters stay fixed. V4 seed 43 uses
the v2 adaptation-seed-43 teacher, both using source seed 42. No UNSW seed-43/44
pretraining is required. This measures adaptation variability conditional on one
source model; it does not estimate variability of the full pretraining pipeline.
`--seed` remains an alias for `--adaptation-seed`. Both seed fields are saved in
checkpoints and JSON results. Final seeds remain unset because no untouched data
is available. These changes do not launch extra training.

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

### Frozen V4 development reference and purity revision

All V5 development comparisons must use
`configs/hda_v4_development_reference.json`: AP=0.5861, ROC-AUC=0.6828,
F1=0.5375, Recall=0.3821, FPR=0.0081. Oracle F1=0.5960 is diagnostic only.
These are user-supplied rounded results, not a newly recomputed run or a frozen
checkpoint artifact. Do not replace this reference with later V4 results.

The current evaluation code requires revision v2 (v1 remains the historical
manifest for its prior code snapshot):

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.hda \
  --protocol configs/hda_thesis_protocol.json \
  --evaluation-revision configs/hda_evaluation_v2.json \
  --phase development --version v4 --adaptation-seed 42
```

The console purity table uses development labels and development margin quantiles
only. It does not measure actual adaptation-train pool purity, whose labels remain
unused. Equal margins stay together; empty bands print N/A and the maximum is
included in the last band. No purity values feed training or reported metrics.
Results are saved under the `hda_eval_v2_purity` evaluation subdirectory.

### Evaluation-only revision (existing checkpoints)

The original training protocol and checkpoint metadata remain unchanged.
`configs/hda_evaluation_v1.json` binds the updated HDA evaluation entry point to
the original protocol SHA-256. The revision loader verifies every original code
hash except `src/evaluation/hda.py`, whose replacement hash is explicitly pinned
along with the revision loader. Training, model, preprocessing, baseline metrics,
and original protocol-loader code must still match the frozen snapshot.
Without `--evaluation-revision`, the original strict loader remains in effect.

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.hda \
  --protocol configs/hda_thesis_protocol.json \
  --evaluation-revision configs/hda_evaluation_v1.json \
  --phase development --version v4 --adaptation-seed 42
```

This reuses checkpoints carrying the original protocol hash, including the v2
teacher required by v4 diagnostics. Results go into an `hda_eval_v1_dev_oracle`
subdirectory of the usual phase/HDA result directory and record the evaluation
revision hash and code hashes. Oracle values are console-only, development-only,
and never replace source-threshold metrics. The existing `cicids_test` is explicitly
development data in this protocol; no untouched final holdout has been created.
Further evaluation edits require a new reviewed revision manifest. Changes to
training or data roles still require a new training protocol and matching checkpoints.

Retraining uses a new `hda_thesis_v2_fixed_source` checkpoint namespace. Previous
protocol checkpoints are preserved and have a different protocol hash. The
current development roles support an internal evaluation, not an untouched test.

```bash
PYTHONPATH=src .venv/bin/python -m training.hda_v2 --protocol configs/hda_thesis_protocol.json --source-seed 42 --adaptation-seed 43
PYTHONPATH=src .venv/bin/python -m training.hda_v4 --protocol configs/hda_thesis_protocol.json --source-seed 42 --adaptation-seed 43
PYTHONPATH=src .venv/bin/python -m evaluation.hda --protocol configs/hda_thesis_protocol.json --phase development --version v4 --adaptation-seed 43
```

Final evaluation checks the holdout readiness flags and requires a checkpoint
carrying the same protocol hash. The final path is deliberately unset; final
evaluation currently stops before reading any target labels.

Train/evaluate v2 and v4 for the three adaptation seeds with `--phase development`.
Preserve each result before aggregating AP, ROC-AUC, F1,
Recall and FPR as mean ± sample standard deviation. Report margin ranking and
geometry as explanatory post-training diagnostics; do not use them to tune the
architecture after opening the final test. Any subsequent tuning requires a new
protocol and fresh final holdout.
