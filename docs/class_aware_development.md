# Class-aware MMD: development experiments

These experiments reuse canonical prepared data and source-only checkpoints. They do
not modify the frozen canonical configs or replace the original 40 runs. Outputs use
`target_val_epoch0_<config filename stem>` under the existing `mmd/` model/result roots.
They are development extensions, not additional members of the canonical matrix.

## Confidence audit

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
python -m evaluation.proposal_confidence_audit \
  --output results/development/class_aware_confidence_all_seeds.json
```

The audit uses no target labels. It evaluates the fixed source-only model in eval mode
and simulates the first adaptation epoch's batch ordering, source class eligibility,
and target iterator cycling. Counts are sample exposures; a target sample can be
repeated when the target loader cycles. Dropout is disabled in this audit, unlike
training, so eligibility estimates are not an exact replay of adaptation. These
numbers measure coverage, not pseudo-label correctness. Threshold comparison is `>=`.
The command refuses to overwrite an existing report; use a new output path to rerun.

## Training

Three configs separate threshold and alignment-weight effects:

| Config suffix | Confidence | Lambda |
| --- | --- | --- |
| `confidence90` | 0.90 | 0.01 |
| `lambda001` | 0.80 | 0.001 |
| `confidence90_lambda001` | 0.90 | 0.001 |

Start with UNSW → CICIDS. The reverse direction has very few accepted pseudo-Attack
samples and needs investigation before raising its confidence threshold.

Pilot run (one seed):

```bash
python -m training.proposal_mmd \
  --direction unsw_to_cicids --seed 42 \
  --config configs/proposal_class_aware_dev_confidence90_lambda001.json
```

Full ablation (3 configs × 5 seeds, UNSW → CICIDS only), as an alternative to the pilot. If the pilot already exists, omit that config/seed pair; the trainer will not overwrite it:

```bash
for variant in confidence90 lambda001 confidence90_lambda001; do
  for seed in 42 43 44 45 46; do
    python -m training.proposal_mmd \
      --direction unsw_to_cicids --seed "$seed" \
      --config "configs/proposal_class_aware_dev_${variant}.json" || break 2
  done
done
```

No source-only retraining is needed. The trainer refuses existing outputs and remains
blocked after a development lock; do not remove a lock to bypass this restriction.
Source validation still selects checkpoint/threshold. Target validation AP is for
exploratory development comparison; target test remains held out. Compare all five
paired seeds and alignment coverage before deciding whether a change helps.

## Class separation extension

Five completed UNSW → CICIDS runs for `confidence90_lambda001` have target-validation
AP 0.146695 ± 0.021582 (sample std), versus source-only 0.162569 ± 0.054400.
Mean paired AP change is -0.015874. The candidate has not established an improvement.

The optional `class_separation` config adds a centroid margin loss to the existing
MLP + class-aware MMD. For each domain, form the two class centroids and normalize
each centroid to unit length. Penalize squared distances below 0.5 for four
opposite-class pairs: source Normal/Attack, target Normal/Attack, source Normal/
target Attack and source Attack/target Normal. Average their hinge penalties.

`L = source_CE + 0.001 * class_MMD + 0.001 * separation_loss`.

Target grouping uses detached pseudo-labels with confidence >= 0.9. Both alignment
and separation require at least two samples per class in each domain. Missing-class
batches skip these terms. No target ground-truth labels are read during training.
Centroid separation is a soft geometric constraint, not a guarantee that all samples
are separated or pseudo-labels are correct. Exact centroid collapse can also have
zero gradient; source CE remains necessary. The margin/weight are exploratory
settings and have not been validated as optimal. Source-only architecture and
checkpoint-selection rules remain unchanged.

Run the five-seed extension (reuse source-only checkpoints):

```bash
for seed in 42 43 44 45 46; do
  python -m training.proposal_mmd \
    --direction unsw_to_cicids --seed "$seed" \
    --config configs/proposal_class_aware_dev_confidence90_lambda001_separation.json \
    || break
done
```

After all five runs finish, print means, sample std and paired changes:

```bash
python -m evaluation.proposal_development_summary \
  --direction unsw_to_cicids \
  --config configs/proposal_class_aware_dev_confidence90_lambda001_separation.json
```

Ablate against the completed version without separation using the same five seeds.
Monitor AP, FPR, separation loss and eligible-batch counts. This is exploratory
validation development; do not select hyperparameters using target test results.
