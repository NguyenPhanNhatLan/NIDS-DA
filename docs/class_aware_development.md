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

## Five-check validation audit

```bash
python -m evaluation.proposal_alignment_audit \
  --direction unsw_to_cicids \
  --config configs/proposal_class_aware_dev_confidence90_lambda001_separation.json \
  --output results/development/class_aware_separation_audit_new
```

This command does not retrain. It reads source/target validation only and uses target
labels strictly for post-hoc diagnostics. Default pseudo-label evaluation uses a fixed,
uniform sample of up to 50,000 target-validation flows at natural class prevalence.
kNN uses balanced subsets of up to 500 flows per class/domain from those samples,
with k=15. MMD uses 10 paired subsamples of 256/domain with a shared baseline
bandwidth before/after. These sampling settings are recorded in the report/command
and are not full-population metrics. Full-population AP/ROC-AUC are taken from the
saved development results. Partial seed matrices fail rather than silently averaging.

| Output | Check | Interpretation |
| --- | --- | --- |
| `per_seed_metrics.csv` | Paired AP/ROC-AUC | Positive delta means improvement for that seed. |
| `training_contribution.csv` | Separation loss and eligibility per epoch | Positive weighted loss shows contribution to the objective; eligibility alone does not imply a positive separation penalty. |
| `pseudo_label_quality.csv` | Confidence 0.8/0.9/0.95, before/after | `attack_precision` = correct accepted Attack / all accepted Attack. `attack_recall` = correct accepted Attack / all true Attack in the sample. Undefined precision is blank. |
| `latent_knn.csv` | Within-domain and cross-domain class mixing | Smaller mixing is better; examine each class, not only pooled results. |
| `paired_mmd.csv` | Marginal and class-conditional discrepancy | Negative delta means MMD decreased; check Attack kNN/precision alongside it. True class labels here are diagnostic only. |
| `summary.json` | Seed improvement counts and paired 95% Student-t CI | CI containing zero does not establish a consistent positive gain across seeds. |
| `manifest.json` | Provenance and limitations | Fixed samples, checkpoint hashes, data role and diagnostic assumptions. |

The first verified report is `results/development/class_aware_separation_audit/`.
Choose a fresh directory when rerunning; existing reports are not overwritten.

## MMD ablation, separation strength and class-specific confidence

New configs keep margin=0.5 and reuse the same MLP/source-only checkpoints:

| Filename suffix after `proposal_class_aware_ablation_` | MMD weight | Separation weight | Normal / Attack confidence |
| --- | --- | --- | --- |
| `sep_only.json` | 0 | 0.001 | 0.90 / 0.90 |
| `sep003.json` | 0.001 | 0.003 | 0.90 / 0.90 |
| `sep010.json` | 0.001 | 0.010 | 0.90 / 0.90 |
| `tau90_95.json` | 0.001 | 0.001 | 0.90 / 0.95 |

Compare each against the existing `confidence90_lambda001_separation` reference,
not against each other alone. These are unvalidated development settings. In
particular, a higher Attack threshold may reduce eligibility without improving
precision. This implementation uses fixed class-specific thresholds; it does not
implement adaptive FlexMatch scheduling or force balanced pseudo-label counts.

Per-class thresholds are configured as:

```json
"target_pseudo_label_confidence_by_class": {"0": 0.90, "1": 0.95}
```

This optional map takes precedence over `target_pseudo_label_confidence`. The same
selection helper is used by initial pseudo-label audit, batch eligibility, MMD,
and separation. Confidence always uses `>=`. Missing either class still skips the
whole class-aware term. With MMD disabled, `aligned_batches` is zero while
`class_separation.eligible_batches` and `positive_loss_batches` remain meaningful.

New configs enable `diagnostics.gradient_every_steps=200`. On sampled batches,
the trainer computes the global parameter-gradient L2 norm of each **weighted**
term via `autograd.grad` on the same forward graph before the normal backward pass.
It retains the graph and does not accumulate these probe gradients into `.grad`.
History stores epoch means of sampled norms and the sample count. These are
parameter-gradient magnitudes, not gradient cosines, optimizer update magnitudes,
or a guarantee of benefit. The diagnostics add computation and graph lifetime.
Set the interval to 0 in a new config to disable them; do not edit a config after
training artifacts have been generated from it.

History also records epoch-average weighted CE/MMD/separation contributions.
The post-hoc audit exports them and gradient norms to `training_contribution.csv`.
Previously trained checkpoints do not have gradient probes; those columns remain
blank for old runs. The pseudo-label quality CSV includes the configured asymmetric
selection rule in addition to the shared-threshold sweep.

Run all four ablations over five seeds (20 adaptation runs):

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

for variant in sep_only sep003 sep010 tau90_95; do
  for seed in 42 43 44 45 46; do
    python -m training.proposal_mmd \
      --direction unsw_to_cicids --seed "$seed" \
      --config "configs/proposal_class_aware_ablation_${variant}.json" || break 2
  done
done
```

After training, print each five-seed summary:

```bash
for variant in sep_only sep003 sep010 tau90_95; do
  python -m evaluation.proposal_development_summary \
    --direction unsw_to_cicids \
    --config "configs/proposal_class_aware_ablation_${variant}.json" || break
done
```

Run all five post-hoc diagnostics per variant:

```bash
for variant in sep_only sep003 sep010 tau90_95; do
  python -m evaluation.proposal_alignment_audit \
    --direction unsw_to_cicids \
    --config "configs/proposal_class_aware_ablation_${variant}.json" \
    --output "results/development/class_aware_ablation_${variant}_audit" || break
done
```

Commands refuse existing outputs. If interrupted, first inspect incomplete pairs,
then run only missing config/seed pairs explicitly. Do not overwrite or delete the
completed reference. All comparisons remain exploratory target-validation work;
keep target test held out until the protocol is fixed.

## Retired experiment: EMA teacher and source-anchor selective alignment

The selective single/multi-kernel branch was removed at the user's request after
the supplied UNSW → CICIDS seed-42 log reported target-validation AP 0.171850 vs
source-only 0.189605 (delta -0.017755). This one run does not establish a general
failure of the method; it records the reason for retiring this implementation.

The active trainer retains the earlier class-aware MMD/separation variants,
including `configs/proposal_class_aware_ablation_sep010.json`. It no longer runs
EMA teachers, source-anchor filtering, weighted selective MMD or `skip_class`.
Per-class confidence thresholds and gradient diagnostics remain available.

Completed model/result artifacts were preserved. Retired config snapshots and the
supplied log are archived in `results/development/retired_selective/`; they are not
runnable configs for the current trainer. Configs with `selective_alignment` are
explicitly rejected to prevent silently running a different method.

## Standalone class-aware MK-MMD with separation

Active replacement experiment, without EMA teachers or source anchors:

| Config | Class kernels | Adaptation learning rate |
| --- | --- | --- |
| `proposal_class_aware_mkmmd_sep010.json` | RBF scales 0.5/1/2 | 0.001 |
| `proposal_class_aware_mkmmd_sep010_lr0003.json` | RBF scales 0.5/1/2 | 0.0003 |

The first config changes only the kernel/scales relative to `sep010`. The second
changes only learning rate relative to the first, testing a smaller adaptation
update from the pretrained source-only checkpoint. Lower learning rate is a
hypothesis for more stable fine-tuning, not a demonstrated AP improvement.

Architecture, source-only checkpoint initialization, confidence >=0.90, λMMD=0.001,
λSep=0.010, separation margin=0.5, batch size=256 and source-validation selection
are unchanged. Both classes still need >=2 source and accepted target samples;
otherwise the entire class-aware MMD/separation term is skipped. Per-class sample
truncation matches the previous estimator. No target labels are read in training.

For each eligible class, calculate the mean of three RBF MMD losses using a common
per-class median-heuristic base bandwidth multiplied by 0.5, 1 or 2. Then average
the two class losses. Kernels are averaged, not summed, to preserve the coefficient
scale. The `method` field remains `class_aware_mmd`; kernel metadata/scales and unique
config filenames distinguish these development variants from canonical runs.

Run both configs, five seeds each:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
for variant in sep010 sep010_lr0003; do
  for seed in 42 43 44 45 46; do
    python -m training.proposal_mmd \
      --direction unsw_to_cicids --seed "$seed" \
      --config "configs/proposal_class_aware_mkmmd_${variant}.json" || break 2
  done
done
```

Summarize after training:

```bash
for variant in sep010 sep010_lr0003; do
  python -m evaluation.proposal_development_summary \
    --direction unsw_to_cicids \
    --config "configs/proposal_class_aware_mkmmd_${variant}.json" || break
done
```

Optional post-hoc diagnostics:

```bash
for variant in sep010 sep010_lr0003; do
  python -m evaluation.proposal_alignment_audit \
    --direction unsw_to_cicids \
    --config "configs/proposal_class_aware_mkmmd_${variant}.json" \
    --output "results/development/class_aware_mkmmd_${variant}_audit" || break
done
```

These reuse source-only checkpoints but require new adaptation runs. Existing
canonical/sep010 artifacts are not overwritten. Compare paired seeds against
sep010 AP=0.203123 ±0.012655; inspect pseudo-Attack precision, FPR and mixing as
well as AP. Multi-kernel alignment cannot by itself correct wrong pseudo-labels.
