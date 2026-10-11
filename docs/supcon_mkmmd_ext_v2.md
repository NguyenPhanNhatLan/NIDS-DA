# SupCon + MK-MMD extension v2: full task audit

The initial extension v1 did not implement the separate 64D projection, skip single-class
batches, configurable MMD warm-up, complete epoch logging or latent/gradient diagnostics.
This v2 implements those requirements separately. Existing v1, Funnel16 and the frozen
`proposal_v2` models/configs/protocol remain unchanged; their results are not overwritten.
No claim of improved AP is made without running the experiments.

## Files

| File | Purpose |
|---|---|
| `src/training/supcon.py` | Stable log-sum-exp SupCon, skip missing classes/positives, balanced class contributions, pair/anchor diagnostics |
| `src/models/contrastive_projection.py` | Separate 168 → 64 → L2 projection; classifier unchanged |
| `src/training/supcon_adaptation.py` | Joint encoder/classifier/projection optimization, source selection and streamed training |
| `configs/proposal_supcon.json` | E1 CE + source SupCon |
| `configs/proposal_supcon_mkmmd.json` | E2 CE + source SupCon + marginal MK-MMD; ten-epoch warm-up |
| `src/experiments/proposal_supcon_v2.py` | Independent initialization, train/evaluate/aggregate CLI, exclusive outputs and provenance |
| `src/evaluation/supcon_diagnostics.py` | Bounded latent/gradient diagnostics and source-only/E1/E2 comparisons |
| `src/experiments/supcon_smoke.py` | Temporary synthetic Parquet one-batch E1/E2 smoke |
| `tests/test_supcon_v2.py` | Focused loss, isolation, initialization, reproducibility and preservation tests |

## Mathematical and training choices

Classification remains on the original 168D latent; SupCon uses a separate 64D linear
projection normalized to unit length; MK-MMD reuses the existing raw-168D, biased,
three-scale implementation. Evaluation loads only BaselineMLP to classify.
Projection weights are stored separately and loaded only for contrastive diagnostics.

Source SupCon excludes self-pairs. Only anchors with a same-label partner contribute.
The mean is computed within each source class with valid anchors, then averaged across
those classes to prevent benign majority counts dominating. Batches missing either
source class or any positive pair skip SupCon with a differentiable zero. A singleton
minority has no positive anchor but remains a negative for majority anchors; counts
are logged so the adequacy of positive coverage can be assessed.

Both E1/E2 load the same source-only checkpoint for the requested direction and seed,
initialize the same projection deterministically, and create fresh Adam optimizers.
E2 never reads E1 artifacts. Source CE weights remain inverse class-frequency weights;
BN running statistics are frozen in both continuation methods, while encoder BN affine
parameters can learn. No EMA, target pseudo-labels or class-aware alignment is included.

E1: `CE + 0.1 SupCon`. E2: `CE + 0.1 SupCon + effective_lambda MK-MMD`.
The E2 weight is `0.01 * min(1, epoch/10)` by default. Set `mmd_warmup_epochs` to 0
to disable warm-up before starting that experiment. Early stopping can occur before
full warm-up; inspect effective weights in history. Config hashes prevent reusing stale
artifacts after a change; preserve existing runs and version changed experiments separately.

Source validation AP selects checkpoints, including pretrained epoch 0. Source validation
Attack F1 selects the threshold. Target training streams request `features` only. Target
validation labels are used only after selection for reporting; target test is never read.
Source-only provenance is checked by the existing validator. Development locks are respected.

Training uses ParquetBatchStream and bounded batches. Validation accumulates labels/scores,
not entire input datasets. Diagnostics sample at most 4096 flows/domain, with 256-row MMD
subsamples by default. Per-batch counts/anchors are retained in history, which increases
JSON/checkpoint size for large datasets; raw training features are not retained.

## Commands

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

# CPU smoke first, then MPS if available. No real data/checkpoints accessed.
python -m experiments.proposal_supcon_v2 --stage smoke --device cpu
python -m experiments.proposal_supcon_v2 --stage smoke --device mps

# Focused tests.
python -m unittest discover -s tests -p test_supcon_v2.py -v

# E1 / E2 for one direction and one seed.
python -m experiments.proposal_supcon_v2 --stage train --experiment E1 \
  --direction unsw_to_cicids --seeds 42 --device cpu
python -m experiments.proposal_supcon_v2 --stage train --experiment E2 \
  --direction unsw_to_cicids --seeds 42 --device cpu
python -m experiments.proposal_supcon_v2 --stage evaluate \
  --direction unsw_to_cicids --seeds 42

# Full extension: run independently instead of the one-seed commands above.
# Requires all corresponding source-only pretrained checkpoints.
python -m experiments.proposal_supcon_v2 --stage train --experiment E1 --direction both --seeds 42 43 44 45 46
python -m experiments.proposal_supcon_v2 --stage train --experiment E2 --direction both --seeds 42 43 44 45 46
python -m experiments.proposal_supcon_v2 --stage evaluate --direction both --seeds 42 43 44 45 46
python -m experiments.proposal_supcon_v2 --stage aggregate --direction both --seeds 42 43 44 45 46
```

`--device auto` tries CUDA, then MPS, then CPU. Explicit MPS fails clearly if unavailable.
Use CPU if the local PyTorch MPS build cannot support a required MK-MMD operation.
No automatic CPU fallback hides a compatibility failure.

All outputs are exclusive-create under `models/<revision>/extensions/supcon_mkmmd_ext_v2`
and `results/<revision>/extensions/supcon_mkmmd_ext_v2`. Both experiments share the same
frozen input revision. Training refuses after development lock; do not remove the lock.
Evaluating existing checkpoints is read-only with respect to model/protocol artifacts.

For reevaluation, choose a new directory:

```bash
python -m experiments.proposal_supcon_v2 --stage evaluate --output-name evaluation_v2
python -m experiments.proposal_supcon_v2 --stage aggregate --output-name aggregate_v2
```

`--no-diagnostics` runs classification/score evaluation without latent/gradient diagnostics.
`--config <path>` requires `--experiment E1` or `E2`; the other config remains its default.

## Output structure (schema example, not measured results)

Training result JSON contains:

```text
extension_id, experiment, direction, seed
source_checkpoint, source_checkpoint_sha256, config_sha256
data_revision, data_manifest_sha256, prepared_split_sha256, preprocessor_sha256
best_epoch, selected_stage, best_source_val_ap, threshold
source_val: pr_auc, roc_auc, precision, recall, fpr, macro_f1, tn, fp, fn, tp
target_development: same metrics
paired_delta_vs_source_only: metric differences
history[]:
  epoch, total_loss, ce, supcon, mkmmd, bandwidth, effective_lambda
  source_val_ap, source_val_roc_auc, valid_contrastive_anchors
  skipped_contrastive_batches
  batch_diagnostics[]:
    class_counts, valid_anchors, valid_anchors_per_class
    positive_pairs, negative_pairs, skipped, skip_reason
```

Checkpoint adds `model_state_dict` and separate `projection_state_dict`.
Evaluation JSON contains `runs.source_only/E1/E2` with full source/target score analyses,
metrics and selected stages, plus `e2_minus_e1` and `negative_transfer_diagnostics`.
Diagnostics include fixed-bandwidth latent MMD before/after on matched subsamples,
grouped-CV domain AUC, existing class-conditional discrepancies/separation ratio,
BN shifts and encoder gradient cosine for CE versus MK-MMD and CE+SupCon versus MK-MMD.
Class-conditional discrepancies use diagnostic labels only and their existing separately
estimated single-RBF kernel; do not treat them as a fixed-kernel causal comparison.
MMD subsample std and domain-AUC fold std are not seed confidence intervals.

Aggregate JSON and `per_run.csv` preserve each seed, with paired mean ΔAP, sample std
and Student-t 95% CI vs source-only and E2 minus E1. Positive mean alone does not establish
statistical improvement; five-seed intervals describe optimization variation on fixed data.

## Validation status

Verified on 2026-10-11:

- 14 focused v2 tests passed, including synthetic diagnostics and Parquet streaming smoke.
- 4 existing v1 extension tests passed.
- Standalone CPU smoke CLI passed (one E1 batch and one E2 batch).
- MPS smoke was attempted; this execution environment reports MPS unavailable before
  any training batch, so MPS numerical/kernel compatibility is not verified.
- SHA-256 comparisons confirmed the eight original task-listed files and v1 config unchanged.

No full training, real-data diagnostic evaluation or final-test evaluation was launched.
