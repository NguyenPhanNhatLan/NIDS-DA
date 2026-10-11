# SupCon + MK-MMD experimental extension v1

This is a separate experiment, not a change to the frozen `proposal_v2` comparison.
The original BaselineMLP (5 → 256 → 168 → classifier), five ordered common features,
source-fitted preprocessing and canonical splits remain unchanged. The Funnel16
experiment and its class-aware continuation are not used by this extension.

## Experiments

- E1: weighted source CE + 0.1 × source SupCon; no target-training loader.
- E2: weighted source CE + 0.1 × source SupCon + 0.01 × marginal MK-MMD.

Both independently load the same corresponding `source_only_target_val/<direction>/seedN.pt`
and start fresh optimizers. E2 never loads E1. Initial checkpoint/result provenance
and source-validation AP/threshold are checked with the existing development validator.
Epoch zero participates in best-source-AP selection. Thresholds are selected by source-val F1.
Target-val labels are accessed only after checkpoint/threshold selection, for diagnostics.
Target test is not evaluated by this extension.

SupCon uses source labels only, cosine similarity on L2-normalized source latent
features, excludes self-pairs, and averages positive log probabilities for anchors
with a same-class partner. Anchors without positives are excluded; batches with none
return a differentiable zero and are counted in history. This is a one-view SupCon
variant: no new augmentation or projection head. Classifier inputs stay unchanged.
E2 reuses the existing biased three-scale marginal MK-MMD on raw 168D source/target
latent features. It does not classify target samples into alignment groups.

No EMA teacher, target pseudo-label supervision or new class-aware alignment is included.
There is no λ schedule in v1; weights are explicit in the separate config.

Source epoch length, shuffling, Adam settings, BN running-stat freezing and source
checkpoint selection match across E1/E2. Target features are forwarded separately
with their dropout RNG isolated from the source stream. History includes CE, SupCon,
MK-MMD, bandwidth, valid SupCon anchors and source-val AP per epoch.

## Run

The user runs these commands; no experiment execution is needed to install the code.
Prerequisites: audited/frozen canonical data and source-only checkpoints for all selected seeds.
An existing development lock is respected: training stops rather than altering the
locked protocol. Use an authorized open revision for further development; do not delete locks.

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

# One direction/seed first; each experiment initializes from source-only.
python -m experiments.proposal_supcon_extension --stage train --experiment E1 \
  --direction unsw_to_cicids --seeds 42
python -m experiments.proposal_supcon_extension --stage train --experiment E2 \
  --direction unsw_to_cicids --seeds 42
python -m experiments.proposal_supcon_extension --stage evaluate \
  --direction unsw_to_cicids --seeds 42

# Both directions, five seeds (E1 and E2; 20 new runs).
python -m experiments.proposal_supcon_extension --stage train
python -m experiments.proposal_supcon_extension --stage evaluate
python -m experiments.proposal_supcon_extension --stage aggregate
```

Do not run the one-seed commands and then the full commands without selecting the
remaining runs: duplicates intentionally raise errors. Use `--experiment`, `--direction`
and `--seeds` to specify missing runs.

Artifacts are exclusive-create under:

```text
models/<revision>/extensions/supcon_mkmmd_ext_v1/{E1,E2}/<direction>/seedN.pt
results/<revision>/extensions/supcon_mkmmd_ext_v1/{E1,E2}/<direction>/seedN.json
results/<revision>/extensions/supcon_mkmmd_ext_v1/evaluation/<direction>/seedN.json
results/<revision>/extensions/supcon_mkmmd_ext_v1/aggregate/{summary.json,per_run.csv}
```

Evaluation replays source selections and evaluates source-only/E1/E2 on full target
validation, including score distributions, PR curves, paired ΔAP vs source-only and
E2 − E1 AP. Aggregation requires every requested run, verifies provenance and checkpoint
hashes and reports mean, sample std and Student-t CI across seeds, including paired
AP differences. It never writes the original aggregate. CSV is ready for Tableau.
The five seeds measure optimizer variation on fixed data, not dataset uncertainty.

For another evaluation/export pass, choose a new directory:

```bash
python -m experiments.proposal_supcon_extension --stage evaluate --output-name evaluation_v2
python -m experiments.proposal_supcon_extension --stage aggregate --output-name aggregate_v2
```

Changing hyperparameters does not grant permission to reuse existing paths: checkpoint
config hashes must match. Preserve v1 artifacts and create a separately versioned experiment
when changing the protocol. No overwrite flag is provided.

## Synthetic checks (run manually)

```bash
PYTHONPATH="$PWD/src" python -m unittest discover -s tests -p test_supcon_extension.py -v
```

Checks cover the SupCon formula/finite gradients, no-positive behavior, permutation
invariance, independent E1/E2 source initialization, E1's absent target loader and
refusal to overwrite existing runs. They do not launch real training or require canonical data.
