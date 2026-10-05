# V5e-1: controlled MK-MMD experiment

## Adapter gradient conflict before changing loss weights

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_v5e_gradient_conflict.py' -v
PYTHONPATH=src .venv/bin/python -m evaluation.v5e_gradient_conflict --config configs/hda_v5e.json --training-seed 42
```

Reads the existing checkpoint and `fixed_diagnostic_batches`; no data sampling,
training or optimizer step. Uses checkpoint weights (1, .05, .02, .10) for
Hidden, Normal, Attack and Ranking. Computes autograd gradients only for adapter
parameters, reporting each weighted gradient norm, norm of H+N+R, dot product,
cosine(A,H+N+R) and Attack/other norm ratio. Zero/near-zero norm yields JSON null
for undefined cosine rather than a misleading 0.

Default `--bandwidth fixed` uses saved bandwidths. Optional `--bandwidth batch`
re-estimates detached median bandwidths on the same saved batches to examine
sensitivity to the training bandwidth rule. Both use eval BN on private copies;
original parameters, buffers, training modes, requires_grad flags and .grad stay
unchanged. This is an eval-mode local diagnostic, not the exact natural-batch
train-mode gradient or Adam update. Source CE has zero adapter gradient because
its computational path uses only source representations and the classifier.

Default output is stdout. Use `--output results/hda_v5e/gradient_conflict_seed42.json`
to save a new JSON; existing files are never overwritten. Optional `--device mps`
enables MPS; CPU is the default. No existing hash-tracked training file is changed
by adding this standalone reader, so it does not itself invalidate checkpoints.
Negative cosine indicates opposing local directions, whereas small Attack norm
indicates weak influence; neither alone establishes causation for AP/F1 changes.
Tests and diagnostics have not been executed by the assistant.

## Inspect an existing result before training another model

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.v5e_diagnostics
```

This read-only command loads the optimized seed-42 report below, verifies equal
fixed bandwidths across the three models for each space, and prints Hidden,
Normal and Attack MMD plus deltas (V5d minus V5b, V5e minus V5d, V5e minus V5b).
It also prints AP/ROC-AUC/F1/Recall/FPR when present. No models are loaded, no
training is performed and no artifact is changed. Use `--report path/to/report.json`
to inspect another existing report. A missing report produces an explicit error.
Compare each kernel column separately, especially Normal and Attack: lower MMD
is an alignment diagnostic, not proof of improved AP or F1. This reader checks
bandwidth values; batch identity relies on the producing pipeline's provenance.

## Optimized implementation v1

The optimized kernel uses one combined `cdist` for detached median bandwidth
and the differentiable SS/TT/ST blocks; one batched `exp` computes all five
kernels. Mathematical loss is unchanged; floating-point results/gradients can
differ slightly. Kernel storage increases to K*(2N)^2, so benchmark memory and
time on the real device rather than assuming a speedup.

Training validates scales once and keeps them on device. Detailed tensor checks
are optional via `kernel_debug_every_steps` (0 disables them); total loss is
still checked for NaN/Inf every step before backward. Epoch statistics accumulate
detached on-device tensors and transfer together every 200 steps/end of epoch.
This reduces, but does not eliminate synchronization: positive-distance masking
for the median, the total-loss check, and the existing ranking loss can still
synchronize. `adaptation.py` and frozen V5b/V5d remain untouched.

Pool cache is dependency-addressed under `results/hda_v5e/pool_cache_v1`.
Keys include data SHA-256 snapshots, source/V2 checkpoint hashes, pseudo policy,
dimensions, batch size, backend, Torch version and pool-producer code hashes.
Existing entries are read-only; content checksum is verified before loading.
Unknown dependencies create another entry; corrupt/partial entries fail rather
than being silently reused or overwritten. CPU teacher/pseudo policy stays the
same. Source latent pools are cached separately per backend. First creation is
still expensive; future hits avoid teacher inference and pool construction.
Data files are still hashed for provenance. Cache hit/miss preserves CPU RNG;
this revision can have a different training RNG trajectory from the old code,
which consumed loader RNG during pool preparation. Re-run as a new experiment.

Profile runs are disposable: they train a temporary in-memory student for the
requested steps, save timing JSON, and never save a trained checkpoint. Cache
creation is allowed. Phase timings include preflight hashes, snapshots, both
pools and diagnostics. Step timings measure data reads, three MK forwards, whole
loss backward and optimizer with explicit synchronization. First 10 steps are
excluded from phase averages; reported end-to-end steps/s includes warmup.
Timing synchronization itself adds overhead; it is disabled in normal training.

Suggested sequence (all commands are for the user; none executed here):

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_mkmmd.py' -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_hda_v5e.py' -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_v5e_performance.py' -v

# Synthetic same-input loss/gradient tests include MPS if available.
# Measure MK forward/backward throughput on real MPS:
PYTHONPATH=src .venv/bin/python -m training.benchmark_mkmmd --device mps --steps 200

# Full data path: legacy math baseline then optimized math, same training loop.
PYTHONPATH=src .venv/bin/python -u -m training.hda_v5e --config configs/hda_v5e.json --device mps --training-seed 42 --profile-steps 200 --profile-kernel legacy
PYTHONPATH=src .venv/bin/python -u -m training.hda_v5e --config configs/hda_v5e.json --device mps --training-seed 42 --profile-steps 200 --profile-kernel optimized
```

Legacy profile retains the old four-cdist mathematics, not all old validation
overhead. Compare step timings with warm cache separately from cold startup;
repeat with `--profile-output` pointing to a new JSON when needed. Profiles live
under `results/hda_v5e/profiles`. Performance is not claimed until measured.

Then run the train/fit/report commands below. Optimized artifacts are isolated
under `models/hda_v5e/optimized_v1/affine_fpr_0p02` and
`results/hda_v5e/optimized_v1/affine_fpr_0p02`. Old checkpoints/code hashes are not
rewritten or resumed. Calibration remains a separate fit after full training.

New files implement MK-MMD and a separate V5e training/evaluation branch.
`adaptation.py`, V5b/V5d source files and frozen checkpoints are unchanged.
V5e uses the same HDAV5DModel architecture: private classifier, frozen source
encoder, V5b-initialized target adapter. It does not initialize from the trained
V5d checkpoint. V5d is a pinned comparison reference only.

## Kernel and controls

`mk_mmd_loss(S, T)` returns `(biased_MMD_squared, base_sigma)`.
The RBF mixture is the arithmetic mean of five kernels:
`exp(-distance_squared / (2 * base_sigma_squared * scale))`, with scales
`[0.25, 0.5, 1, 2, 4]`. These multiply sigma squared, not sigma.
The base bandwidth is the detached median of positive pooled pairwise squared
distances, matching the original single-RBF heuristic. Degenerate batches fall
back to bandwidth squared 1. Batch lengths are truncated to the smaller count,
as in V5b/V5d. Self-pairs are included. Kernel weights are fixed, not learned.
The mean keeps total kernel weight 1. No clamp conceals small rounding errors.

Only Hidden, Normal and Attack MMD calls change. Source CE weight 0.10,
ranking weight 0.10, conditional weights 0.05/0.02, source class weighting,
adapter/classifier learning rates 1e-4/1e-5, weight decay, seed, epochs,
initialization, frozen V5b teacher and BN policy stay the same as V5d joint.
The config and reference checks reject accidental changes to these controls.
No extra source-CE-free control is included in V5e-1.

## Fixed diagnostics

Before training, select the first deterministic source/target batches and first
64 rows of each class pool. Freeze these tensors and three base bandwidths at
the V5b initialization; record them in the checkpoint. No target labels are read.
The deterministic loaders preserve training RNG state. Diagnostic forwards use
eval/no_grad and restore module modes; they do not update BN statistics.

Compute Hidden, Normal and Attack MMD under BOTH single RBF and MK-RBF for:

- V5b adapter initialization;
- the pinned trained V5d reference;
- V5e at every epoch, including the final checkpoint.

Logs and checkpoint history include these diagnostics. Compare different models
within the SAME kernel/bandwidth column. Do not compare single-RBF loss magnitude
directly against MK-RBF loss magnitude to claim better alignment. These small,
fixed batches are diagnostic samples, not population estimates. Training loss
uses a fresh median per term/batch; diagnostics use fixed initial bandwidths.

## Run sequence (not executed during code authoring)

```bash
# Start with the kernel tests; continue only if they pass.
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_mkmmd.py' -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_hda_v5e.py' -v

PYTHONPATH=src .venv/bin/python -u -m training.hda_v5e --config configs/hda_v5e.json --training-seed 42 --preflight-only
PYTHONPATH=src .venv/bin/python -u -m training.hda_v5e --config configs/hda_v5e.json --training-seed 42
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5e --config configs/hda_v5e.json --training-seed 42 --stage fit
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5e --config configs/hda_v5e.json --training-seed 42 --stage report
```

Kernel tests cover the manual formula, single-kernel equivalence to original
MMD, unequal batches, symmetry, identity, collapsed representations, fixed-
bandwidth gradient checking, detached bandwidth and invalid inputs. Branch tests
cover three MK-MMD calls, CE preservation, frozen teacher/source/BN and diagnostic
RNG/buffer invariance. These tests have been written but not run by the assistant.

Preflight checks frozen V5b calibration and pinned V5d report/checkpoint. The
current reference is seed 42; additional seeds require corresponding pinned
V5d references, not comparison to seed 42 by accident. No artifact is overwritten.
Checkpoint selection stays last epoch, not development AP.

V5e fits its own positive affine calibration on UNSW validation and V2 pseudo
pools from unlabeled adaptation-train. It uses the same 2% source FPR policy.
V5b uses its frozen artifact. Development is read only in report; V5d report
data snapshots must match. Raw and calibrated operating points are both saved.
AP/ROC-AUC are ranking metrics and are not improved by positive affine fitting.

The report prints AP, ROC-AUC, F1, Recall, FPR for V5b, V5d and V5e, with deltas.
Existing reference values (not new measurements):

| Model | AP | ROC-AUC | F1 |
| --- | ---: | ---: | ---: |
| V5b single RBF | 0.5851 | 0.8098 | 0.5693 |
| V5d single RBF + source CE | 0.6280 | 0.7873 | 0.5245 |
| V5e-1 MK-MMD + same source CE | pending | pending | pending |

Outputs: `models/hda_v5e/optimized_v1/affine_fpr_0p02/v5e_seed42.pt`,
`results/hda_v5e/optimized_v1/affine_fpr_0p02/calibration/seed42.json`, and
`results/hda_v5e/optimized_v1/affine_fpr_0p02/development/v5e_vs_v5b_seed42.json`.
The final JSON contains all three models despite the two-model filename.
