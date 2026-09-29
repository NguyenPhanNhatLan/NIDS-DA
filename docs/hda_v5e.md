# V5e-1: controlled MK-MMD experiment

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

Outputs: `models/hda_v5e/affine_fpr_0p02/v5e_seed42.pt`,
`results/hda_v5e/affine_fpr_0p02/calibration/seed42.json`, and
`results/hda_v5e/affine_fpr_0p02/development/v5e_vs_v5b_seed42.json`.
The final JSON contains all three models despite the two-model filename.
