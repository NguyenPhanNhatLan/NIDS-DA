# V5k: hard-anchor latent VAT

V5k preserves V5b asymmetric weights (hidden=1, normal=.05, attack=.02,
rank=.10), frozen source/classifier and V2 initialization. VAT weight=.10,
epsilon_ratio=.05, xi_ratio=.001, power_iterations=1.

VAT is evaluated on balanced hard V2 anchors only: Normal margin <= q02 and
Attack q95 <= margin < q98. It shares the conditional latent forward with
adapter BN in eval mode. Natural target batches still drive hidden MMD and
ranking, with one natural adapter BN update per step. VAT does not use target
labels or middle-region samples. The radius uses max(||z||, 1), as in V5j.

The supplied V5k configuration uses source FPR 1% and its pinned V5b report.
V5j currently uses 2%; operating-point comparisons across those policies are
not controlled comparisons. AP/ROC-AUC remain threshold-independent.

```bash
PYTHONPATH=src .venv/bin/python -u -m training.hda_v5k --seed 42 --preflight-only
PYTHONPATH=src .venv/bin/python -u -m training.hda_v5k --seed 42 --device auto
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5k --seed 42 --stage fit
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5k --seed 42 --stage report
```

Checkpoint and reports record vat_scope=hard_v2_anchors_only_balanced.
Existing version files and artifacts are not overwritten.
