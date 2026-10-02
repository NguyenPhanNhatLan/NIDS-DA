# V5n: V5d joint training + EMA adapter

One fixed experiment: beta=0.999. Training starts from the same frozen V5b
adapter and original source classifier as V5d. Loss weights, Adam parameter
groups, LR, weight decay, pseudo pools, BN policy and epoch schedule are copied
from V5d. The private classifier is trained as an auxiliary component.

EMA clones adapter parameters before the first optimizer step, then updates
after every step. It does not enter the loss, gradient graph or RNG stream.
After the final epoch, parameters are replaced with EMA (no bias correction).
Adapter BN running statistics are reset and recomputed with cumulative moving
averages on natural unlabeled CICIDS adaptation-train batches. The existing
make_unlabeled_loader shuffles full batches and drops the final incomplete
batch. The checkpoint records actual rows/batches used. BN momentum and modes
are restored after recalibration. Source BN is untouched.

The auxiliary classifier is replaced by the original source classifier and
is NOT saved. Deployment checkpoints contain only EMA adapter state (including
recalibrated BN buffers), provenance and diagnostics. Existing V5d/V5m files
are not modified. This is a new training run, not a recoverable EMA of an old
last-epoch checkpoint.

V5n fits a fresh affine calibration on UNSW validation + frozen V2 hard anchors,
with the same 2% source-FPR policy. No development labels enter training,
BN recalibration or affine fitting.

```bash
PYTHONPATH=src .venv/bin/python -u -m training.hda_v5n --config configs/hda_v5n.json --training-seed 42 --preflight-only
PYTHONPATH=src .venv/bin/python -u -m training.hda_v5n --config configs/hda_v5n.json --training-seed 42 --device auto
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5n --config configs/hda_v5n.json --training-seed 42 --stage fit
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5n --config configs/hda_v5n.json --training-seed 42 --stage report
```

Tests are supplied but were not executed during authoring:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_hda_v5n.py -v
```
