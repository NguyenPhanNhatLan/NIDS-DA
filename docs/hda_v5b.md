# V5b: V5a adapter with ranking preservation

V5b uses `HDAV1Model`. The student adapter starts from the same-seed V2 teacher
adapter; only `student.adapter.parameters()` are optimized with Adam, lr 0.001,
weight decay 0.0001. The entire UNSW seed-42 source and the teacher stay frozen.
There is no residual correction, alpha, LayerNorm, source CE, CORAL, or MK-MMD.

The natural target batch runs through the student adapter in train mode and
updates adapter BN once. Its hidden vector supplies both hidden MMD and student
ranking logits. Teacher margins use the same batch in eval mode with no gradients.
Balanced pseudo batches temporarily use adapter BN in eval mode, then restore
train mode, as in V4. Shared source BN and classifier always stay in eval mode.

The rank term is `1 - Pearson correlation(teacher margin, student margin)`.
Teacher margins are detached; constant teacher margins contribute zero rank
loss. All teacher scores and pseudo pools use target training features only.
Pseudo selection rules, source pools, epochs, batch sizes and source threshold
are inherited from the V5a protocol. The optimized loader preserves batch order.

## Symmetric first

Loss: `hidden + 0.05 Normal + 0.05 Attack + 0.10 rank`.

```bash
PYTHONPATH=src .venv/bin/python -u -m training.hda_v5b --config configs/hda_v5b_rank_0p10_symmetric.json --seed 42
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5b --config configs/hda_v5b_rank_0p10_symmetric.json --seed 42
```

## Asymmetric next

Loss: `hidden + 0.05 Normal + 0.02 Attack + 0.10 rank`.

```bash
PYTHONPATH=src .venv/bin/python -u -m training.hda_v5b --config configs/hda_v5b_rank_0p10_asymmetric.json --seed 42
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5b --config configs/hda_v5b_rank_0p10_asymmetric.json --seed 42
```

Both runs start independently from V2. Only the Attack coefficient differs.
Seed 43 or 44 selects its corresponding V2 teacher; source remains seed 42.
There is no `--mode rank` argument for these entry points.

Checkpoints contain `target_adapter_state_dict` and use
`models/hda_v5b/{symmetric,asymmetric}/v5b_seed42.pt`. Results use
`results/hda_v5b/{symmetric,asymmetric}/development/v5b_seed42.json`.
Previous residual experiments remain in their old directories and cannot be
loaded as V5b checkpoints.

Evaluation uses the UNSW validation threshold and target development split.
It evaluates the original V2 teacher on the same split and compares V5a lambda
0.10 results when their seed, source/teacher hashes, protocol, threshold and data
path match. Missing or incompatible V5a results are reported explicitly. Oracle
values print only as diagnostics. Target development labels never enter training.
