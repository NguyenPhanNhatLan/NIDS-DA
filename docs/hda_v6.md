# HDA V6 development

| Version | Architecture | Parameters trained during adaptation | Loss |
| --- | --- | --- | --- |
| V6a | Existing widths, BN replaced by LN, latent 168D | Target stem | marginal + 0.05 normal + 0.05 attack |
| V6b | Two residual blocks + LN, latent 128D | Target stem | marginal + 0.05 normal + 0.05 attack |
| V6c | Same as V6b | Both stems, shared encoder, classifier | source CE + 0.10 marginal + 0.05 normal + 0.02 attack |

V6a/b retain V5a's conditional weight 0.10 because
`0.10 * (normal + attack) / 2 = 0.05 normal + 0.05 attack`.
V6c changes both the adaptation training policy and loss weights. A V6b→V6c
improvement cannot be attributed to asymmetric weighting alone.

The residual architecture follows the supplied specification: separate
`Linear → GELU → Linear → LayerNorm` stems produce 256D vectors; two pre-norm
residual blocks use widths 256→384→256 and dropout 0.1; the bottleneck is
`LayerNorm → Linear(256,128) → GELU → LayerNorm`; the classifier is
`128 → GELU(32) → Dropout(0.1) → 2`. There is no BatchNorm in the V6 model.

## Training sequence

Source pretraining uses UNSW labels, class-weighted CE, and source-validation AP
for checkpoint selection, as in the existing baseline. Source seed remains 42.
Each architecture needs its own new source checkpoint; BN/168D weights cannot
be loaded as the residual LN/128D model.

Warmup freezes the source stem, encoder, and classifier and trains the target stem
for 10 epochs using hidden 256D marginal MMD. Adaptation then runs for 10 epochs
and selects the last epoch. V6b/c share the exact same source and warmup checkpoint.
All alignment terms use the existing single-RBF MMD; conditional terms operate
on current latent vectors. V6c recomputes source latents with the current trainable
encoder rather than caching old latents. Its source CE is ordinary, unweighted CE.

Target pseudo pools retain the old same-seed V2 teacher and its rules:
Normal `margin <= q02`; Attack `q95 <= margin < q98`. Teacher scores come only
from adaptation-train features. The old teacher may contain BN, but it is frozen
and separate from the new V6 model. This keeps pseudo-pool selection fixed across
the architecture ablations. No target CE or CORAL is added.

## Commands

Run V6a first:

```bash
PYTHONPATH=src .venv/bin/python -u -m training.hda_v6 --version v6a --stage source
PYTHONPATH=src .venv/bin/python -u -m training.hda_v6 --version v6a --stage warmup --seed 42
PYTHONPATH=src .venv/bin/python -u -m training.hda_v6 --version v6a --stage adapt --seed 42
PYTHONPATH=src .venv/bin/python -m evaluation.hda_v6 --version v6a --seed 42
```

Then V6b:

```bash
PYTHONPATH=src .venv/bin/python -u -m training.hda_v6 --version v6b --stage source
PYTHONPATH=src .venv/bin/python -u -m training.hda_v6 --version v6b --stage warmup --seed 42
PYTHONPATH=src .venv/bin/python -u -m training.hda_v6 --version v6b --stage adapt --seed 42
PYTHONPATH=src .venv/bin/python -m evaluation.hda_v6 --version v6b --seed 42
```

V6c reuses V6b's source and warmup, not its adapted checkpoint:

```bash
PYTHONPATH=src .venv/bin/python -u -m training.hda_v6 --version v6c --stage adapt --seed 42
PYTHONPATH=src .venv/bin/python -m evaluation.hda_v6 --version v6c --seed 42
```

For seeds 43/44, repeat warmup, adaptation, and evaluation with that seed; source
pretraining is shared at seed 42. Existing checkpoints are not overwritten.
Config: `configs/hda_v6.json`. Checkpoints: `models/hda_v6/`.
Results: `results/hda_v6/development/`.

## Evaluation

After adaptation, select the decision threshold on UNSW validation using the
final model's source branch. This is necessary because V6c updates the encoder
and classifier. The historical UNSW BN-model threshold is not reused.
Then evaluate the explicitly configured target development split. Target labels
are never read by training. Purity and oracle metrics print only as diagnostics;
oracle values are not saved as primary metrics. JSON results contain deltas
against the fixed V4 development reference and hashes of config, checkpoint,
training code, and evaluation code.

The current `cicids_test` is previously inspected development data. This workflow
does not implement final holdout evaluation. These architecture and loss changes
are hypotheses to test; they do not guarantee ROC-AUC ≥ 0.80.
