# V5j = V5b + latent VAT

Files:
- `src/training/latent_vat.py`
- `src/training/hda_v5j.py`
- `src/evaluation/hda_v5j.py`
- `configs/hda_v5j.json`
- `tests/test_latent_vat.py`

The experiment preserves V5b:
- hidden MMD = 1.0
- normal conditional MMD = 0.05
- attack conditional MMD = 0.02
- ranking loss = 0.10
- V2 q02 / q95-q98 fixed pseudo pools
- frozen source tail and classifier
- same optimizer and 10 epochs

Only new term:
`+ 0.10 * latent_VAT`

VAT:
- natural target batch only
- perturb post-ReLU shared latent `z`
- classifier/source tail remain frozen
- no target labels
- no pseudo labels used by VAT
- epsilon = 5% of max(each sample latent L2 norm, 1)
- xi = 0.1% of max(latent L2 norm, 1)
- one power iteration

## Preflight
```bash
PYTHONPATH=src .venv/bin/python -u -m training.hda_v5j --preflight-only --seed 42
```

The default calibrated V5b reference uses a 2% source FPR budget. Its SHA-256 and seed are pinned in the config. Additional seeds require their corresponding reference report and pin. Code hashes include training and evaluation. Existing checkpoints are never overwritten.

## Test
```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_latent_vat.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_hda_v5j.py
PYTHONPATH=src .venv/bin/python -m py_compile \
  src/training/latent_vat.py \
  src/training/hda_v5j.py \
  src/evaluation/hda_v5j.py
```

## Train seed 42
```bash
PYTHONPATH=src .venv/bin/python -u -m training.hda_v5j \
  --config configs/hda_v5j.json \
  --seed 42 \
  --device mps
```

## Fit affine calibration
```bash
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5j \
  --config configs/hda_v5j.json \
  --seed 42 \
  --stage fit
```

## Development report
```bash
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5j \
  --config configs/hda_v5j.json \
  --seed 42 \
  --stage report
```

First-run decision:
- do not tune VAT hyperparameters before seed-42 result
- compare AP, ROC-AUC, F1, Recall, FPR to frozen V5b
- if ranking/operating point is promising, then run seeds 43/44
