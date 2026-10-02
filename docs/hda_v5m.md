# V5m: V5d adapter + original source classifier

No optimizer or training loop is present. The assembly command reads and
validates the existing V5d checkpoint, copies only target_adapter_state_dict,
and keeps the original source classifier and source tail. All parameters are frozen.

Calibration is a NEW V5m artifact. It selects the 2% source-FPR threshold on
UNSW validation and fits positive affine parameters using UNSW validation and
frozen V2 hard anchors from unlabeled adaptation-train. V5d affine parameters
are not reused. Development labels are read only by the report command.

```bash
# Assemble only; this does not train V5d or V5m.
PYTHONPATH=src .venv/bin/python -u -m training.hda_v5m --config configs/hda_v5m.json

# Fit independent calibration without development labels.
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5m --config configs/hda_v5m.json --stage fit

# Evaluate the frozen calibrated V5m on development.
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5m --config configs/hda_v5m.json --stage report
```

Existing files are never overwritten. Paths are under models/hda_v5m/adapter_swap
and results/hda_v5m/adapter_swap. Source/V5d dependencies are hashed; changing
the parent checkpoint or code invalidates the derived artifact.

Optional synthetic assembly test (not run during authoring):

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_hda_v5m.py -v
```
