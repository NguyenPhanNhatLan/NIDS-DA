# V5o: fixed cross-class MMD separation

V5m is frozen. V5o copies V5d training with seed 42, the same optimizer,
sampling, teacher, five losses and last-epoch checkpoint selection. Its only
loss change is `-0.02 * MMD(source_attack, target_pseudo_normal)`. The source
Attack tensor is reused from the attack alignment term; no extra pool sampling.
History records separation and its negative weighted contribution.

The auxiliary classifier trains as in V5d but is omitted from the checkpoint.
Loading deployment uses the trained adapter, frozen original source fc2/bn2
and original source classifier. No EMA or BN recalibration is introduced.

From the project root, run in order (these commands have not been executed):

```bash
PYTHONPATH=src python -m training.hda_v5o --preflight-only
PYTHONPATH=src python -m training.hda_v5o --device auto
PYTHONPATH=src python -m evaluation.hda_v5o --stage fit
PYTHONPATH=src python -m evaluation.hda_v5o --stage report
```

Calibration is newly fitted using UNSW validation and frozen V2 q02/q95-q98
anchors from adaptation train. Source FPR policy stays 2%. No target development
labels are used in training or calibration. Outputs are separate under hda_v5o;
existing checkpoints, calibrations and reports cannot be overwritten.

Report includes raw/calibrated AP, ROC-AUC, F1, Recall and FPR, plus development
oracle Recall@FPR 1%, 2%, 3%, 5% with achieved FPR and raw threshold. Oracle
thresholds are diagnostic only, never deployment thresholds. They maximize
recall subject to empirical FPR <= budget, keep ties together and do not interpolate.

V5m comparison uses user-provided rounded benchmarks: AP 0.645015, ROC 0.776097,
R@1% 0.383218, R@2% 0.401620, R@3% 0.522398, R@5% 0.556643. These are not
recomputed; compare on the same split and Recall@FPR convention.

Keep V5o only after reviewing the specified AP/ROC/recall tradeoff. No automatic
pass/fail is assigned because “approximately 0.64” and “no significant decrease”
have no numeric tolerances. Otherwise stop at V5m. No V5p or gamma sweep.
