# Frozen thesis protocol: proposal_v2

Schema authority: `configs/common_features_v2.json`; suite: `configs/proposal_suite_v2.json`.
Exactly five ordered inputs: flow_duration (seconds), fwd_packets, bwd_packets,
fwd_bytes, bwd_bytes. CICIDS duration is converted from microseconds to seconds.
Feature mappings, direction and measurement assumptions remain documented in
`feature_mapping_audit.csv` and `data_dictionary/`. Matching column concepts does
not guarantee identical extraction semantics across datasets.

Both transfer directions are required: UNSW → CICIDS and CICIDS → UNSW.
Split membership is fixed using the five raw columns and hash seed 42 (70/15/15);
equal split keys stay together. Run seeds 42–46 vary optimization, not split membership.
The canonical freeze then moves complete float32 preprocessing-collision groups
to the furthest held-out split and refits source-only processors until neither
direction has cross-split overlap. All flows and original labels are preserved.
The resulting membership is fixed for all training seeds; realized split proportions
can differ from 70/15/15. See `canonical_data_revision.md` and the completion manifest
at `data/revisions/canonical/manifest.json`.
The raw cleaning/export notebooks precede splitting; harmonization subsequently
orders features and converts duration units. All fitted preprocessing is downstream
of splitting: source-train median imputation → signed log1p → RobustScaler.
The same fitted processor transforms all source and target splits for that direction.
Canonical preprocessing uses exact sklearn source-train median/IQR after the
precision grouping correction. Only `experiments.proposal_pipeline` is the supported
entry point; its stages and paths are documented in `README.md`.

| Method | Configuration | Alignment |
|---|---|---|
| Source-only MLP | suite settings / existing baseline trainer | None |
| Marginal MMD | proposal_mmd_v2.json | Single RBF, shared latent space |
| MK-MMD | proposal_mkmmd_v2.json | Existing multiple RBF scales |
| Class-aware MMD | proposal_class_aware_v2.json | Per-class RBF using confident target predictions |

The three MMD methods share `training.proposal_mmd.run`, source initialization,
checkpoint selection and output format. Class-aware uses the confidence rule
specified in its config; historical quantile-teacher standalone experiments remain
in legacy and are excluded from this suite. No new architecture or loss is introduced
into the thesis comparison. LR/XGBoost are optional auxiliary baselines.

Target-train labels are never requested by adaptation loaders or pseudo-label audit.
Checkpoint selection uses source-val AP and includes pretrained epoch 0; source-val
F1 selects the threshold. Target-val labels are post-hoc development diagnostics,
including domain shift, negative transfer and pseudo-label behavior; they do not select
checkpoints or thresholds. Domain classifier and diagnostic oracle thresholds are
analysis outputs, not deployable target-calibrated models.

Report AP (JSON key `pr_auc`, computed as average precision), macro F1, recall, FPR,
ROC AUC and confusion counts for each seed. For each direction/method, retain every
paired ΔAP = AP(method, seed) − AP(source-only, same seed), then report mean, sample
standard deviation (ddof=1) and two-sided 95% Student-t CI:

`mean_delta ± t(0.975, n−1) × sample_std(delta) / sqrt(n)`.

The interval is formed from paired differences, not from independent method intervals.
For one seed CI is null. Duplicate seeds and missing runs are errors. With n=5,
normality and independence across seeds are assumptions and precision is limited;
this describes run variation on fixed datasets, not dataset-level uncertainty.
No p-value/significance test is included. Do not infer significance from five seeds
alone or pool seeds across transfer directions.

Aggregate validates the current feature-schema hash, method-config hashes and matched
prepared-data provenance. Existing artifacts must not be renamed or edited to make
hashes pass. Archive incompatible artifacts and regenerate when an actual run is
requested. Diagnostics are optional in aggregation, with included run counts reported.

Only after development is frozen may final evaluation read target test. It first checks
checkpoint/result identity and provenance and reproduces source-val AP/threshold.
Final outputs are exclusive-create and separate from development summaries. The
current aggregate is a development report; final target-test metrics remain explicitly
labeled final and must not be presented as development metrics or used for tuning.
