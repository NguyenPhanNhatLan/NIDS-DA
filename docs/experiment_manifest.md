# Experiment manifest — proposal_v2

Audit date: 2026-10-05. This is an inventory, not a completed five-seed result.

Required matrix: 2 directions × seeds 42,43,44,45,46 × 4 methods = 40 runs.
Protocol: `configs/proposal_suite_v2.json`; schema: `configs/common_features_v2.json`.

| Config | SHA-256 |
|---|---|
| `common_features_v2.json` | `fd5d60a3ee1b4684ab0efe610092066428e75c9ea229ea6589e3f80127276057` |
| `proposal_class_aware_v2.json` | `643088fdeeaecfd90c69c89b8e317df2b79eb307ecedc3f2d245339e8219b966` |
| `proposal_mkmmd_v2.json` | `e8323778eb8f913ff1544356e721871afd225f50d9b7270ed5d76e0436363757` |
| `proposal_mmd_v2.json` | `8771176f99a4093e92e3c5e5cb535156c49ed02db58c1ec928c3d7f8af7629b7` |
| `proposal_suite_v2.json` | `2b5ffd942ccd5ae8a87423e24f9f21c891bcb33aed107b5d732d81d5aead8a89` |

## Local source-only artifacts

| Direction | Seed | Schema/preprocessor/prepared hashes | Status |
|---|---|---|---|
| unsw_to_cicids | 42 | Match | Existing; protocol metadata absent |
| unsw_to_cicids | 43 | — | Missing |
| unsw_to_cicids | 44 | — | Missing |
| unsw_to_cicids | 45 | — | Missing |
| unsw_to_cicids | 46 | — | Missing |
| cicids_to_unsw | 42 | Match | Existing; protocol metadata absent |
| cicids_to_unsw | 43 | — | Missing |
| cicids_to_unsw | 44 | — | Missing |
| cicids_to_unsw | 45 | — | Missing |
| cicids_to_unsw | 46 | — | Missing |

The inventory validates JSON provenance against current prepared-data hashes; it does
not certify checkpoint replay or final-test readiness. Existing source-only JSON lacks a protocol field; the current five-feature schema and
prepared hashes match. These artifacts are never relabeled or rewritten. Checkpoint
replay and selection checks still run before any final-test labels are read.

No current canonical MMD/MK-MMD/Class-aware results were generated in this task.
The five-seed suite is incomplete; no empirical aggregate or final test was run.
Old standalone Class-aware and MCD artifacts and the corresponding purity report
are archived under `legacy/`. Old proposal_v1 artifacts are archived without relabeling.

## Frozen output locations

- Source-only: `{models,results}/proposal_v2/source_only_target_val/{direction}/seed{seed}.{pt,json}`.
- MMD methods: `{models,results}/proposal_v2/mmd/target_val_epoch0_{config_stem}/{direction}/seed{seed}.{pt,json}`.
- Diagnostics: `results/proposal_v2/diagnostics_target_val/{direction}/seed{seed}[_config_stem].json`.
- Development aggregate: `results/proposal_v2/aggregate/summary.json`.
- Final test: `results/proposal_v2/final_target_test/{method}/{direction}/seed{seed}.json`.

Artifacts bind feature-config, method-config, preprocessor and prepared-split hashes.
Final evaluation additionally verifies checkpoint hashes and source-val AP/threshold.
To resume runs, inspect existing outputs first; generators reject overwrite. Never
change recorded hashes to force compatibility. Keep any subsequent empirical results
separate from synthetic test validation.

## Validation

27 unittest cases pass on synthetic fixtures, including five-feature Parquet loading,
target features-only adaptation, shared method dispatch, provenance checks, final-test
threshold protection, sample std/Student-t CI, duplicate seed rejection and rejection
of a ten-feature checkpoint. No thesis-scale training was executed.

Static compilation and `git diff --check` pass. The local `.venv` lacks `pyspark`,
so the Spark splitting CLI was not executed; README includes its installation step.
The remaining 11 pipeline CLIs successfully import and expose `--help`.
