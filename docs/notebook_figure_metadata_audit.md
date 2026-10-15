# Notebook figure metadata audit — 2026-10-14

Notebook: `notebooks/research_csv_visualizations.ipynb`.

## Figure 4.1 (Cell 11)

Use `results/analysis/notebook_figures/figure_4_1_class_ecdf.pdf` or `.png`.
The caption is in `figure_4_1_class_ecdf.caption.txt`; provenance and checks are
in `figure_4_1_class_ecdf.metadata.json`. This replaces the earlier dashboard
export `04_class-conditional_feature_ecdf` for manuscript use.

The class diagnostics manifest identifies `unsw_to_cicids`, `unsw_val` as Source,
`cicids_val` as Target, and `post_hoc_validation_diagnostic_only`.
ECDF is based on all complete cases across the five common features:

| Domain | Benign | Attack | Excluded nonfinite rows |
|---|---:|---:|---:|
| UNSW-NB15 | 298,017 | 11,172 | 0 |
| CICIDS2017 | 305,730 | 67,146 | 2 |

Checked all 763,838 ECDF CSV rows: `value_log1p` matches
`sign(value_raw) * log1p(abs(value_raw))`. Each feature/domain/class series has
`sample_n` matching the manifest and a monotone ECDF ending at 1. The CSV has
no sampling seed. `scatter_sampling_seed=42` applies to the scatter export,
not the full-validation ECDF. The notebook only applies a sampling-seed filter
if that column exists, and records the filter in the figure metadata.

The checks verify CSV consistency with the declared manifest and the numerical
transform; they do not regenerate the ECDF from raw Parquet or independently
prove the exporter lineage. Inputs are repository-local and are not embedded
inside the notebook.

## Common conventions

- Models use `METHOD_COLORS`; domain/class series use separate named palettes.
  Before/after or attack class bars use hatching when they share a model color.
- `finish(note=None)` has no default data-role claim. Plot calls provide specific
  roles and uncertainty descriptions; ordinary exports also have `.metadata.json`.
- Intrinsic feature diagnostics use unlabeled training samples (100,000/domain
  per sampling seed), with five sampling seeds 42–46 in the intrinsic manifest.
- Class-conditioned ECDF/KS and joint plots use labeled validation, post-hoc.
- RQ1 uses development validation across five training seeds. PR and score curves
  use the explicitly selected training seed. UMAP selects training and sampling
  seeds explicitly, using precomputed coordinates.
- Latent alignment is post-hoc. Gradient diagnostics use labeled source training
  and feature-only target training minibatches. kNN is a balanced validation
  diagnostic, with comparison, k, query/reference roles and class kept separate.
- Raw-run errors use `sd_low/high` (mean ± SD). Exported CI uses only actual
  `ci95_low/high`. Intrinsic KS min–max uses `ks_min/max`, never CI names.
- `load(filters=...)` applies the same filter regardless of file size. Chunking
  controls memory only. Unfiltered loads retain all seeds.

Figure 4.1 has five panels and a single shared legend, no figure title/footer,
and a portrait layout approximately 16 × 20 cm, exported at 300 dpi and PDF.
The Word caption should supply figure number, data role, domains and transform.
