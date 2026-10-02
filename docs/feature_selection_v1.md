# Feature selection protocol V1

This is an opt-in experiment before model training. It does not modify the heterogeneous HDA architecture, separate domain stems, existing feature vectors, splits, or proposal-suite defaults. The goal is semantic compatibility followed by train-only quantitative ranking, not maximizing target evaluation scores.

## Repository audit

The inputs are `data/splits/unsw_train` (1,442,111 rows) and `data/splits/cicids_train` (1,764,877 rows). `data/processed/clean_{unsw,cicids}.parquet` contains the cleaned unsplit data. Never substitute `data/features/*`: those are independently normalized model vectors.

Cleaning lives in `notebooks/unsw.ipynb` and `notebooks/cicids.ipynb`, not the legacy `data_processing` entry points still listed in pyproject. Both normalize column names and deduplicate. UNSW fills selected HTTP/FTP fields, cleans categorical values, and drops addresses, ports, timestamps, sequence numbers and handshake components. CICIDS removes constant/near-constant and correlated columns. Its exported `cicids` frame is not the separate `cicids_log` exploratory frame. UNSW's exploratory logs similarly operate on other variables. The named split parquets are therefore pre-log and pre-scaler feature columns.

`src/features/splitting.py` hashes records into 70/15/15 splits. `feature_engineering.py` later fits domain-specific median imputation, different log lists and separate StandardScalers; UNSW additionally encodes categorical columns. None of those model vectors is used here. `configs/proposal_suite_v1.json` mentions 16 `COMMON_FEATURES`, but there is no corresponding definition in the current source tree; its referenced experiment modules are also absent. It is left unchanged.

**Historical limitation:** CICIDS correlation pruning occurred before the split, using an unlabeled sample of the full dataset. This scorer is train-only, but the existing dataset's column availability is not the result of a fully train-only preprocessing protocol. Also, the existing UNSW split hash excludes `label` but includes `attack_cat`. No historical splits are regenerated here. A stronger end-to-end methodology would independently version unpruned, feature-only splits before cleaning/selection; recover missing columns through verified row provenance, never a positional join to raw CSV.

## Semantic audit

References:

- [UNSW dataset authors](https://research.unsw.edu.au/projects/unsw-nb15-dataset): Argus/Bro extraction and official feature dictionary provenance.
- [CICFlowMeter authors](https://www.unb.ca/cic/research/applications.html): bidirectional flows with first-packet forward direction.
- [CICFlowMeter implementation](https://github.com/ahlashkari/CICFlowMeter/blob/master/src/main/java/cic/cs/unb/ca/jnetpcap/BasicFlow.java): packet/byte/interval feature implementation. The current upstream implementation is supporting evidence, not proof of the exact historical extractor build.

The checked-in candidate JSON is the versioned semantic decision record. Semantics gate meaning, direction, physical units, computation and flow aggregation **before** statistical scores. Distribution similarity cannot establish meaning. The conservative accepted pairs are:

| Canonical | UNSW | CICIDS | Conversion / availability |
|---|---|---|---|
| flow_duration | dur | flow_duration | seconds / microseconds; CICIDS divided by 1e6 |
| forward_packets | spkts | total_fwd_packets | packet counts; identity |
| backward_packets | dpkts | total_backward_packets | packet counts; CICIDS column absent |

These are compatible flow-level quantities, not a claim that extractor timeout policies or connection segmentation are identical. Directional counts assume source corresponds to forward. Capture-level flow direction cannot be reconstructed from the cleaned files, which lack endpoint identifiers. This assumption must accompany reporting.

Only **two** defensible, available pairs pass this initial audit. This is intentionally less than 10–16. Unresolved byte/header/payload accounting rejects `sbytes/dbytes` and mean-size pairs. Window summary versus initial advertised window is unverified. `sintpkt/dintpkt` versus directional mean IAT remains rejected until the exact interval estimator, retransmission treatment and singleton convention are validated. Jitter is not automatically IAT standard deviation. Load, loss, TTL and HTTP-body suggestions fail quantity, direction or aggregation checks. A rejected mapping receives no final score, even if its distributions happen to match. A semantically valid but absent column receives an explicit missing-columns status.

Do not manufacture derived duplicates to meet K. Expanding the valid pool requires new definition evidence and possibly new independently versioned unpruned training data, followed by rerunning the fixed protocol.

## Transformations and quality

Convert units first. Negative values (all V1 quantities are nonnegative), infinities, missing and nonnumeric values become missing; record quality **before imputation**. Missing ratio counts NaN/nonnumeric; invalid ratio counts infinities/negatives; finite-value ratio counts usable observations. Fit one median on source training values in physical units and use it in both domains, preserving source row/label alignment. Apply the same configured `log1p` to both sides. Save the source median for later aligned preprocessing.

For each domain, constant means at most one distinct usable value. Near-zero variance means constant, a dominant value fraction >= 0.995, or transformed variance <= 1e-12. Domain quality is `finite_ratio × penalty`, with penalty 0 for constant, 0.1 for near-constant, otherwise 1. Pair quality is the minimum of the two domain qualities. Constant/all-invalid pairs are removed. All thresholds are fixed in the protocol config.

Independent StandardScaler outputs erase physical shifts in location and scale and can falsely make different domain measurements look equivalent. No StandardScaler (or other domain-specific fitted normalization) is applied here. Any downstream model scaling is separate from selection.

## Metrics and stability

For transformed training samples X and Y:

- `KS_similarity = 1 - KS_statistic(X, Y)`.
- `W_norm = Wasserstein(X, Y) / (IQR(concatenate(X, Y)) + 1e-9)`.
- `W_similarity = exp(-W_norm)`.
- `MI_i = mutual_info_classif(source_feature_i, source_binary_label)`; `MI_similarity_i = MI_i / max(MI)` (all zero when max MI is zero).
- `score = 0.40 KS_similarity + 0.35 W_similarity + 0.15 MI_similarity + 0.10 quality_score`.

MI uses one deterministic source training sample, up to 100,000 rows, seed 42. Packet counts are marked discrete and category-encoded for MI; continuous features use sklearn's estimator. MI normalizes across eligible available candidates. Quality and source-fit medians use the complete projected training columns.

Distribution scoring uses five uniform samples **without replacement**, up to 100,000 rows per domain, seeds 42–46. This is repeated-subsample stability, not a bootstrap confidence interval. Samples use sorted parquet part order; every candidate uses the same rows within a seed and domain. Population standard deviations (`ddof=0`) describe variation across seeds. MI and quality stay fixed, isolating distribution-sampling variability. Reports include score/KS/W-similarity mean and standard deviation; `wasserstein_mean/std` refer to similarity, while `W_normalized_mean` and per-seed `W_normalized` expose distance.

## Redundancy and subsets

For candidate i and already selected j:

`redundancy_i = max_j mean_seeds((abs(Spearman_source(i,j)) + abs(Spearman_target(i,j))) / 2)`.

`selection_gain_i = score_mean_i - 0.20 * redundancy_i`.

Greedily recompute gains after each addition. The first feature has redundancy zero. Ties prefer lower score standard deviation, then canonical name. Undefined correlations conservatively count as 1. All Top-K outputs are nested prefixes of this selection order; `rank` is greedy selection rank, not score-only rank. Spearman captures monotonic dependence without requiring linear relations in skewed data.

Every requested K (10,12,14,16) gets a CSV and JSON. If fewer candidates survive, CSV contains only available pairs and JSON declares `requested_k`, `actual_k` and `status: insufficient_eligible_pairs`. These are **not completed Top-K experiments**. Each feature contains mapping, rationale, conversion, similarities, quality, mean score, standard deviation, redundancy, gain and rank. The console states the shortfall. `feature_pair_scores.csv` also includes rejection reasons and per-domain quality metrics. `seed_scores.csv` preserves per-seed metrics. `manifest.json` records protocol, complete candidate decisions, projected-column data hashes, file order, row counts and package versions.

## Running and separate common-feature mode

```bash
PYTHONPATH=src .venv/bin/python -m features.feature_pair_scoring \
  --config configs/feature_selection_v1.json
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests \
  -p test_feature_pair_scoring.py -v
```

Config paths are resolved relative to the config's project root (parent of its directory). To run CICIDS → UNSW, copy the protocol within `configs`, set `source_domain` to `cicids`, and use separate output/config-output directories. The source domain is the only label provider in either direction. Run separate directions to separate locations to preserve both artifacts.

The optional `features.common_features.transform_common_features(frame, artifact, domain)` consumes a generated JSON artifact and returns canonical columns in selection order using saved conversions, logs and source medians. It never needs labels or fits on new data. It refuses insufficient-K artifacts unless `allow_shortfall=True` is explicitly supplied. It deliberately does not train or change a model. Existing HDA remains the full heterogeneous comparison arm; future aligned models can explicitly consume this new mode.

## Leakage rules and validation

Target labels are evaluation-only. The parquet reader projects approved candidate columns, adding `label` only for the source. It restricts input paths to named raw `splits/{domain}_train` directories. Labels, attack categories, IDs and standardized `features` vectors are forbidden candidate inputs. Neither validation nor test rows are sampled. A path convention is an input guard, not proof of arbitrary user-supplied data provenance.

Do not pick K, weights, mappings, seeds or quality thresholds using target AUC/AP. Report all feasible preregistered K values; choose any model hyperparameters under a separately specified source-validation protocol. Tests compare full scores and rankings with target labels present, absent and poisoned; also cover reverse direction, parquet projection, determinism, conversion, KS/Wasserstein, semantic and availability gates, invalid/constant data, redundancy and opt-in aligned transformation.

## Exact local training schemas

unsw:

```text
proto, state, dur, sbytes, dbytes, sttl, dttl, sloss, dloss, service, sload, dload, spkts, dpkts, swin, dwin, smeansz, dmeansz, trans_depth, res_bdy_len, sjit, djit, sintpkt, dintpkt, tcprtt, is_sm_ips_ports, ct_state_ttl, ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd, ct_srv_src, ct_srv_dst, ct_dst_ltm, ct_src__ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, attack_cat, label
```

cicids:

```text
destination_port, flow_duration, total_fwd_packets, total_length_of_fwd_packets, fwd_packet_length_max, fwd_packet_length_min, fwd_packet_length_mean, bwd_packet_length_max, bwd_packet_length_min, flow_bytes_s, flow_packets_s, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min, fwd_iat_mean, fwd_iat_std, fwd_iat_min, bwd_iat_total, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min, fwd_psh_flags, bwd_packets_s, min_packet_length, max_packet_length, packet_length_mean, packet_length_variance, fin_flag_count, psh_flag_count, ack_flag_count, urg_flag_count, down_up_ratio, fwd_avg_bytes_bulk, fwd_avg_packets_bulk, bwd_avg_bytes_bulk, bwd_avg_packets_bulk, init_win_bytes_forward, init_win_bytes_backward, min_seg_size_forward, active_mean, active_std, active_max, active_min, idle_std, label
```
