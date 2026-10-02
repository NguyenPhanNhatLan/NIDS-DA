# Unpruned feature-equivalence audit V1

The local sources support reproducible **column recovery**, but not exact individual split recovery for every record: historical pruning discarded distinguishing fields, and identical retained-feature keys sometimes span saved splits. This experiment preserves every **provable** boundary and quarantines ambiguous groups. No positional join, new random split, nearest-value match, or label-based CICIDS assignment is used. Existing datasets, notebooks, model architectures and experiments were not overwritten.

## Result

| Stage | Count |
|---|---:|
| A. Candidates before semantic filtering | 19 |
| B. Passing semantics | 3 |
| C. Available in both unpruned schemas, irrespective of semantics | 19 |
| C. Passing semantics and available in both | 3 |
| D. Surviving quality filtering | 3 |
| E. Final eligible pool | 3 |

Greedy selection order, with the unchanged V1 weights, five seeds and 100,000 training rows per domain per seed:

| Rank | UNSW | CICIDS | Score mean | Score std |
|---|---|---|---:|---:|
| 1 | dur | flow_duration | 0.602154 | 0.000709 |
| 2 | dpkts | total_backward_packets | 0.582505 | 0.002055 |
| 3 | spkts | total_fwd_packets | 0.553540 | 0.000521 |

The 16 rejected decisions remain rejected. Restoring a column changes availability, not semantics. No new Top10/12/14/16 artifacts were attempted or generated; older shortfall artifacts from the previous experiment remain untouched. These scores are from different eligible training subsets and a different MI normalization pool, so changes from the pruned experiment should not be attributed solely to adding backward packets.

## Exact CICIDS preprocessing audit

Notebook cell indexes below are zero-based. The input has 79 columns: 78 measurements and a text label. Column normalization occurs before all filtering. Cells 4 and 11 deduplicate original complete rows and remove rows with nonfinite `flow_bytes_s` or `flow_packets_s`. They do not remove measurement columns.

Cell 12 has manually specified constant and near-constant lists. Their actual effect, confirmed against the saved schema:

- Constant removal (4): `bwd_psh_flags`, `bwd_urg_flags`, `fwd_avg_bulk_rate`, `bwd_avg_bulk_rate`.
- Near-constant removal (4): `fwd_urg_flags`, `cwe_flag_count`, `rst_flag_count`, `ece_flag_count`.
- No-op constant requests (4): `fwd_avg_bytes/bulk`, `fwd_avg_packets/bulk`, `bwd_avg_bytes/bulk`, `bwd_avg_packets/bulk`. Names had already been normalized to `_bulk`, so Spark did not remove these four columns. They survived in the old parquet.

Cells 14–15 estimate Pearson correlations from an unlabeled sample of up to 20,000 rows of the whole cleaned dataset, then greedily drop features at absolute correlation >0.95. The saved cell-15 output lists exactly 24 removals:

```text
total_backward_packets, total_length_of_bwd_packets,
fwd_packet_length_std, bwd_packet_length_mean, bwd_packet_length_std,
fwd_iat_total, fwd_iat_max, fwd_header_length, bwd_header_length,
fwd_packets_s, packet_length_std, syn_flag_count, average_packet_size,
avg_fwd_segment_size, avg_bwd_segment_size, fwd_header_length_1,
subflow_fwd_packets, subflow_fwd_bytes, subflow_bwd_packets,
subflow_bwd_bytes, act_data_pkt_fwd, idle_mean, idle_max, idle_min
```

There are no additional manual measurement-column removals in this notebook. The constant/near-constant lists themselves are manual lists, not a rerun of estimated thresholds. The 32 actual removals explain the difference between the original 78 and retained 46 measurement columns exactly. The audit reads historical notebook output and validates it against the saved schema; it does **not** fit another full-data correlation selector.

The notebook's exploratory log frame is separate from the exported cleaned frame. Binary-label generation and splitting later replace the text label. In the new audit representation, CICIDS labels are omitted entirely after historical row reconstruction.

## Regeneration and row provenance

Inputs are the original local `data/raw/CICIDS2017.csv` and `data/raw/UNSW-NB15.csv`. SHA-256 checksums for them, the notebook and every old split parquet appear in the provenance artifacts. The builder replays the historical row cleaning, preserving all original measurement columns and bypassing every column-pruning stage. For UNSW it also restores the manually removed raw columns; identifiers are retained as provenance context only and never enter scoring.

Reconstruction counts:

| Dataset | Raw rows | Original full-row deduplication | Historical row cleaning |
|---|---:|---:|---:|
| CICIDS | 2,830,743 | 2,522,362 | 2,520,798 |
| UNSW | 2,540,047 | 2,059,414 | 2,059,414 |

CICIDS loses 1,564 nonfinite-rate rows, exactly as the notebook recorded. The recorded local UNSW source count is used as observed; it is not replaced by a published dataset-size assumption.

Membership is recovered from a typed Spark struct containing **every retained old feature**, excluding `label`. For UNSW only, the historical `attack_cat` column participates in provenance because it was present in the saved source schema; it is removed from the new representation and never scored. CICIDS matching never reads or uses its old target labels.

The builder groups old keys with split membership and multiplicity, and compares their complete multiset with projected regenerated rows. It aborts on missing keys, extra keys, or count differences. Matching uses actual typed feature values, not just a hash; `_boundary_key_sha256` is a report identifier after the join. Distinct raw records sharing a pruned key are recoverable if the entire old group belongs to one split and counts agree. If a key occurs across splits, no individual assignment can be proved; all raw records in that group go to `quarantine/`, with possible splits recorded. None enters any recovered training, validation or test split.

Both raw/old feature-key multisets matched exactly **before** quarantine. The quarantine affects 2,737 CICIDS keys / 5,506 rows, and 8,526 UNSW keys / 33,076 rows.

| Dataset / split | Original rows | Verified recovered rows | Withheld as ambiguous |
|---|---:|---:|---:|
| CICIDS train | 1,764,877 | 1,762,420 | 2,457 |
| CICIDS validation | 377,526 | 376,056 | 1,470 |
| CICIDS test | 378,395 | 376,816 | 1,579 |
| UNSW train | 1,442,111 | 1,422,435 | 19,676 |
| UNSW validation | 308,648 | 301,818 | 6,830 |
| UNSW test | 308,655 | 302,085 | 6,570 |

These are verified subsets of the original boundaries, **not complete reconstructions of all original split rows**. Recovering the withheld rows requires an original row-to-split manifest or other identifying provenance. Their exclusion is based only on ambiguity, not target labels or model scores; nonetheless this changes the analyzed population and must be reported in the thesis.

A read-only diagnostic also compared saved CICIDS membership to the current `splitting.py` hash rule on the old retained schema: 1,172,278 assignments disagreed. Thus the current code is not evidence of how these particular saved splits were made. The builder treats the saved files as authoritative and never rehashes the expanded schema.

New paths:

```text
data/processed/common_feature_audit/v1/
  manifest.json
  _SUCCESS
  cicids_clean_unpruned/
    splits/cicids_train/
    splits/cicids_val/
    splits/cicids_test/
    quarantine/
  unsw_clean_unpruned/
    splits/unsw_train/
    splits/unsw_val/
    splits/unsw_test/
    quarantine/
```

`_SUCCESS` at the version root is written only after both datasets pass checks. The manifest status is `complete_with_quarantine`. The initial strict attempt aborted without output data; its empty directory is preserved separately as `v1_failed_strict_provenance_check`. Existing version directories cannot be overwritten by the builder.

Historical full-row deduplication included the raw text label, as the original notebook did. Reading it for faithful reconstruction is distinct from using target labels for selection: no target class values drive matching, quarantine, semantic decisions, sampling, imputation, scores or output. New CICIDS split and quarantine parquets contain no label column. Old validation/test features are read only to establish provenance; their distributions never enter scoring.

## Semantic investigation of the nine requested pairs

The full 19-row [audit table](../results/feature_mapping/unpruned_v1/semantic_audit.csv) records all requested fields. An empty computation-match cell means unverified, not accepted. Direction matches for directional quantities use the predeclared convention UNSW source = CICIDS forward and destination = backward; they do not claim extractor orientation is identical for every midstream capture. Aggregation match means a directional/bidirectional measurement within an observed flow record, not identical timeout segmentation across extractors.

Evidence is pinned to upstream commits, with file SHA-256 checksums in `implementation_evidence.json`. The exact historical extractor binaries/configurations and original packet captures are unavailable locally. Current source evidence can establish a mismatch or explain an estimator, but cannot certify undocumented historical settings.

| Pair | Decision | Meaning, direction, units, computation and aggregation |
|---|---|---|
| dur ↔ flow_duration | Retain accepted | Elapsed record duration; bidirectional; seconds versus microseconds; first-to-last observed time; flow record. Boundary policy remains extractor-dependent. |
| spkts ↔ total_fwd_packets | Retain accepted | Observed source/forward packet count; packets; directional list/counter count within a flow record. |
| dpkts ↔ total_backward_packets | Retain accepted, now available | Same count definition in the destination/backward direction. |
| sbytes ↔ total_length_of_fwd_packets | Remain rejected | Packet-input bytes versus transport-payload bytes; nominal bytes/direction/flow level agree, accounting does not. |
| dbytes ↔ total_length_of_bwd_packets | Remain rejected | Same accounting mismatch in the backward direction. |
| smeansz ↔ fwd_packet_length_mean | Remain rejected | Directional packet-byte mean versus payload-length mean; differing numerator and possible output precision. |
| dmeansz ↔ bwd_packet_length_mean | Remain rejected | Same mismatch in the backward direction. |
| sintpkt ↔ fwd_iat_mean | Remain unverified/rejected | Units convert (milliseconds/microseconds), but matching interval population and historical handling are unproved. |
| dintpkt ↔ bwd_iat_mean | Remain unverified/rejected | Same unresolved interval semantics in the backward direction. |

[CICFlowMeter PacketReader](https://github.com/ahlashkari/CICFlowMeter/blob/98a5ebad0df579cc8b43eedd3421b3ae87699901/src/main/java/cic/cs/unb/ca/jnetpcap/PacketReader.java#L141) supplies TCP/UDP payload length separately from header length. Its IPv4 timestamp path uses microseconds; this is not a blanket claim about every protocol path in every version.

[BasicFlow](https://github.com/ahlashkari/CICFlowMeter/blob/98a5ebad0df579cc8b43eedd3421b3ae87699901/src/main/java/cic/cs/unb/ca/jnetpcap/BasicFlow.java#L174) adds payload lengths to directional totals and size statistics. It adds packet observations without TCP-sequence deduplication. Directional IAT uses successive observed timestamps; its exporter emits zero for directions with at most one packet. Thus a payload-free control/singleton packet can have zero payload size despite a nonzero packet length. Retransmitted payload observations contribute again; these are not unique delivered bytes.

[ArgusModeler](https://github.com/openargus/argus/blob/fc7a7a844e3f119d70f69963f2aec3ac0fdc068a/argus/ArgusModeler.c#L3065) increments directional packet counters and accumulates `ArgusThisBytes`, initialized from input packet length before header processing. This is not the CICFlowMeter transport-payload counter. The precise encapsulation/capture configuration used to create UNSW is not established, so no fixed subtraction is proposed.

[Argus client routines](https://github.com/openargus/clients/blob/e12739eb478900a2d7465ac24e75fad0b0bccd05/common/argus_client.c#L13875) divide directional bytes by packets for mean size, with zero when no packets exist. The IAT accessor combines active and idle means weighted by their sample counts, converts to milliseconds, and defaults to zero when interval statistics are absent. These routines alone do not prove which retransmissions/out-of-order events populated historical UNSW interval statistics, how state carried across report boundaries, or which jitter options were enabled. Consequently even similar singleton defaults and convertible units do not justify accepting IAT pairs.

The [UNSW authors](https://research.unsw.edu.au/projects/unsw-nb15-dataset) identify Argus/Bro-based extraction; [CICFlowMeter's authors](https://www.unb.ca/cic/research/applications.html) describe first-packet direction. Existing accepted duration/count conventions are retained; no rejected flag was flipped.

## Freeze, scoring and reproduction

Curated decisions live in `configs/feature_pair_candidates_unpruned_v1.json`. `--freeze` inspects schemas, verifies accepted pairs' five checks, writes the table, and binds its hash plus candidate/config/evidence/dataset-manifest hashes. It performs no scoring. The scoring command checks those hashes before calling the existing scorer. Editing any frozen input fails closed.

The numerical protocol remains `0.40 KS + 0.35 normalized-Wasserstein similarity + 0.15 normalized source MI + 0.10 quality`, with redundancy penalty 0.20 and seeds 42–46. The separate config uses `top_k: []`. Target labels are absent and never projected. Only training paths in the completed versioned build are allowed by the audit wrapper. Projected values are sorted deterministically before sampling, making sample order independent of Spark's physical row ordering. The original scorer's default ordering is unchanged.

Outputs live in `results/feature_mapping/unpruned_v1/`: semantic table/freeze, implementation evidence, column-removal audit, dataset provenance copy, feature scores, seed scores, final eligible pool, scoring manifest and A–E summary. No model is trained.

To reproduce to a **new** version directory, update paths in a new config and freeze to a new audit directory:

```bash
PYSPARK_DRIVER_PYTHON="$PWD/.venv/bin/python" \
PYSPARK_PYTHON="$PWD/.venv/bin/python" SPARK_LOCAL_IP=127.0.0.1 \
spark-submit --driver-memory 6g src/features/unpruned_audit_data.py \
  --output data/processed/common_feature_audit/NEW_VERSION \
  --quarantine-ambiguous

# For the checked-in V1 configuration, freezing is already complete.
# The freeze command refuses to replace an existing freeze.
PYTHONPATH=src .venv/bin/python -m features.unpruned_feature_audit --freeze
PYTHONPATH=src .venv/bin/python -m features.unpruned_feature_audit
```

Omit `--quarantine-ambiguous` for strict all-row recovery: the build will abort on cross-split duplicate keys. The matching Spark runtime is installed separately from the Python project; this machine used Spark 4.2.0. Driver/worker Python must match when running Python-based Spark tests.

Tests cover exact matching after reordered input, retained-key multiplicity, null keys, ambiguity rejection/quarantine, missing/extra record rejection, notebook removals, immutable semantic decisions/freeze, deterministic projected ordering, and the original scoring/target-label tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_feature_pair_scoring.py -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_unpruned_feature_audit.py -v
PYTHONPATH=src SPARK_LOCAL_IP=127.0.0.1 \
PYSPARK_DRIVER_PYTHON="$PWD/.venv/bin/python" \
PYSPARK_PYTHON="$PWD/.venv/bin/python" \
spark-submit --driver-memory 2g tests/test_unpruned_audit_data.py
```
