# Semantic review before canonical freeze — 2026-10-09

The five columns have matching concepts and directions, but extractor equivalence
is not established by matching names. This review preserves the 5-feature protocol
and records uncertainty; it does not silently reinterpret existing byte values.

| Feature | Local rule | Verification | Remaining limitation |
|---|---|---|---|
| flow_duration | UNSW `dur` seconds; CICIDS `flow_duration` µs / 1,000,000 | Full raw-split/common multiset replay, including labels and multiplicities; all six splits match | Extractor timeout/flow segmentation need not match; 107 CICIDS negative raw durations became null before source-only imputation |
| fwd_packets | `spkts` / `total_fwd_packets` | Full mapping replay; nonnegative integer values | Source/destination orientation and flow segmentation are extractor dependent |
| bwd_packets | `dpkts` / `total_backward_packets` | Full mapping replay; nonnegative integer values | Same orientation/segmentation limitation |
| fwd_bytes | `sbytes` / `total_length_of_fwd_packets` | Full mapping replay confirms correct columns, no arithmetic change | CONDITIONAL: payload bytes vs transaction bytes/header inclusion not proven equivalent |
| bwd_bytes | `dbytes` / `total_length_of_bwd_packets` | Full mapping replay confirms correct columns, no arithmetic change | CONDITIONAL: same unresolved byte-accounting issue |

Primary evidence consulted:

- [UNSW dataset publisher](https://research.unsw.edu.au/projects/unsw-nb15-dataset):
  Argus, Bro and additional algorithms generated the features. The local CSV's
  exact extractor/configuration has not been independently identified.
- [CICIDS2017 publisher](https://www.unb.ca/cic/datasets/ids-2017.html): flows were
  generated using CICFlowMeter. The original generating commit is not pinned by
  the local dataset manifest.
- [CICFlowMeter BasicFlow.java](https://github.com/ahlashkari/CICFlowMeter/blob/master/src/main/java/cic/cs/unb/ca/jnetpcap/BasicFlow.java):
  forward/backward totals accumulate `getPayloadBytes()`, separately from header
  bytes; duration is last-seen minus start, with µs→s for rate calculations.
- [CICFlowMeter PacketReader.java](https://github.com/ahlashkari/CICFlowMeter/blob/master/src/main/java/cic/cs/unb/ca/jnetpcap/PacketReader.java):
  TCP/UDP payload and header lengths are recorded separately. This is evidence
  about the inspected implementation, not proof of the exact 2017 generating build.
- [Argus ra manual](https://openargus.org/oldsite/man/man1/ra.1.pdf): distinguishes
  source/destination transaction bytes from application-byte metrics. Thus matching
  `sbytes` with a payload accumulator requires further extractor evidence.

No header subtraction/addition is justified from these CSVs alone. Keep byte
mapping conditional in thesis claims and report intrinsic MMD sensitivity without
both byte features. Sensitivity without duration and sigma multipliers 0.5/2 is
also reported on the same sampled rows; none selects model hyperparameters.

Reduced-feature label conflicts are reported separately from raw duplicate flows.
Multiple original flows can share the same five measurements and have different
labels. The canonical policy retains their original labels and multiplicities and
keeps feature-key groups entirely within one split. No majority relabeling or
test-driven filtering is applied.
