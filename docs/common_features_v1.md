# Common features v1

| Canonical feature | UNSW | CICIDS | Direction | Transformation |
|---|---|---|---|---|
| flow_duration | dur | flow_duration | bidirectional | CICIDS µs → s |
| fwd_packets | spkts | total_fwd_packets | forward | numeric cast |
| fwd_bytes | sbytes | total_length_of_fwd_packets | forward | numeric cast |
| fwd_packet_length_mean | smeansz | fwd_packet_length_mean | forward | numeric cast |
| fwd_iat_mean | sintpkt | fwd_iat_mean | forward | UNSW ms → s; CICIDS µs → s |
| bwd_iat_mean | dintpkt | bwd_iat_mean | backward | UNSW ms → s; CICIDS µs → s |
| fwd_iat_std | sjit | fwd_iat_std | forward | UNSW ms → s; CICIDS µs → s |
| bwd_iat_std | djit | bwd_iat_std | backward | UNSW ms → s; CICIDS µs → s |
| fwd_window_bytes | swin | init_win_bytes_forward | forward | numeric cast |
| bwd_window_bytes | dwin | init_win_bytes_backward | backward | numeric cast |

All features receive numeric casting, nonfinite values become null, and nulls are filled with domain-specific medians fitted on training rows only. No log transform is applied. Output input dimension: **10**. The generated Parquet files contain an ordered 10-value `features` vector and `label`; the canonical order is the table order. Mappings are semantic and directional, not claims of strict extractor equivalence.
