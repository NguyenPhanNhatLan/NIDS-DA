# Thesis: cross-domain NIDS — proposal_v2

Protocol chính duy nhất: `proposal_v2`, schema `configs/common_features_v2.json`.
Năm features, đúng thứ tự: `flow_duration`, `fwd_packets`, `bwd_packets`, `fwd_bytes`, `bwd_bytes`.

Bốn methods: **Source-only MLP → Marginal MMD → MK-MMD → Class-aware MMD**.
LR/XGBoost là baseline tham khảo phụ, không tham gia bảng so sánh bốn methods.
Code và artifacts HDA/v1, MCD, standalone Class-aware cũ được lưu trong `legacy/`.

```text
raw data → split → harmonization → preprocessing → baselines
         → domain shift → MMD → diagnostics → aggregate → final test
```

Cài đặt (Python >=3.11):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export PYTHONPATH="$PWD/src"
# Spark cho bước chuẩn bị dữ liệu, Java phù hợp với phiên bản Spark được cài:
pip install pyspark
# Tuỳ chọn baseline XGBoost:
pip install -e '.[classical]'
```

Chuẩn bị raw CSV bằng `notebooks/data_merging.ipynb`, sau đó chạy
`notebooks/unsw.ipynb` và `notebooks/cicids.ipynb` để chuẩn hóa tên cột, nhãn nhị phân
và xuất `data/processed/{unsw,cicids}_common5.parquet`. Những bước này chưa fit
imputer/scaler. Chạy notebook từ repo root. Dữ liệu không được đóng gói trong Git.

```bash
python -m features.splitting --dataset unsw
python -m features.splitting --dataset cicids
python -m features.common_features
for direction in unsw_to_cicids cicids_to_unsw; do
  python -m features.proposal_preprocessing --direction "$direction"
done
```

Split cố định hash 70/15/15, seed 42, ở `data/splits_v2/`; harmonization ra
`data/features/common_raw2/`. Preprocessing fit **source train** theo từng hướng,
ra `data/features/proposal_v2/` và `models/proposal_v2/`.
Các lệnh từ chối ghi đè outputs đã tồn tại.

Chạy development suite (2 hướng × 5 seeds × 4 methods):

```bash
for direction in unsw_to_cicids cicids_to_unsw; do
  python -m evaluation.proposal_domain_shift --direction "$direction"
  for seed in 42 43 44 45 46; do
    python -m experiments.proposal_source_only --direction "$direction" --seed "$seed"
    for config in proposal_mmd_v2 proposal_mkmmd_v2 proposal_class_aware_v2; do
      python -m training.proposal_mmd --direction "$direction" --seed "$seed" \
        --config "configs/$config.json"
      python -m evaluation.proposal_negative_transfer --direction "$direction" --seed "$seed" \
        --config "configs/$config.json"
    done
  done
done
python -m evaluation.proposal_aggregate
```

Có thể gọi `training.proposal_mkmmd` hoặc `training.proposal_class_aware` trực tiếp;
chúng dùng cùng trainer và artifact contract. Source-only phải chạy trước MMD.
Checkpoint chọn bằng **source-val AP**, threshold bằng **source-val F1**.
Target train chỉ đọc features; target val dùng diagnostics sau training.
Aggregate chỉ đọc development JSON, báo từng paired ΔAP, mean ± sample std và CI 95%.
Với 5 seeds, CI phản ánh biến thiên giữa seeds, phụ thuộc giả định phân phối của mean;
không đại diện cho bất định lấy mẫu toàn bộ dataset. Không báo significance test.

Chỉ sau khi khóa configs, checkpoint và threshold mới chạy held-out final test:

```bash
for direction in unsw_to_cicids cicids_to_unsw; do
  for seed in 42 43 44 45 46; do
    for method in source_only marginal_mmd mk_mmd class_aware_mmd; do
      python -m evaluation.proposal_final_test --direction "$direction" \
        --method "$method" --seed "$seed"
    done
  done
done
```

Final test kiểm tra schema, hashes, selection và tái lập source-val threshold trước
khi đọc target test; lưu riêng trong `results/proposal_v2/final_target_test/`.
Không dùng target test để chọn method/hyperparameters hoặc train lại.

Kiểm tra code bằng fixtures tổng hợp, không chạy thesis training:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Xem [protocol](docs/thesis_protocol.md), [manifest](docs/experiment_manifest.md),
[data dictionary](docs/data_dictionary/) và [mapping audit](docs/feature_mapping_audit.csv).
