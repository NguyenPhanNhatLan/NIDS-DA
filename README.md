# Thesis: cross-domain NIDS — proposal_v2

Protocol chính duy nhất: `proposal_v2`, schema `configs/common_features_v2.json`.
Năm features, đúng thứ tự: `flow_duration`, `fwd_packets`, `bwd_packets`, `fwd_bytes`, `bwd_bytes`.

Bốn methods: **Source-only MLP → Marginal MMD → MK-MMD → Class-aware MMD**.
LR/XGBoost là baseline tham khảo phụ, không tham gia bảng so sánh bốn methods.
Code và artifacts HDA/v1, MCD, standalone Class-aware cũ được lưu trong `legacy/`.

```text
raw CSV → Spark ingest → cleaning/schema validation → profiling
        → split → harmonization → source-fitted preprocessing → Parquet
        → PyTorch baselines → domain shift → MMD → diagnostics
        → aggregate → final test → analytics Parquet → DuckDB → Tableau
```

**Spark handles data; PyTorch handles learning.** Luồng chạy chính thức bắt đầu
trực tiếp từ raw flow CSV qua `bigdata.pipeline`. Chạy các lệnh từ repo root.

Cài đặt (Python >=3.11, Java 17 cho dependency PySpark 3.5 của repo):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[bigdata,analytics,classical]'
export PYTHONPATH="$PWD/src"
export PYSPARK_PYTHON="$PWD/.venv/bin/python"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
```

Driver và workers cần dùng cùng Python minor version. Nếu shell đang đặt
`SPARK_HOME` sang một bản Spark khác với PySpark trong `.venv`, chạy
`unset SPARK_HOME` để dùng bản được cài trong môi trường này.

Raw CSV phải có header và các cột trong `configs/common_features_v2.json`.
UNSW dùng label nhị phân 0/1; CICIDS dùng category label (`BENIGN` là normal).
Các CSV chứa flow features đã được extractor tính sẵn. Spark kiểm tra/làm sạch
các giá trị, chọn 5 features và harmonize units. Dữ liệu không đóng gói trong Git.

Chạy canonical data pipeline (thay đường dẫn bằng raw files của bạn):

```bash
python -m bigdata.pipeline \
  --unsw-csv data/raw/UNSW-NB15.csv \
  --cicids-csv data/raw/CICIDS2017.csv
```

Mỗi tham số CSV nhận nhiều file; không cần merge bằng notebook trước:

```bash
python -m bigdata.pipeline \
  --unsw-csv /path/to/unsw_part1.csv /path/to/unsw_part2.csv \
  --cicids-csv /path/to/cicids_monday.csv /path/to/cicids_tuesday.csv
```

Hai ví dụ là hai cách truyền inputs cho cùng một lần chạy. Sau khi thành công:

```text
data/bigdata/proposal_v2/
    manifest.json
    ingested/
    clean/
    profiles/
    splits/
    common/
    class_counts/

data/features/proposal_v2/
    unsw_to_cicids/{unsw,cicids}_{train,val,test}/
    cicids_to_unsw/{unsw,cicids}_{train,val,test}/

models/proposal_v2/
    unsw_to_cicids/preprocessor.joblib
    cicids_to_unsw/preprocessor.joblib
```

`manifest.json` là completion marker, chỉ được tạo sau khi mọi stage và cả hai
hướng preprocessing hoàn thành. Split cố định hash 70/15/15, seed 42, trên raw
feature keys; các dòng có cùng keys nằm trong cùng split. Median imputation,
signed log1p và robust scaling fit **source train** theo từng hướng. Spark dùng
quantile xấp xỉ, mặc định `--relative-error 0.001`; statistics, Spark version và
`data_revision` được lưu trong artifacts. Không gom toàn bộ source train về RAM
driver để fit processor. PyTorch đọc Parquet thành batch 5 features để train.

Pipeline từ chối chạy nếu một trong ba output roots đã tồn tại, kể cả output của
lần chạy dở. Trước canonical final run, lưu dữ liệu/preprocessors/checkpoints/results
cũ cùng nhau vào `legacy/` theo revision, rồi dùng các canonical roots mới. Không
trộn checkpoint hoặc results cũ với preprocessing Spark vừa tạo.

Để kiểm tra data pipeline ở workspace riêng mà giữ artifacts hiện có:

```bash
python -m bigdata.pipeline \
  --unsw-csv data/raw/UNSW-NB15.csv \
  --cicids-csv data/raw/CICIDS2017.csv \
  --work-root data/spark_preview/proposal_v2/work \
  --feature-root data/spark_preview/proposal_v2/features \
  --model-root data/spark_preview/proposal_v2/models
```

Các training/evaluation CLI bên dưới vẫn đọc canonical roots, nên preview outputs
không tự động trở thành dữ liệu của thesis suite. Spark mặc định `local[2]`; đây là
cấu hình chạy local. Nhận định hiệu năng Big Data cần số đo từ lần chạy thật với
kích thước dữ liệu và cấu hình executor được ghi lại.

`notebooks/data_merging.ipynb`, `notebooks/unsw.ipynb`, `notebooks/cicids.ipynb` và
luồng cũ `features.splitting → features.common_features` dành cho khám phá/tham khảo.
Chúng không phải prerequisite hay canonical final run của data layer Spark.

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

Interface Research → BI nằm ở `src/analytics/build_dashboard_tables.py`:

```bash
python -m analytics.build_dashboard_tables \
  --output analytics_snapshot_01 --duckdb \
  --dataset-manifest data/bigdata/proposal_v2/manifest.json
```

Xuất 5 bảng Parquet và DuckDB snapshot cho Tableau; dùng một output directory mới
mỗi lần refresh. Kết nối Tableau cần cài DuckDB connector/driver. Xem
[analytics pipeline](docs/analytics_pipeline.md) để biết grain, phase filters,
provenance, connector và cách refresh snapshot.
