# Cross-domain NIDS — canonical proposal_v2 pipeline

Một entry point chính thức: `python -m experiments.proposal_pipeline`.
Một manifest: `data/revisions/canonical/manifest.json`.

```text
Nguồn Spark có manifest → raw identity + split + mapping/quality audit
→ thống nhất precision-collision groups → source-train preprocessing → freeze
→ intrinsic/directional diagnostics → 4 methods × 2 directions × 5 seeds
→ development aggregate → lock-development → final-test riêng
```

Nguồn đầu vào hiện tại: `data/bigdata/thesis_20261005/`, gồm `manifest.json`,
`ingested/`, `splits/`, `common/`. Đây là upstream, không phải phiên bản training
thứ hai. Pipeline không phụ thuộc prepared roots cũ hoặc `data/features/common_raw`.

Năm inputs theo `configs/common_features_v2.json`: `flow_duration`, `fwd_packets`,
`bwd_packets`, `fwd_bytes`, `bwd_bytes`. CICIDS duration µs → seconds.
Giữ mọi flow/label, báo ambiguity; nhóm raw hoặc prepared float32 giống nhau
phải nằm trong một split. Hai chiều dùng cùng common splits và source-only fitted
median → signed log1p → RobustScaler. Manifest khóa hashes trước training.

## Cài đặt

```bash
cd /Users/thonph/Desktop/KLTN
source .venv/bin/activate
pip install -e '.[analytics,bigdata]'
export PYTHONPATH="$PWD/src"
unset SPARK_HOME
export PYSPARK_PYTHON="$PWD/.venv/bin/python"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
```

## Chạy theo thứ tự

```bash
python -m experiments.proposal_pipeline --stage prepare
python -m experiments.proposal_pipeline --stage verify
python -m experiments.proposal_pipeline --stage diagnostics
python -m experiments.proposal_pipeline --stage train
python -m experiments.proposal_pipeline --stage evaluation
python -m experiments.proposal_pipeline --stage aggregate
```

`prepare` gồm raw duplicate/label audit, replay splits, mapping/quality audit và
freeze. Split replay dùng Spark `xxhash64` trực tiếp, không có implementation NumPy
dự phòng; cần Java 17/PySpark trong `.venv`. Raw audit tính bucket một lần, ghi 64
Parquet partitions tạm rồi audit từng partition; không quét input 64 lần. So sánh
vẫn chính xác trên toàn bộ cột, hash chỉ dùng để phân vùng. Mỗi bước dừng khi audit lỗi.
Không tự ghi đè output của lần chạy trước hoặc build dở.

`relocation_audit.json` và `.csv` ghi counts/class counts ban đầu/cuối, các hướng
chuyển split, prevalence trước/sau và overlap cuối. Policy chốt trước prepare ở
`configs/proposal_data_policy.json`: tối đa 1% flow bị chuyển trên mỗi dataset và
thay đổi attack prevalence tối đa 0.005 (0.5 điểm phần trăm) ở mỗi split. Đây là
ngưỡng yêu cầu xem xét thiết kế, không phải ngưỡng thống kê; vượt ngưỡng sẽ dừng
freeze để xem lại split strategy. Không tự tăng ngưỡng sau khi thấy final-test.
Reports cũng ghi elapsed time và peak RSS của process; freeze còn nạp common data
vào NumPy, không được coi là streaming hoặc có tổng RAM cố định 1 GB.

`diagnostics` gồm intrinsic controls/sensitivity và hai chiều source-fitted.
Intrinsic dùng cùng flow, subsamples và bandwidth khi hoán đổi domain. AUC của cả
hai chế độ và controls dùng đủ 5 StratifiedGroupKFold folds, gom duplicate vectors
cùng fold và fit imputer/scaler/classifier bên trong train fold. Báo mean ± sample
std giữa folds, không gọi đó là CI 95%. MMD ± std là độ
biến thiên subsampling, **không phải CI 95%**. Byte mapping còn có điều kiện về
extractor semantics, nên luôn báo sensitivity bỏ hai byte features.

`train` chạy source-only rồi marginal MMD, MK-MMD, class-aware MMD, theo cả hai
chiều và seeds 42–46. Cả ba adaptation methods dùng một trainer. Checkpoint chọn
bằng source-val AP; threshold bằng source-val F1. Không dùng target test để tuning.
Class-aware có policy `missing_class_policy=skip_batch`: chỉ alignment khi cả hai
lớp có ít nhất 2 source và 2 accepted target samples; thiếu lớp thì bỏ toàn bộ
class-aware term. Không ép pseudo-label Attack. History ghi active classes,
eligible/aligned/skipped batches và acceptance theo predicted class (không theo
target ground truth).

`evaluation` chạy negative-transfer diagnostics trên validation sau training,
cho cả hai chiều, năm seeds và ba adaptation methods.

Chỉ sau khi chốt development mới chạy:

```bash
python -m experiments.proposal_pipeline --stage lock-development
python -m experiments.proposal_pipeline --stage final-test
```

`lock-development` bắt buộc đủ 40 kết quả và aggregate khớp, xác minh checkpoint,
config/provenance và replay source-validation AP/threshold cho toàn bộ matrix.
Nó khóa hashes aggregate, checkpoints, results và method configs. Final test bị
chặn nếu lock thiếu hoặc artifact đổi. Sau khi lock, không retrain/overwrite protocol;
không dùng target-test để chọn lại feature/hyperparameter. Source-validation replay
vẫn chạy trước khi đọc target-test labels.

## Output duy nhất

```text
results/data_audit/canonical/                 # reports trước freeze
data/revisions/canonical/                    # common, prepared, manifest, audits
models/canonical/                            # preprocessing và checkpoints
results/canonical/                           # diagnostics, training, aggregate, final test
```

Pipeline tự chọn manifest; không cần export root hoặc direction riêng.
Hash/inventory thay đổi sẽ bị chặn trước training/evaluation.
Các module audit/training là thành phần nội bộ của cùng pipeline.
Notebook và `legacy/` là lịch sử/EDA, không thuộc luồng chạy chính thức.

Chi tiết: [canonical revision](docs/canonical_data_revision.md),
[semantic mapping](docs/semantic_feature_review_20261009.md).

Nếu cần tự chạy synthetic tests, tách khỏi dữ liệu thật:

```bash
KLTN_DATA_MANIFEST='' python -m unittest discover -s tests -v
```

Integration test riêng cho Spark hash/bucket/membership và fixture null/NaN,
negative values, ±0, int/long/float/double:

```bash
KLTN_DATA_MANIFEST='' python -m unittest discover -s tests -p test_proposal_split_replay.py -v
```
