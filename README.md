# Cross-domain NIDS

Thực nghiệm phát hiện xâm nhập giữa UNSW-NB15 và CICIDS với 5 đặc trưng chung:
`flow_duration`, `fwd_packets`, `bwd_packets`, `fwd_bytes`, `bwd_bytes`.

Pipeline chính dùng revision `canonical`, gồm **40 runs = 2 chiều × 4 methods × 5 seeds (42–46)**.
Methods: `source_only`, `marginal_mmd`, `mk_mmd`, `class_aware_mmd`.
Ma trận được khai báo trong `configs/proposal_suite_v2.json`; đường dẫn config từng
method và các hằng số dùng chung nằm trong `src/experiments/proposal_protocol.py`.

## Cài đặt

Dùng Python ≥ 3.11. Bước chuẩn bị dữ liệu cần Java 17 và PySpark.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[analytics,bigdata]'
export PYTHONPATH="$PWD/src"
unset SPARK_HOME
export PYSPARK_PYTHON="$PWD/.venv/bin/python"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
```

Chạy các lệnh từ thư mục gốc repository. `src/spark_session.py` chọn Spark đi kèm
PySpark; khi dùng `spark-submit`, module giữ cấu hình gateway hiện có.

## Chạy pipeline

Entry point: `python -m experiments.proposal_pipeline --stage <stage>`.

```text
prepare → verify → diagnostics → train → evaluation → aggregate
                                                    ↓
                                    lock-development → final-test
```

| Stage | Công việc |
| --- | --- |
| `prepare` | Audit dữ liệu upstream, xử lý nhóm trùng xuyên split, fit preprocessing và freeze manifest. |
| `verify` | Xác minh checksum manifest, hashes frozen files và inventory Parquet. |
| `diagnostics` | Đo domain shift intrinsic và theo chiều, với controls/sensitivity. |
| `train` | Huấn luyện source-only và 3 adaptation methods cho toàn bộ ma trận. |
| `resume-train` | Tiếp tục train bị gián đoạn; replay-validation trước khi dùng lại run đã có. Cặp checkpoint/result thiếu một file sẽ dừng để kiểm tra. |
| `evaluation` | Tính post-hoc diagnostics trên target validation cho 30 adaptation runs. |
| `aggregate` | Tổng hợp 40 development runs, thống kê theo seeds và paired ΔAP. |
| `lock-development` | Kiểm tra đủ ma trận, aggregate, provenance; replay source-validation AP/threshold và khóa artifacts. |
| `final-test` | Đánh giá target test sau khi xác minh development lock. |

Ví dụ kiểm tra dữ liệu và tổng hợp các runs đã hoàn thành:

```bash
python -m experiments.proposal_pipeline --stage verify
python -m experiments.proposal_pipeline --stage aggregate
```

`prepare` từ chối output đã tồn tại. Khi train bị gián đoạn, dùng `resume-train`.
Sau `lock-development`, pipeline chặn retraining; chỉ chạy `final-test` khi đã chốt
features, configs, checkpoints và thresholds.

## Dữ liệu và lựa chọn mô hình

Upstream hiện tại là `data/bigdata/thesis_20261005/`. Manifest chính thức duy nhất:
`data/revisions/canonical/manifest.json`, kèm `manifest.sha256`.

Hai chiều dùng chung common splits và membership. Mỗi chiều fit median imputation
→ signed log1p → RobustScaler trên source train. CICIDS duration đổi từ µs sang giây.
Giữ flow, multiplicity và labels gốc; ambiguity được báo cáo. Nhóm feature giống nhau
xuyên split được chuyển về holdout xa nhất (`test > val > train`), theo giới hạn đã
chốt trong `configs/proposal_data_policy.json`.

Checkpoint chọn bằng source-validation AP; threshold chọn bằng source-validation F1.
Target train chỉ cung cấp features cho adaptation; target validation phục vụ development
diagnostics; target test dành cho stage cuối. Class-aware MMD bỏ alignment của cả batch
nếu một trong hai lớp không đủ 2 source và 2 accepted target samples.

Aggregate báo mean, sample std và CI Student-t qua 5 seeds; ΔAP dùng paired seeds.
Std của MMD subsamples và domain-AUC folds không phải CI 95%. Byte mapping còn phụ thuộc
extractor semantics, nên diagnostics có sensitivity bỏ byte features.

## Cấu trúc

| Thư mục | Nội dung |
| --- | --- |
| `src/bigdata/` | Ingest và xử lý dữ liệu upstream bằng Spark. |
| `src/features/` | Feature mapping, preprocessing, audit hỗ trợ và freeze revision. |
| `src/experiments/` | Điều phối pipeline và cấu hình protocol dùng chung. |
| `src/training/`, `src/models/` | Model, loss và trainers. |
| `src/evaluation/` | Audit, diagnostics, aggregate, lock, final-test và CSV exporters. |
| `src/analytics/`, `analytics/` | Tạo và lưu bảng phục vụ BI. |
| `notebooks/` | Phân tích, hình và các thử nghiệm bổ sung. |
| `docs/` | Data contract, semantic review và báo cáo kiểm tra. |

Outputs canonical:

```text
results/data_audit/canonical/    # audits trước freeze
data/revisions/canonical/       # common/prepared data và manifest
models/canonical/               # preprocessors và checkpoints
results/canonical/              # diagnostics, runs, aggregate và final-test
```

Các công cụ EDA, classical baselines, extension/funnel và BI không được tính vào
ma trận 40 runs. Tên `target_val` trong đường dẫn artifacts tương ứng target validation;
`pr_auc` trong kết quả là average precision (AP). Giữ các tên này để đọc artifacts đã lưu.

Chi tiết: [data contract](docs/canonical_data_revision.md),
[manifest](docs/experiment_manifest.md),
[semantic review](docs/semantic_feature_review_20261009.md),
[analytics](docs/analytics_pipeline.md),
[kiểm tra artifacts ngày 15/10/2026](docs/canonical_pipeline_audit_20261015.md).

Thử nghiệm ngưỡng pseudo-label và trọng số class-aware MMD:
[hướng dẫn development](docs/class_aware_development.md). Các config bổ sung giữ nguyên
canonical artifacts và dùng tên output riêng.
