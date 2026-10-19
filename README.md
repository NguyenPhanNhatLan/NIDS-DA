# Cross-domain NIDS

Thực nghiệm phát hiện xâm nhập giữa UNSW-NB15 và CICIDS2017 bằng source-only MLP,
MMD, MK-MMD và class-aware MMD. Hai miền dùng năm đặc trưng chung:
`flow_duration`, `fwd_packets`, `bwd_packets`, `fwd_bytes`, `bwd_bytes`.

Ma trận canonical gồm **40 runs = 2 chiều × 4 methods × 5 seeds (42–46)**,
khai báo trong `configs/proposal_suite_v2.json`. Các cấu hình development bổ sung
được chạy riêng và không được tính vào ma trận này.

## Cài đặt từ source

Dùng Python ≥ 3.11. Phần xử lý Spark cần Java 17 và PySpark.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[analytics,bigdata]'
export PYTHONPATH="$PWD/src"
```

Chạy lệnh từ thư mục gốc repository. Repository không kèm dữ liệu gốc, prepared
data hoặc checkpoints; bạn cần cung cấp dữ liệu và chạy prepare, hoặc sử dụng
workspace có revision canonical đã được tạo hợp lệ. Manifest khóa đường dẫn và
hashes artifacts: không sửa manifest/checksum để ghép checkpoints từ revision khác.

## Chuẩn bị dữ liệu

```text
Raw CSV → Spark ingestion/cleaning/profiling → deterministic splitting
→ five-feature harmonization → canonical audits and preprocessing → freeze
```

Trên một workspace mới, đặt CSV trong `data/raw/` rồi tạo upstream:

```bash
unset SPARK_HOME
export PYSPARK_PYTHON="$PWD/.venv/bin/python"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"

python -m bigdata.pipeline \
  --unsw-csv data/raw/UNSW-NB15.csv \
  --cicids-csv data/raw/CICIDS2017.csv \
  --work-root data/bigdata/thesis_20261005 \
  --feature-root data/features/upstream_20261005 \
  --model-root models/upstream_20261005

python -m experiments.proposal_pipeline --stage prepare
python -m experiments.proposal_pipeline --stage verify
```

Các lệnh tạo dữ liệu từ chối output đã tồn tại. Pipeline canonical đọc upstream
`data/bigdata/thesis_20261005/`; nó audit và tạo prepared data riêng, không dùng
bản preprocessing xấp xỉ của Spark làm đầu vào training canonical.

Canonical fit **median imputation → signed log1p → RobustScaler** chính xác bằng
NumPy/scikit-learn, chỉ trên source train của mỗi chiều. Bộ xử lý đó transform
mọi split của cả source và target. CICIDS duration đổi từ µs sang giây.
Nhóm vector giống nhau xuyên split được chuyển về holdout xa nhất
(`test > val > train`) rồi fit lại, theo limits chốt trong
`configs/proposal_data_policy.json`. Giữ flows và labels nhận từ upstream;
ambiguity được báo cáo. Upstream đã loại trùng trên toàn bộ raw row.

Manifest chính thức: `data/revisions/canonical/manifest.json` và `manifest.sha256`.

## Chạy ma trận canonical

Sau khi verify thành công, chạy từng stage theo thứ tự:

```bash
python -m experiments.proposal_pipeline --stage diagnostics
python -m experiments.proposal_pipeline --stage train
python -m experiments.proposal_pipeline --stage evaluation
python -m experiments.proposal_pipeline --stage aggregate
```

| Stage bổ sung | Công việc |
| --- | --- |
| `resume-train` | Replay-validation các cặp checkpoint/result hiện có rồi chạy phần thiếu; cặp thiếu một file sẽ dừng để kiểm tra. |
| `lock-development` | Kiểm tra đủ ma trận/aggregate, replay source-validation AP và threshold, khóa artifacts. |
| `final-test` | Đánh giá target test sau khi xác minh development lock. |

Chỉ chạy `lock-development` rồi `final-test` sau khi đã chốt thực nghiệm.
Sau lock, pipeline chặn retraining và ghi đè development artifacts.

Checkpoint chọn bằng **source-validation AP**; threshold chọn bằng
**source-validation F1**. Target train chỉ cung cấp features cho adaptation.
Target validation phục vụ development; target test dành cho đánh giá cuối.

## Class-aware MK-MMD development

Hai config độc lập, không dùng EMA/source-anchor filtering:

| Config | Kernel scales | Adaptation learning rate |
| --- | --- | --- |
| `proposal_class_aware_mkmmd_sep010.json` | 0.5, 1, 2 | 0.001 |
| `proposal_class_aware_mkmmd_sep010_lr0003.json` | 0.5, 1, 2 | 0.0003 |

Cả hai giữ confidence ≥0.90, λMMD=0.001, λSeparation=0.010 và margin=0.5.
Cần source-only checkpoints canonical trước khi chạy. Ví dụ một run:

```bash
python -m training.proposal_mmd \
  --direction unsw_to_cicids --seed 42 \
  --config configs/proposal_class_aware_mkmmd_sep010.json
```

Mỗi run ghi output theo tên config riêng và từ chối ghi đè. Sau khi đủ 5 seeds:

```bash
python -m evaluation.proposal_development_summary \
  --direction unsw_to_cicids \
  --config configs/proposal_class_aware_mkmmd_sep010.json
```

Các ablation và cấu hình development bổ sung nằm trong `configs/`;
outputs được lưu riêng theo tên config.
Các cấu hình thử nghiệm không được coi là tối ưu khi chưa đánh giá.

## Cấu trúc source

| Thư mục | Vai trò |
| --- | --- |
| `src/bigdata/` | Spark ingestion, cleaning, profiling, splitting và harmonization. |
| `src/features/` | Feature mapping, preprocessing và freeze revision; công cụ EDA/extraction. |
| `src/experiments/` | Pipeline, protocol và các entry points thực nghiệm. |
| `src/models/` | MLP và preprocessing classes dùng trong artifacts. |
| `src/training/` | Trainers, losses, batch streaming và kiểm tra revision. |
| `src/evaluation/` | Diagnostics, aggregate, development lock, final-test và CSV exporters. |
| `src/analytics/` | Tổng hợp thống kê dữ liệu cho luận văn. |
| `src/project_utils/` | SHA-256, ánh xạ source/target, chọn thiết bị và thống kê qua seeds. |

Tools EDA, PCAP extraction, classical baselines và BI có entry point riêng;
chúng không thuộc ma trận canonical. Tên module và preprocessing classes được
giữ để tương thích checkpoints/joblib hiện có.

```text
results/data_audit/canonical/    # Audits trước freeze
data/revisions/canonical/       # Common/prepared data và manifest
models/canonical/               # Preprocessors và checkpoints
results/canonical/              # Runs, diagnostics và aggregate
results/development/            # Báo cáo development bổ sung
```

Dữ liệu, model binaries, `.env` và outputs sinh ra được loại khỏi Git bằng
`.gitignore`. Chỉ source, configs, notebooks và tài liệu cần thiết được đưa lên repo.

## Kiểm tra

```bash
python -m experiments.proposal_pipeline --stage verify
python -m compileall -q src
```

`pr_auc` trong artifacts là average precision (AP). Thống kê qua seeds dùng
sample std và CI Student-t; ΔAP dùng paired seeds. Std của MMD subsamples hoặc
AUC folds không phải CI 95%. Các diagnostics dùng target labels phải được báo
là phân tích sau huấn luyện.

Từ điển đặc trưng nằm trong `configs/data_dictionary/`; CSV candidate/mapping
nằm trong `results/feature_mapping/`. `docs/` chỉ giữ
[semantic review](docs/semantic_feature_review_20261009.md), là artifact đã
được data manifest khóa theo đường dẫn và SHA-256.

## Vẽ biểu đồ

Một script duy nhất ở root, đọc diagnostic JSON đã lưu và kiểm tra development lock:

```bash
.venv/bin/python plot_results.py --seed 42
```

Xuất đường Precision–Recall và phân vị điểm Attack cho hai hướng vào
`results/analysis/figures/seed42_<timestamp>/`, gồm PNG 300 dpi, PDF và metadata.
Dùng `--seed 43` đến `46` cho các run khác; `--output <thư-mục-mới>` để chọn nơi lưu.
Hình dùng target-validation của một seed, không phải trung bình năm seeds.
Không cần train hoặc inference lại. Biểu đồ phân vị sử dụng năm phân vị đã lưu,
không phải ECDF đầy đủ.

## Config nào cần giữ

Sáu file thuộc canonical: `common_features_v2.json`, `proposal_suite_v2.json`,
`proposal_data_policy.json`, `proposal_mmd_v2.json`, `proposal_mkmmd_v2.json`,
`proposal_class_aware_v2.json`. Chúng được manifest hoặc development lock ràng buộc.
Tám config còn lại gắn với checkpoint/result development đã lưu; giữ tên và nội dung
để replay và kiểm tra provenance. `configs/data_dictionary/` là đầu vào mapping/EDA.

## Thành phần dùng chung

- `training.common`: seed, cross-entropy cân bằng lớp, giữ BatchNorm statistics
  và đo gradient norm; source-only và adaptation dùng chung.
- `models.baseline.BaselineMLP`: cùng kiến trúc MLP cho source-only và adaptation.
- `training.proposal_data.ParquetBatchStream`: đọc batch Parquet cho train/evaluate.
- `evaluation.baseline`: thu điểm, chọn threshold và tính metric dùng chung.
- `evaluation.proposal_results`: đọc và kiểm tra kết quả cho bảng CSV và script vẽ.
- `training.adaptation.kernel_mmd_loss`: một bộ tính RBF MMD cho cả single-kernel
  và multi-kernel; các hàm public cũ giữ vai trò wrapper để tương thích.
- `training.proposal_class_aware.prepare_class_batch`: tạo pseudo-label, mask và
  eligibility một lần mỗi batch; MMD, separation và batch statistics cùng dùng lại.
- `project_utils.statistics.seed_statistics`: mean, sample std và CI Student-t
  dùng chung cho aggregate, development summary và diagnostics.

Các biến thể tiếp tục dùng chung MLP, trainer, Parquet loader và config validation.
Refactor giữ nguyên nội dung các config có kết quả, cách chọn checkpoint và artifacts đã lưu.
