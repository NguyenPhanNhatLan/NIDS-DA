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

Tools EDA, PCAP extraction và classical baselines có entry point riêng;
chúng không thuộc ma trận canonical. Tên module và preprocessing classes được
giữ để tương thích checkpoints/joblib hiện có.

```text
results/data_audit/canonical/    # Audits trước freeze
data/revisions/canonical/       # Common/prepared data và manifest
models/canonical/               # Preprocessors và checkpoints
results/canonical/              # Kết quả gốc và lock, giữ nguyên đường dẫn
results/json/                   # JSON kết quả và audit
results/images/                 # PNG
results/tables/                 # CSV và bảng Markdown
results/arrays/                 # Mẫu NPZ
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
nằm trong `results/tables/feature_mapping__*.csv`. `docs/` chỉ giữ
[semantic review](docs/semantic_feature_review_20261009.md), là artifact đã
được data manifest khóa theo đường dẫn và SHA-256.

## Tìm kết quả ở đâu

- `results/images/`: tất cả hình PNG; hiện có hai hình seed 42 và bốn hình cơ chế.
- `results/json/`: kết quả và metadata chưa bị khóa, gom chung theo tên bộ chạy.
- `results/tables/`: CSV và bảng Markdown. File `mechanisms__table*.csv` là bảng
  chính; `mechanisms__appendix*.csv` là số liệu kiểm tra, không bắt buộc đưa vào bài.
- `results/arrays/`: bảy file NPZ phục vụ kiểm tra và lấy mẫu lại.

Các file trong `results/canonical/` và `results/data_audit/canonical/` là 46
artifact gốc được manifest/lock ràng buộc; giữ nguyên vị trí và nội dung.
Tên file gom dùng `__` để phân cách bộ chạy, chiều chuyển miền và seed.
Các bộ `development__*`, `canonical__extensions__*` và `canonical__funnel*`
là thử nghiệm phụ, không thuộc bộ ba phương pháp canonical để báo cáo chính.
Không cần đưa mọi output vào khóa luận. `src/` chỉ chứa mã nguồn.

Các lệnh audit vẫn nhận `--output` làm tên bộ chạy. Khi nằm trong `results/`,
file được gom theo loại sau khi audit hoàn tất; output ngoài `results/` được giữ
nguyên. `project_utils.result_files` dùng chung quy tắc này và đọc được các đường
kết quả cũ. Dữ liệu và chỉ số không được tính lại khi đổi cách lưu.

## Vẽ biểu đồ

Một script duy nhất ở root, đọc diagnostic JSON đã lưu và kiểm tra development lock:

```bash
.venv/bin/python plot_results.py --seed 42
```

Xuất đường Precision–Recall và phân vị điểm Attack cho hai hướng vào
`results/images/`, với tên `seed42__target_precision_recall.png` và
`seed42__target_attack_score_quantiles.png` (300 dpi). Chạy lại sẽ cập nhật hai ảnh.
Dùng `--seed 43` đến `46` cho các run khác; `--output <thư-mục>` để chọn nơi lưu.
Hình dùng target-validation của một seed, không phải trung bình năm seeds.
Không cần train hoặc inference lại. Biểu đồ phân vị sử dụng năm phân vị đã lưu,
không phải ECDF đầy đủ.

Hình ECDF năm đặc trưng chung, bản rộng 11 × 10 inch (3300 × 3000 px):

```bash
.venv/bin/python plot_results.py --section ecdf
```

Lưu `results/images/class_stratified_feature_ecdf.png`. Source là UNSW, target là
CICIDS; dùng toàn bộ canonical common validation, tách theo nhãn, biến đổi signed
log1p trước scaling. Mỗi đường ECDF chỉ dùng giá trị hữu hạn của đặc trưng tương ứng
(hai giá trị duration thiếu ở CICIDS Benign được loại riêng khỏi đường duration).
Hình này tính lại từ dữ liệu, không dùng số liệu lấy từ ảnh tham chiếu.

### Figure và bảng cho mục 4.5

Bốn figure nằm trong `results/images/mechanisms__*.png`; bảng và phụ lục nằm
trong `results/tables/mechanisms__*.csv`. Đọc `results/tables/mechanisms__tables.md`
để xem các bảng chính. Thông tin phép đo nằm trong
`results/json/mechanisms__audit_manifest.json`.
Vẽ lại figure từ số liệu này, không vẽ lại hai hình mục 4.4:

```bash
.venv/bin/python plot_results.py --section mechanisms
```

Khi cần đo lại, chọn một tên bộ kết quả mới:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.proposal_mechanism_audit \
  --output results/mechanisms_run2
.venv/bin/python plot_results.py --section mechanisms \
  --audit-dir results/mechanisms_run2
```

Audit này dùng đúng ba cấu hình canonical và năm seeds 42–46; không đọc target test,
không cập nhật trọng số. Gradient lấy đúng alignment loss của từng phương pháp.
Figure 3 đo classification với alignment loss thực tế của từng phương pháp tại
mô hình được chọn theo source-validation AP, trên 50 cặp batch training cố định.
Đây là phép đo tại một trạng thái mô hình của mỗi run. Muốn phân tích diễn biến
tối ưu, cần ghi cosine trong training ở nhiều bước/epoch; log gradient norm đơn
thuần không thay thế cho cosine và không tái dựng được lịch sử xung đột.
Batch có gradient bằng không hoặc không đủ hai lớp pseudo-label được ghi riêng;
cosine không xác định không được gán thành 0. BN shift là chẩn đoán, chưa có ablation
để kết luận tác động lên AP. Chi tiết mẫu và cách tổng hợp nằm trong `audit_manifest.json`.

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

## Thử riêng λ_MMD, một seed

Dùng source-only đã khóa làm điểm khởi tạo, cùng dữ liệu và trainer canonical.
Truyền λ cần thử bằng `--lambda-mmd`; tham số này bắt buộc và không có giá trị
mặc định. Mặc định còn lại là Marginal MMD, chiều UNSW → CICIDS, seed 42.

Thêm `--dry-run` để kiểm tra đầu vào mà không train. Script giữ nguyên batch size,
learning rate, tối đa 50 epoch và early stopping patience 7 từ cấu hình gốc;
chỉ thay λ. Model selection vẫn dùng source-validation AP, gồm epoch 0; ngưỡng
phân loại được chọn bằng source-validation Attack F1. Target validation chỉ được
đánh giá sau khi chọn mô hình; không đánh giá target test.

Checkpoint nằm trong `models/trials/`; kết quả và config hiệu lực nằm trong
`results/json/trial__*.json`, với tên chứa phương pháp, λ, chiều và seed.
Không sửa config canonical hoặc ghi đè kết quả đã khóa. Khi chạy lại cùng cấu hình,
dùng `--tag <tên-mới>` để giữ các lần thử riêng. Có thể đổi `--direction` sang
`cicids_to_unsw` hoặc `--method` sang `mk_mmd` / `class_aware_mmd` khi cần thử riêng.
Đây là thử nghiệm bổ sung sau khóa; không tự thay thế kết quả canonical.

## Thử Deep Embedded Validation (DEV)

DEV là cách **chấm điểm/chọn mô hình**, không thay classification loss hay MMD loss.
Module `evaluation.proposal_dev` trích latent 168 chiều của từng mô hình, học một
MLP phân biệt miền từ **source train và target train chỉ có features**, rồi ước lượng
trọng số `p_target(z) / p_source(z)` trên source validation có nhãn. Điểm DEV là
importance-weighted zero-one error cộng control variate theo You et al. (ICML 2019);
**điểm thấp hơn được ưu tiên**. Đây không phải ước lượng AP, Attack F1 hay loss có
trọng số lớp của training. Với dữ liệu mất cân bằng, tối thiểu hóa zero-one risk
không đồng nghĩa tối đa hóa AP.

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.proposal_dev \
  --direction both --seeds 42 --max-iter 300 --output-prefix dev_trial42
```

Lần thử seed 42 đã lưu tại:

- `results/tables/dev_trial42__runs.csv`: từng ứng viên, DEV risk, source error,
  domain-classifier accuracy/AUC, thống kê trọng số và effective sample size (ESS).
- `results/tables/dev_trial42__selection.csv`: ứng viên có raw DEV risk thấp nhất
  trong từng chiều/seed, kèm cờ chẩn đoán.
- `results/json/dev_trial42__manifest.json`: công thức, mẫu, tham số, hash và giới hạn.

Lần thử seed 42 hiện có ESS chỉ khoảng 1–40/10.000 mẫu; các domain MLP được
chọn chưa hội tụ và nhiều risk nằm ngoài [0,1]. Vì vậy chưa dùng thứ hạng này để
chọn λ hay thay kết luận canonical; đây là kết quả kiểm tra tính khả dụng của DEV.

Chọn tên `--output-prefix` mới khi chạy lại. Mặc định dùng 2.000 mẫu tự nhiên mỗi
miền để học bộ phân biệt miền và 10.000 mẫu source validation để tính risk; không
cân bằng Normal/Attack. Các mẫu cố định giữa ứng viên. Domain MLP có hidden layers
`(168,168,2)` và thử chín mức L2 như code tham khảo; chọn theo domain holdout accuracy
(80/20, stratified), giữ ứng viên đầu tiên khi hòa. Khác code tham khảo ở chỗ có
`StandardScaler`, chỉ fit trên phần train của bộ phân biệt miền. Không cập nhật
mạng MLP phát hiện xâm nhập. Mẫu domain train và source validation tách biệt.

Risk được giữ nguyên kể cả âm hoặc lớn hơn 1; không clip risk, không chuẩn hóa
trọng số về tổng bằng 1. Probability floor chỉ tránh chia cho 0 và được báo tỷ lệ
kích hoạt. Khi trọng số gần hằng số, đặt hệ số control variate bằng 0. ESS dưới
10% số mẫu được gắn cờ theo **heuristic kiểm tra của dự án**, không phải ngưỡng
lý thuyết của DEV. Cờ không làm đổi phép xếp hạng raw risk; ứng viên đứng đầu có
cờ không được xem là một lựa chọn đã được xác nhận đáng tin cậy. Thêm các cờ cho
bộ phân biệt miền chưa hội tụ và risk ngoài [0,1]. Các giả định covariate shift
trong không gian biểu diễn và support của target được source bao phủ chưa được
xác nhận tự động.

Pool mặc định chỉ gồm bốn mô hình **đã được chọn trước bằng source-validation AP**.
Vì vậy, đây là thử nghiệm xếp hạng có điều kiện trên pool đó, chưa phải thay toàn bộ
cách chọn epoch/λ bằng DEV. Không dùng nhãn target validation/test để chấm điểm,
không đổi canonical lock, checkpoint, config hay kết quả test đã lưu. Bước kiểm tra
tính toàn vẹn vẫn hash các artifact frozen, không giải mã nhãn target để tính risk.

Để thử chọn λ hoặc epoch, cần lưu các mô hình ứng viên của một thử nghiệm riêng.
Truyền `--candidates <file.json>` với danh sách như sau (đường dẫn tương đối tính từ
thư mục chứa JSON; các checkpoint phải có cùng data revision và preprocessing):

```json
[
  {"name": "mmd_lambda_001", "direction": "unsw_to_cicids", "seed": 42,
   "checkpoint": "models/dev_sweep/mmd_lambda_001_seed42.pt"},
  {"name": "mmd_lambda_010", "direction": "unsw_to_cicids", "seed": 42,
   "checkpoint": "models/dev_sweep/mmd_lambda_010_seed42.pt"}
]
```

Đây là định dạng minh họa; module không tự tạo các checkpoint còn thiếu. Xếp hạng
được thực hiện riêng trong mỗi chiều/seed; không trộn risk giữa hai chiều. Khi cần
chọn chung một λ qua nhiều seed, phải tổng hợp DEV theo cấu hình với cùng tập seed,
không lấy riêng seed có risk thấp nhất.

Nguồn: [DEV paper](https://proceedings.mlr.press/v97/you19a.html),
[reference implementation](https://github.com/thuml/Deep-Embedded-Validation/blob/master/dev.py).
