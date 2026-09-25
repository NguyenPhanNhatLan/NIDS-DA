# Cấu trúc source code

Source code được tổ chức theo trách nhiệm, không theo tên tuần hoặc câu hỏi
nghiên cứu.

```text
src/
├── data_processing/   Đọc và kiểm tra dữ liệu đã map
├── preprocessing/     Ánh xạ schema riêng của từng dataset
├── features/          Fit imputer/scaler và tạo feature dùng để train
├── training/          Chia train/validation/test và chống duplicate leakage
├── models/            Kiến trúc model, loss và Optuna tuning
├── evaluation/        Metric và kiểm tra an toàn dùng chung
├── experiments/       Các pipeline thực nghiệm có thể chạy bằng CLI
└── reporting/         Tạo bảng và biểu đồ từ artifact kết quả
```

## File nên sửa theo mục đích

| Mục đích | File |
|---|---|
| Thay kiến trúc MLP baseline | `models/baseline.py` |
| Thay MLP, Adam, RBF-MMD hoặc MMD loss | `models/domain_adaptation.py` |
| Thay không gian 16 common features | `preprocessing/common_schema.py` |
| Thay cách chia train/validation/test | `training/data_split.py` |
| Thay protocol source-only nhiều seed | `experiments/source_only_transfer.py` |
| Thay protocol MMD hoặc lambda | `experiments/mmd_adaptation.py` |
| Thay metric báo cáo | `evaluation/metrics.py` |
| Thay kiểm tra seed/test leakage dùng chung | `evaluation/protocol.py` |
| Thay cách đọc và kiểm tra dataset đã map | `data_processing/datasets.py` |
| Thay biểu đồ kết quả MMD | `reporting/mmd_results.py` |

## Lệnh chính

```bash
# Chia toàn bộ mapped data theo chunk (mặc định ghi vào data/splits_full)
PYTHONPATH=src .venv/bin/python -m training.data_split \
  --input-file data/processed/cicids2017_mapped.csv --prefix cicids
PYTHONPATH=src .venv/bin/python -m training.data_split \
  --input-file data/processed/unsw_nb15_mapped.csv --prefix unsw

# Source-only transfer
PYTHONPATH=src .venv/bin/python -m experiments.source_only_transfer --help

# MMD adaptation
PYTHONPATH=src .venv/bin/python -m experiments.mmd_adaptation --help

# Vẽ lại kết quả MMD
PYTHONPATH=src .venv/bin/python -m reporting.mmd_results --help

# Optuna
PYTHONPATH=src .venv/bin/python -m models.tuning --help
```

Các experiment mặc định đọc `data/splits_full`. Splitter đọc 200.000 dòng
mỗi chunk, giữ vector feature trùng nhau trong cùng partition và tìm seed hash
cân bằng tỷ lệ nhãn trước khi ghi file.

## Pipeline cải tiến đã kiểm chứng trên mẫu độc lập

Hai profile theo chiều nằm trong `models/transfer_profiles.py`: bỏ TCP window
(cả hai chiều), bỏ thêm destination port ở CICIDS → UNSW, dùng QuantileTransformer
fit trên source train. MMD theo lớp/warm-up/bandwidth thích nghi được hỗ trợ trong
`models/domain_adaptation.py`; các mặc định cũ vẫn được giữ để tái lập baseline.

```bash
PYTHONPATH=src .venv/bin/python -m experiments.improved_mmd \
  --output-dir data/metadata/improved_mmd_full \
  --seeds 42,43,44,45,46 --paired-source-only --evaluate-test
```

Lệnh từ chối ghi đè output. Bỏ `--evaluate-test` nếu chỉ muốn huấn luyện và đánh giá
source validation. `--paired-source-only` đo riêng đóng góp của MMD trên cùng đầu vào.

Mẫu xác nhận cho thấy AP tăng so với baseline cũ ở 5/5 seed trong cả hai chiều;
đóng góp riêng của MMD chưa ổn định 4/5 seed. Profile đã được chọn bằng nhãn
target-development, không phải quy trình chọn model hoàn toàn không giám sát.
Chưa chạy benchmark full-data; xem `data/metadata/mmd_improvement_search/IMPROVEMENT_REPORT.md`
để biết metric, hạn chế recall/FPR và giao thức lấy mẫu.

## Theo đề cương: UDA chính + mở rộng 1%/5%/10% nhãn target

Dùng `experiments.proposal_suite` cho giao thức mới. Phần UDA cố định common features,
source-fit preprocessing và cấu hình; CORAL là đối chứng. Phần ít nhãn tách riêng,
tính cả nhãn validation vào ngân sách và so sánh fine-tune có/không MMD từ cùng
source checkpoint. Các profile target-tuned của vòng thăm dò trước không được dùng
trong bảng UDA chính.

```bash
PYTHONPATH=src .venv/bin/python -m experiments.proposal_suite \
  --split-dir data/proposal_pilot \
  --output-dir data/metadata/proposal_run_new --phase both
PYTHONPATH=src .venv/bin/python -m reporting.proposal_results \
  data/metadata/proposal_run_new
```

Chi tiết budget, threshold và giới hạn lấy mẫu: `experiments/PROPOSAL_PROTOCOL.md`.
Mọi model được lưu trước khi đọc test. Không ghi đè kết quả cũ.
