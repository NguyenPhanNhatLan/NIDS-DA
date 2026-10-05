# Tái tạo mục 3.2 từ Spark

`src/bigdata/pipeline.py` ghi thống kê của dữ liệu thực tế vào manifest sau khi hoàn thành tất cả các bước. `src/analytics/thesis_dataset_section.py` đọc manifest, đối chiếu tổng số dòng với các split và xuất nội dung mục 3.2; không huấn luyện mô hình.

Chạy từ thư mục gốc, dùng Java 17 và cùng Python cho Spark driver/workers:

```bash
export JAVA_HOME="$(/usr/libexec/java_home -v 17)"
export PYTHONPATH="$PWD/src"
export PYSPARK_PYTHON="$PWD/.venv/bin/python"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
.venv/bin/python -m bigdata.pipeline \
  --unsw-csv data/raw/UNSW-NB15.csv \
  --cicids-csv data/raw/CICIDS2017.csv \
  --work-root data/bigdata/thesis_20261005 \
  --feature-root data/features/thesis_20261005 \
  --model-root models/thesis_20261005
.venv/bin/python -m analytics.thesis_dataset_section \
  --manifest data/bigdata/thesis_20261005/manifest.json \
  --output docs/thesis_section_3_2.md
```

Đây là các đường dẫn của lần chạy hiện tại. Lần tiếp theo phải chọn ba output root mới và tên file báo cáo mới; pipeline từ chối ghi đè. Dữ liệu/preprocessor mới chưa thay thế các canonical roots `proposal_v2`, vì checkpoint cũ phải giữ cùng phiên bản tiền xử lý đã dùng để huấn luyện.

## Ý nghĩa bảng

- Rows, benign_rows, attack_rows: dữ liệu sau loại trùng toàn bộ dòng đã ingest, trước split.
- attack_ratio: attack_rows / rows, dạng tỷ lệ 0–1 trong manifest, chuyển sang phần trăm khi trình bày.
- input_rows, input_benign_rows, input_attack_rows: số liệu đầu vào trước loại trùng.
- original_feature_count: số cột không phải nhãn sau chuẩn hóa header/hợp nhất alias, trước chọn 5 feature; vẫn bao gồm identifier nếu CSV có.
- UNSW loại `label` và `attack_cat` khi đếm feature. CICIDS loại `label`; `Fwd Header Length.1` được hợp nhất với `Fwd Header Length` sau kiểm tra giá trị nhất quán.
- Không dùng số feature của các notebook HDA cũ hoặc số dòng trong báo cáo tuần trước.
- `*.provenance.json` cạnh bản thảo ghi đường dẫn và SHA-256 của manifest đã sử dụng.
