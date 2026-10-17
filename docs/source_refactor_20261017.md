# Source refactor — 17/10/2026

Phạm vi: tinh gọn source để review và sử dụng từ GitHub, giữ nguyên dữ liệu,
checkpoints, đường dẫn artifacts, configs thí nghiệm và giao diện CLI đang dùng.

## Thay đổi

- Gom SHA-256, ánh xạ direction → source/target và chọn thiết bị vào
  `src/project_utils/`. Các module cũ vẫn export aliases để callers hiện có dùng được.
- Bỏ các hàm `parquet_files`, `read_batches`, `write_split` không còn được gọi trong
  `features.proposal_preprocessing`; giữ luồng preprocessing Spark đang được
  `bigdata.pipeline` sử dụng. Canonical freeze có writer riêng và không bị xóa.
- Bỏ imports không sử dụng trong batch loader, classical baselines, model và Spark split.
- Rút gọn MLP và single-RBF MMD, giữ thứ tự phép toán, tên parameters và cấu trúc
  state_dict. Giữ module/class/function paths của preprocessors để đọc joblib cũ.
- Dọn headings comment và khoảng trắng dư trong các module training/diagnostics.
- Cho phép resolve canonical CLI defaults khi clone chưa có manifest. Training và
  verify vẫn báo thiếu manifest; manifest override sai không được bỏ qua.
- Khai báo `spark_session` và `rq1` trong setuptools để hai module cấp cao này được
  đưa vào wheel; bổ sung hướng dẫn cài editable từ source trong README.
- Viết lại README với ingestion upstream, canonical stages, MK-MMD development,
  cấu trúc source và lệnh kiểm tra. Các standalone EDA/export/PCAP tools được giữ
  vì có entry point riêng, không thể suy ra chúng vô dụng chỉ vì không được import.

## Kiểm tra

- 26 unit tests đạt, gồm helpers, manifest behavior, losses, gradients và config.
- MLP output train/eval và MMD values/gradients bằng chính xác phiên bản HEAD
  trước refactor trên fixture cố định.
- Bảy CLI chính chạy `--help` trên bản copy source/configs không có data/models;
  thử training trên bản copy này vẫn bị chặn vì chưa prepare.
- Build wheel thành công, kiểm tra có project_utils, spark_session và rq1.
- Compile toàn bộ src và kiểm tra whitespace diff.

Không cần train lại vì refactor này. Kết quả canonical/development đã lưu vẫn là
artifacts của các runs cũ, không được tái tạo hay sửa trong bước dọn source.

## Tái sử dụng logic chung

- Single-RBF và MK-MMD dùng chung engine `kernel_mmd_loss`; giữ các API wrapper cũ.
- `ClassBatch` chứa pseudo-labels, mask và eligibility để chia sẻ giữa class-aware
  MMD, separation và history. Trainer chỉ chọn pseudo-label một lần mỗi batch.
- Thống kê seeds chuyển sang `project_utils.statistics`, không còn cần import
  module aggregate/trainer chỉ để tính mean/std/CI. Alias cũ vẫn được giữ.
- Confidence audit tái sử dụng hàm đếm pseudo-label của training.
- 30 tests đạt; so sánh trực tiếp trước/sau cho giá trị và gradient single/multi/
  class-aware MMD khớp chính xác, separation và batch statistics không đổi.
- Không thêm comment Python; cấu hình và cách gọi CLI giữ nguyên.
