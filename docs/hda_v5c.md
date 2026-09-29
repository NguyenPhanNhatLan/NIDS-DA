# V5c: frozen V5b teacher + filtered pseudo labels

## Ablation lambda_PL = 0

Dùng `--config configs/hda_v5c_lambda_pl_0.json` cho cả bốn lệnh bên dưới.
Config này lưu artifacts riêng dưới `lambda_pl_0`. CE vẫn được tính và ghi log
để giữ cùng luồng lấy mẫu với bản 0.10, nhưng nhân 0 nên không đóng góp gradient.
Audit và filtered pools vẫn cần cho protocol calibration chung. Đây là tiếp tục
train từ V5b với loss nền, không phải đánh giá nguyên checkpoint V5b frozen.
Validation chấp nhận lambda_PL hữu hạn >= 0.

Sau thay đổi code, audit/checkpoint cũ có thể báo hash mismatch. Khi đó chạy lại
toàn bộ pipeline với các thư mục artifact mới; không sửa hash artifact cũ.

Chạy từ thư mục gốc repo, lần lượt bốn lệnh sau. Không thay đổi config/code
giữa các bước vì artifact được kiểm tra hash.

```bash
PYTHONPATH=src .venv/bin/python -u -m evaluation.v5c_pseudo_audit --config configs/hda_v5c.json
PYTHONPATH=src .venv/bin/python -u -m training.hda_v5c --config configs/hda_v5c.json
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5c --config configs/hda_v5c.json --stage fit
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5c --config configs/hda_v5c.json --stage report
```

Audit cũ không còn hợp lệ sau khi bổ sung pipeline. Config hiện dùng
`results/hda_v5c/audit_filtered_v1` để không ghi đè audit trước đó.
Nếu thư mục này cũng đã có artifact, chọn audit_dir mới trước bước đầu tiên.
Training chỉ tiếp tục khi audit có `training_allowed=true`.

## Training

Teacher là V5b asymmetric seed 42 frozen. Student dùng HDAV1Model và nạp adapter
từ chính checkpoint teacher. Chỉ adapter student được tối ưu; source tail,
classifier và teacher được freeze, ở eval mode.

Loss = hidden MMD + 0.05 Normal MMD + 0.02 Attack MMD + 0.10 ranking
+ 0.10 pseudo-label cross-entropy.

- Hidden MMD dùng source và natural target batch.
- Conditional MMD giữ pool V2 q02 / [q95, q98) như V5b gốc.
- Ranking so khớp margin student với teacher V5b frozen trên cùng target batch.
- CE dùng riêng pool đã audit: V2 tạo ứng viên quantile, V5b xác nhận lớp.
  Không yêu cầu V2 xác nhận theo threshold. Mỗi batch lấy 64 Normal + 64 Attack,
  có hoàn lại; CE lấy trung bình cân bằng hai lớp.
- Chỉ natural target batch cập nhật BN adapter; các batch cân bằng dùng BN eval.
- Pseudo pools cố định từ CICIDS adaptation-train, không đọc target labels.
- Kế thừa 10 epochs, batch size 256, Adam lr 0.001, weight decay 0.0001.
  Lưu epoch cuối, không chọn checkpoint bằng development.

Thiết bị tự chọn CUDA/MPS/CPU; thêm `--device cpu` vào lệnh train để ép CPU.

## Calibration và comparison

`--stage fit` không đọc development. Dùng UNSW validation có nhãn để chọn
threshold margin với FPR source tối đa 0.01. Fit hai affine calibrator riêng
`a > 0, b` cho V5b và V5c, ánh xạ median margin của cùng filtered Normal/Attack
pools lên median của hai lớp source validation. Đây là protocol calibration
riêng của thí nghiệm V5c; không ghi đè calibration V5b trước đó.
Nếu anchors collapse/đảo thứ tự, fit dừng với lỗi thay vì dùng development chỉnh ngưỡng.

`--stage report` mới đọc nhãn development, chấm cả hai model trên cùng split,
dùng calibrator đã freeze và cùng source threshold. Báo cáo AP (`pr_auc` = average
precision), ROC-AUC, F1, Recall, FPR và delta V5c trừ V5b. Có thêm metrics raw
tại ngưỡng source ban đầu để đối chiếu. Affine dương không thay đổi thứ hạng;
AP/ROC phản ánh ranking, F1/Recall/FPR phụ thuộc calibration và threshold.
FPR <= 0.01 trên source không bảo đảm FPR đó trên target.

Artifacts:

- `models/hda_v5c/v5c_seed42.pt`: adapter, history và provenance.
- `results/hda_v5c/calibration/seed42.json`: hai calibrator đã freeze.
- `results/hda_v5c/development/v5c_vs_v5b_seed42.json`: metrics và delta.

Không ghi đè artifact đã tồn tại. Muốn thử config khác, chọn bộ audit_dir,
checkpoint_dir, calibration_dir, result_dir mới trước khi audit. Đồng thuận
pseudo labels không phải ground-truth accuracy. Development là split khảo sát
hiện có, chưa phải untouched final holdout.
