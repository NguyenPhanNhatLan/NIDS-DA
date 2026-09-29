# Pseudo-label audit V2 / V5b

Chạy từ thư mục gốc repo:

```bash
PYTHONPATH=src .venv/bin/python -u -m evaluation.v5c_pseudo_audit --config configs/hda_v5c.json
```

Audit dùng V2 và V5b asymmetric seed 42 đã frozen, chỉ đọc features của
adaptation train; không đọc nhãn target hoặc development và không train model.
Ngưỡng mặc định là threshold source lưu trong checkpoint, đổi sang logit margin.

Báo cáo: `results/hda_v5c/audit/seed42.json`.
Pool đã lọc: `results/hda_v5c/audit/pools_seed42.pt` (normal, attack và row indices
0-based theo thứ tự file Parquet đã sắp xếp).

- `agreement_rate`: tỷ lệ dự đoán giống nhau trên toàn train.
- `cohen_kappa`: đồng thuận đã hiệu chỉnh theo tần suất dự đoán; null nếu không xác định.
- Ma trận 2x2: hàng V2, cột V5b; thứ tự Normal, Attack.
- Candidate Normal: V2 margin <= q02; Attack: q95 <= margin < q98.
- Pool giữ lại yêu cầu cả V2 và V5b dự đoán đúng lớp candidate theo threshold.
- `training_allowed`: gate heuristic, yêu cầu ít nhất 64 mẫu được giữ mỗi lớp và
  V5b xác nhận ít nhất 50% Attack candidates theo config hiện tại.
- `blocked_reasons`: lý do chưa đạt gate. Audit vẫn lưu kết quả khi gate không đạt.

Đồng thuận không phải accuracy: hai model có thể cùng sai. Gate này chỉ kiểm tra
pool có đủ mẫu và Attack có bị phủ nhận hàng loạt hay không; không bảo đảm chất
lượng nhãn hoặc hiệu quả V5c. Module training V5c chưa được triển khai.

Audit không ghi đè kết quả cũ. Muốn chạy cấu hình khác, sao chép config và đổi
`audit_dir`. Hash model/config/code/data và pool được lưu để phát hiện thay đổi.
