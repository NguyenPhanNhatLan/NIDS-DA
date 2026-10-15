# Kiểm tra canonical pipeline & kết quả — 15/10/2026

Phạm vi: checklist 06:30–07:30 (khung công việc do người dùng cung cấp, không phải thời gian thực thi đo được). Kiểm tra artifacts hiện tại; không chạy lại training, sửa kết quả hoặc mở final test.

Kết luận: **đủ các hạng mục development được yêu cầu; không phát hiện phần thiếu trong manifest, 40 runs, 5 seeds, aggregate và diagnostic JSON.**

| Hạng mục | Kết quả kiểm tra |
| --- | --- |
| Manifest canonical | Frozen, quality gate passed; checksum manifest và hashes toàn bộ 39 frozen files hợp lệ; inventory Parquet khớp. |
| Dữ liệu đầu vào | Số dòng common và prepared khớp ở cả hai chiều, mọi domain/split. |
| Matrix | Đủ 40 cặp result JSON/checkpoint: 2 chiều × 4 methods × 5 seeds (42–46). Checkpoint đọc được; identity, feature schema và revision khớp. |
| Aggregate | `results/canonical/aggregate/summary.json` có 40 per-run, 8 nhóm summary; bằng chính xác kết quả tính lại qua aggregator hiện tại. Aggregator kiểm tra config, prepared-data provenance, source-checkpoint reference và post-hoc diagnostic association. |
| Post-hoc diagnostic JSON | Đủ 30/30: 2 chiều × 3 adaptation methods × 5 seeds. |
| Directional diagnostic JSON | Đủ 10/10, đúng chiều/seed/revision. |
| Intrinsic diagnostic JSON | Đủ 5/5, đúng seed/revision; đủ 5 `samples.npz`, SHA-256 khớp JSON. |

Methods: `source_only`, `marginal_mmd`, `mk_mmd`, `class_aware_mmd`. Hai chiều: `unsw_to_cicids`, `cicids_to_unsw`. Các extension/funnel không được tính vào 40 canonical runs.

## Phần còn thiếu ở giai đoạn tiếp theo

- Chưa có `results/canonical/development_lock.json` và checksum lock.
- Chưa có kết quả final test (0 JSON trong `final_target_test`). Đây là stage riêng sau khi chốt development, không phải thiếu runs development.
- Lần kiểm tra này chưa replay source-validation AP/threshold từ 40 checkpoints. Bước `lock-development` bắt buộc thực hiện replay trước final test; sự hiện diện và identity của checkpoints không thay thế replay.

Theo contract hiện tại, bước tiếp theo sau khi chốt development là `PYTHONPATH=src .venv/bin/python -m experiments.proposal_pipeline --stage lock-development`; chỉ khi lock thành công mới chạy `--stage final-test`.

Chi tiết có cấu trúc: `docs/canonical_pipeline_audit_20261015.json`.
