# V5d: source-supervised private classifier

V5d dùng file riêng, không sửa architecture V1, V5b hoặc checkpoint frozen.
Target adapter deepcopy từ V5b asymmetric seed 42. Classifier deepcopy từ UNSW
baseline và bật gradient riêng. Source encoder được copy và frozen; source path
và target path dùng chung FC2/BN2 frozen trong V5d. Hai path dùng cùng classifier
mới. `train()` luôn giữ source encoder, shared BN và source dropout ở eval.

Loss khởi đầu:

`hidden MMD + 0.05 Normal MMD + 0.02 Attack MMD + 0.10 rank + 0.10 source CE`.

Source CE dùng `N / (2 * N_class)` từ cùng UNSW metadata như baseline, không đổi
sang balanced source sampling. Conditional MMD dùng V2 q02/[q95,q98) pools cố
định từ CICIDS adaptation-train; không dùng filtered V5c pools hay target labels.
Ranking teacher là V5b frozen, eval/no_grad trên cùng natural target batch.
Natural batch cập nhật adapter BN một lần; balanced pseudo batch tạm BN eval.
Shared layers frozen nhưng giữ autograd qua target path để adapter nhận gradient.

Adam chỉ nhận adapter (lr 1e-4) và classifier copy (lr 1e-5), weight decay 1e-4.
Kế thừa epochs/batch sizes của protocol V5b: 10 epochs, natural batch 256,
class batch 64. Lưu epoch cuối, không chọn checkpoint bằng development labels.
Ranking chỉ giữ tương quan margin, không trực tiếp tối ưu AP. Các hệ số và learning
rate trên là điểm bắt đầu, chưa được tối ưu.

Chạy từ thư mục gốc repo:

```bash
# Test tùy chọn trước khi train (chưa được chạy khi viết code)
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_hda_v5d.py

PYTHONPATH=src .venv/bin/python -u -m training.hda_v5d --config configs/hda_v5d.json
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5d --config configs/hda_v5d.json --stage fit
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5d --config configs/hda_v5d.json --stage report
```

Train tự chọn CUDA/MPS/CPU, có thể thêm `--device cpu`. Không cần chạy V5c audit.

Evaluation dùng protocol source-only threshold calibration: `fit` chọn ngưỡng
margin riêng cho V5b và V5d trên UNSW validation với cùng FPR tối đa 0.01.
V5b source path dùng nguyên baseline; V5d source path dùng classifier đã học.
Không fit affine từ target pools và không đọc development trong bước fit.
`report` mới đọc nhãn development và so sánh AP (`pr_auc` = average precision),
ROC-AUC, F1, Recall, FPR và delta V5d trừ V5b. FPR source không đảm bảo FPR target.
Đây là protocol khác affine calibration V5c; không so trực tiếp F1 giữa hai
report khác protocol. Development không phải untouched final holdout.

Artifacts không ghi đè:

- `models/hda_v5d/v5d_seed42.pt`: adapter và classifier mới, history, hashes.
- `results/hda_v5d/calibration/seed42.json`: ngưỡng và metrics source validation.
- `results/hda_v5d/development/v5d_vs_v5b_seed42.json`: so sánh development.

Giữ nguyên config/code giữa các bước. Muốn thử nghiệm khác, tạo config với bộ
checkpoint_dir/calibration_dir/result_dir mới. Không sửa hashes để tái sử dụng
artifact không còn khớp code hoặc dữ liệu.
