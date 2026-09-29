# V5d và V5d-A

V5d dùng classifier deepcopy riêng, khởi tạo adapter từ V5b asymmetric frozen.
Không sửa V1/V5b hoặc checkpoint source/teacher. Teacher seed luôn 42;
training_seed có thể là 42, 43, 44, chỉ đổi randomness của training.

Loss mặc định:

`hidden MMD + 0.05 Normal MMD + 0.02 Attack MMD + 0.10 rank + 0.10 source CE`.

Source CE giữ class weights `N / (2 * N_class)` như baseline. Conditional MMD
và calibration dùng V2 q02/[q95,q98) pools cố định từ adaptation-train không nhãn.
Ranking teacher là V5b frozen. Source encoder và shared FC2/BN2 luôn frozen/eval.
Adam dùng adapter lr 1e-4, classifier lr 1e-5, weight decay 1e-4; 10 epochs,
batch 256, class batch 64, lưu epoch cuối. Không chọn model bằng target labels.

## Các chế độ

- `configs/hda_v5d.json`: adapter và classifier trainable; natural target batch
  cập nhật adapter BN, pseudo-balanced batch dùng BN eval.
- `configs/hda_v5d_a.json`: `classifier_only=true`. Adapter weights, running_mean,
  running_var và num_batches_tracked đều frozen kể cả khi gọi `train()`.
  Optimizer chỉ nhận classifier. MMD vẫn được tính/log để giữ các bước lấy mẫu
  tương ứng, nhưng không có gradient tới representation. Classifier nhận source
  CE và ranking loss. Đây không phải thí nghiệm source-CE-only.
- `configs/hda_v5d_source_ce_only.json`: chỉ classifier trainable, source CE
  weight 1.0, bốn loss còn lại weight 0. Không tạo conditional pools khi train,
  không tính MMD, ranking, teacher forward hoặc adapter forward. Adapter weights
  và BN statistics vẫn frozen. Các loss bị tắt được log là 0 (không đo).
  Vẫn duyệt target loader để giữ số optimizer steps mỗi epoch giống joint;
  target features không đi vào loss. Adam weight decay vẫn là 1e-4.
  Calibration sau training vẫn cần V2 pseudo pools như protocol chung.

## Calibration trước development

Cả hai model dùng source FPR policy **2%**:

- V5b dùng lại nguyên artifact
  `results/hda_v5b/asymmetric/calibration/fpr_0p02_seed42.json`.
  Payload hash được pin trong config; code xác minh checkpoint, dependencies,
  source/target snapshots và FPR policy. Không fit lại V5b.
- V5d chọn threshold trên UNSW validation qua classifier mới. Fit affine riêng
  bằng median source margins theo nhãn UNSW và median target margins trên hai
  V2 pseudo pools không nhãn. Freeze `a > 0`, `b` và source threshold trước report.
- Report chấm `a * target_margin + b` ở source threshold riêng từng model.
  `operating_points` lưu cả raw (margin chưa calibrate tại source threshold) và
  calibrated metrics, kèm threshold tương đương trong raw margin space.
  `metrics` và `delta_v5d_minus_v5b` là kết quả calibrated; có raw delta riêng.

Pseudo anchors có thể nhiễu. Affine dương không tăng AP/ROC-AUC; nó phục vụ F1,
Recall và FPR. FPR source 2% không đảm bảo FPR target 2%. Development chưa phải
untouched final holdout. Không lấy target development labels để train hoặc fit.

## Lệnh chạy

Code/test chưa được chạy khi chỉnh sửa. Từ thư mục gốc repo:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_hda_v5d.py

PYTHONPATH=src .venv/bin/python -u -m training.hda_v5d --config configs/hda_v5d.json --training-seed 42 --preflight-only

PYTHONPATH=src .venv/bin/python -u -m training.hda_v5d --config configs/hda_v5d.json --training-seed 42
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5d --config configs/hda_v5d.json --training-seed 42 --stage fit
PYTHONPATH=src .venv/bin/python -u -m evaluation.hda_v5d --config configs/hda_v5d.json --training-seed 42 --stage report
```

Chỉ chạy bước tiếp theo nếu bước trước thành công. Training luôn tự chạy preflight
trước khi tạo pools/optimizer, kể cả khi đã chạy `--preflight-only`. Preflight
kiểm tra checkpoint, artifact pinned, reference, code và snapshots dữ liệu;
không tạo checkpoint hoặc thay đổi artifact. Chỉ bắt đầu với seed 42 để kiểm tra
pipeline đầy đủ, sau đó mới chạy seed 43/44. Quy tắc chọn checkpoint vẫn là epoch
cuối, không thay đổi sau khi xem development AP.

Compatibility Stage 1: artifact được tạo bằng `hda_v5b_calibration.py` hash
`27615884…`, code hiện tại hash `af7de972…`. Diff Git `0e1cf842 -> 815daced`
thêm helper kiểm tra pseudo policy/split/số dòng và metadata; không đổi model
loading, scoring, affine math hoặc threshold selection. Loader V5d chỉ chấp nhận
đúng cặp full SHA-256 đã rà soát này để đọc artifact cũ; các file khác vẫn phải
khớp tuyệt đối. Loader in thông báo compatibility, không thay hash/parameters
trong artifact và không refit V5b. Revision mới ngoài cặp này bị chặn với tên file,
hash kỳ vọng và hash hiện tại. Test bao phủ cả trường hợp chấp nhận và từ chối.

Để chạy V5d-A, thay config của cả ba lệnh bằng `configs/hda_v5d_a.json`.
Đối chứng Source CE dùng `configs/hda_v5d_source_ce_only.json`, output tách riêng
dưới `models/hda_v5d_source_ce_only` và `results/hda_v5d_source_ce_only`.
Để lặp seed, thay `--training-seed 42` bằng 43 hoặc 44 trong cả ba lệnh.
Nếu không truyền CLI, dùng training_seed trong config. Teacher vẫn luôn seed 42.
Có thể thêm `--device cpu` cho lệnh train. Không cần V5c audit.

Artifacts được tách theo chế độ và training seed:

- V5d: `models/hda_v5d/affine_fpr_0p02/v5d_seed42.pt`;
  `results/hda_v5d/affine_fpr_0p02/{calibration,development}/...`.
- V5d-A: cùng cấu trúc dưới `models/hda_v5d_a` và `results/hda_v5d_a`.

File calibration là `seed42.json`; report là `v5d_vs_v5b_seed42.json`, tương tự
cho seed 43/44. Không ghi đè artifact cũ. Các checkpoint/calibration V5d từ code
cũ không tương thích với revision này: dùng pipeline mới từ bước train. Không
sửa hash của artifact cũ. Giữ nguyên code/config giữa train, fit và report.
