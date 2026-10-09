# Canonical pipeline contract

Entry point và các lệnh chạy nằm trong `README.md`.
Revision chính thức duy nhất: `canonical`.

- Upstream: `data/bigdata/thesis_20261005/`, với manifest và ingested/splits/common.
- Audits: `results/data_audit/canonical/`; không đọc prepared roots cũ.
- Frozen manifest: `data/revisions/canonical/manifest.json` và `manifest.sha256`.
- Training/evaluation: `models/canonical/` và `results/canonical/`.

Prepare kiểm tra counts, hashes, binary labels, raw duplicates/conflicts, canonical
invalid/missing values, split membership, cross-split groups và multiset replay
raw splits → common features. Raw identity audit phân 64 buckets để giới hạn RAM;
so sánh vẫn chính xác trên toàn bộ cột. Byte mappings và duration conversion được
kiểm tra trên toàn bộ splits; extractor equivalence vẫn có điều kiện như semantic
review đã ghi rõ.

Giữ toàn bộ flow, multiplicity và label gốc theo lựa chọn của người dùng. Khác
label ở cùng 5 feature được báo là ambiguity, không majority relabel. Nếu
log/scaling/float32 tạo nhóm giống nhau xuyên split ở một trong hai chiều, chuyển
cả nhóm về holdout xa nhất (`test > val > train`), rồi fit lại hai source-only
processors và kiểm tra đến khi hết overlap. Không chuyển test vào train, không
dùng test metrics để quyết định. Proportions có thể khác 70/15/15 vì giữ nguyên nhóm.

Freeze fit exact source-train median/IQR. Mỗi chiều dùng một processor fit trên
source train của chính nó. Cả hai chiều dùng cùng common data và membership.
Audit các file prepared trên đĩa trước khi tạo completion manifest. Hashes khóa
data, source manifest, audit reports, schema và preprocessing. Không sửa hashes
để ép checkpoint cũ tương thích.

Diagnostics tính intrinsic trên common train chưa scale; pooled unlabeled
transformation chỉ dùng cho diagnostic. Cặp MMD/bandwidth giữ nguyên khi đảo chiều,
float64 epsilon = 64 × machine epsilon. Báo same-domain controls và sensitivity
bỏ bytes/duration, bỏ shared scaler và sigma ×0.5/×2. ± std của năm subsamples
không phải confidence interval. Hai chiều source-fitted là diagnostics trong
không gian thực nghiệm, không dùng để xếp hạng intrinsic shift theo chiều.

Train chạy 40 experiments qua một source-only baseline và một adaptation trainer.
Aggregate dùng paired seeds. Final test là stage riêng, chạy sau khi chốt
checkpoints/configs/thresholds, với provenance và source-validation replay.

Build dở hoặc outputs đã tồn tại sẽ bị từ chối. Xử lý đúng các output canonical
của lần chạy đó trước khi chạy lại. Pipeline không tự xóa upstream hoặc checkpoint.
