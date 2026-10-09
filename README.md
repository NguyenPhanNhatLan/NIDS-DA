# Cross-domain NIDS — canonical proposal_v2 pipeline

Một entry point chính thức: `python -m experiments.proposal_pipeline`.
Một manifest: `data/revisions/canonical/manifest.json`.

```text
Nguồn Spark có manifest → raw identity + split + mapping/quality audit
→ thống nhất precision-collision groups → source-train preprocessing → freeze
→ intrinsic/directional diagnostics → 4 methods × 2 directions × 5 seeds
→ development aggregate → final-test riêng
```

Nguồn đầu vào hiện tại: `data/bigdata/thesis_20261005/`, gồm `manifest.json`,
`ingested/`, `splits/`, `common/`. Đây là upstream, không phải phiên bản training
thứ hai. Pipeline không phụ thuộc prepared roots cũ hoặc `data/features/common_raw`.

Năm inputs theo `configs/common_features_v2.json`: `flow_duration`, `fwd_packets`,
`bwd_packets`, `fwd_bytes`, `bwd_bytes`. CICIDS duration µs → seconds.
Giữ mọi flow/label, báo ambiguity; nhóm raw hoặc prepared float32 giống nhau
phải nằm trong một split. Hai chiều dùng cùng common splits và source-only fitted
median → signed log1p → RobustScaler. Manifest khóa hashes trước training.

## Cài đặt

```bash
cd /Users/thonph/Desktop/KLTN
source .venv/bin/activate
pip install -e '.[analytics]'
export PYTHONPATH="$PWD/src"
```

## Chạy theo thứ tự

```bash
python -m experiments.proposal_pipeline --stage prepare
python -m experiments.proposal_pipeline --stage verify
python -m experiments.proposal_pipeline --stage diagnostics
python -m experiments.proposal_pipeline --stage train
python -m experiments.proposal_pipeline --stage evaluation
python -m experiments.proposal_pipeline --stage aggregate
```

`prepare` gồm raw duplicate/label audit, replay splits, mapping/quality audit và
freeze. Raw identity audit chia 64 buckets để giới hạn bộ nhớ; so sánh vẫn chính
xác trên toàn bộ cột, hash chỉ dùng để phân vùng. Mỗi bước dừng khi audit lỗi.
Không tự ghi đè output của lần chạy trước hoặc build dở.

`diagnostics` gồm intrinsic controls/sensitivity và hai chiều source-fitted.
Intrinsic dùng cùng flow, subsamples và bandwidth khi hoán đổi domain; AUC dùng
holdout theo domain và gom duplicate vectors cùng partition. MMD ± std là độ
biến thiên subsampling, **không phải CI 95%**. Byte mapping còn có điều kiện về
extractor semantics, nên luôn báo sensitivity bỏ hai byte features.

`train` chạy source-only rồi marginal MMD, MK-MMD, class-aware MMD, theo cả hai
chiều và seeds 42–46. Cả ba adaptation methods dùng một trainer. Checkpoint chọn
bằng source-val AP; threshold bằng source-val F1. Không dùng target test để tuning.

`evaluation` chạy negative-transfer diagnostics trên validation sau training,
cho cả hai chiều, năm seeds và ba adaptation methods.

Chỉ sau khi chốt development mới chạy:

```bash
python -m experiments.proposal_pipeline --stage final-test
```

## Output duy nhất

```text
results/data_audit/canonical/                 # reports trước freeze
data/revisions/canonical/                    # common, prepared, manifest, audits
models/canonical/                            # preprocessing và checkpoints
results/canonical/                           # diagnostics, training, aggregate, final test
```

Pipeline tự chọn manifest; không cần export root hoặc direction riêng.
Hash/inventory thay đổi sẽ bị chặn trước training/evaluation.
Các module audit/training là thành phần nội bộ của cùng pipeline.
Notebook và `legacy/` là lịch sử/EDA, không thuộc luồng chạy chính thức.

Chi tiết: [canonical revision](docs/canonical_data_revision.md),
[semantic mapping](docs/semantic_feature_review_20261009.md).

Nếu cần tự chạy synthetic tests, tách khỏi dữ liệu thật:

```bash
KLTN_DATA_MANIFEST='' python -m unittest discover -s tests -v
```
