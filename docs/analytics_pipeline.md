# Research pipeline → BI

`analytics.build_dashboard_tables` chỉ đọc research JSON và data manifest; không đọc
raw flows, checkpoints hay target-test labels, không train/evaluate lại model.

```text
results/proposal_v2/*.json + Spark data manifest (optional)
    → analytics/*.parquet
    → DuckDB
    → Tableau Desktop
```

Cài đặt và chạy từ repo root:

```bash
source .venv/bin/activate
pip install -e '.[analytics]'
export PYTHONPATH="$PWD/src"
python -m analytics.build_dashboard_tables --duckdb
```

Mặc định đọc `results/proposal_v2`, xuất snapshot mới vào `analytics/`, gồm đúng
5 bảng Parquet, `_manifest.json`, `create_views.sql` và, với `--duckdb`,
`dashboard.duckdb`. Không ghi đè thư mục output đã tồn tại. Dùng
`--output analytics_snapshot_02` để refresh thành snapshot mới. Nếu chỉ cần Parquet,
bỏ `--duckdb`; không cần cài DuckDB cho bước này.

Nếu đã hoàn thành Spark data pipeline, thêm manifest của đúng revision đang nghiên cứu:

```bash
python -m analytics.build_dashboard_tables \
  --output analytics_snapshot_02 --duckdb \
  --dataset-manifest data/bigdata/proposal_v2/manifest.json
```

| Bảng | Grain / mục đích |
|---|---|
| experiment_runs | Một hàng mỗi direction × method × seed × phase; metrics, lambda, paired ΔAP và diagnostics |
| feature_shift | Một hàng mỗi direction × sampling seed × feature; KS/p-value, mean/median, input MMD và domain AUC |
| negative_transfer | Một hàng mỗi adapted development run có diagnostics; ΔAP, gradient conflict và heuristic causes |
| dataset_summary | Với manifest: một hàng mỗi domain × split; nếu thiếu manifest: counts suy ra từ confusion counts/source train, tách direction và split hash |
| model_summary | Một hàng mỗi direction × method × phase × metric; n, seeds, mean, sample std và CI 95% |

`lambda` là cột DOUBLE (truy vấn SQL cần viết `"lambda"`). Source-only là 0;
MMD đọc `lambda_mmd`; final kế thừa lambda từ development checkpoint tương ứng,
hoặc NULL nếu development không có. Không suy đoán hyperparameters từ tên file.

`adaptation_gain = AP(method, seed, phase) − AP(source_only, same seed, same phase)`.
Không lấy development baseline để tính final gain. Diagnostics chỉ gắn vào development
run có cùng config hash và checkpoint; final diagnostics để NULL.
`feature_shift.input_mmd` ở input space khác với `experiment_runs.mmd_before/after`
ở latent space: không coi chúng là cùng phép đo.

Các diagnostics thiếu là NULL, các artifact families thiếu cho bảng có schema đầy đủ
nhưng 0 rows. Aggregate per_run là fallback khi individual JSON thiếu; không đếm lại
individual runs. Summary mean/std/seeds của aggregate phải khớp các exported facts.
`source_file`, hashes JSON trong manifest và `summary_source` ghi nguồn của dữ liệu.
Exporter kiểm tra consistency metadata; việc xác minh checkpoint/data bytes vẫn thuộc
research evaluation pipeline. Không xem bảng xuất thành bằng chứng suite đã chạy đủ.

Chỉ dùng 4 thesis methods và canonical v2 config tags. Các v1/lambda sweep/auxiliary
JSON nằm ngoài suite được ghi vào `_manifest.json.ignored_files`. Duplicate run,
stale diagnostic, sai schema/phase, unmatched baseline provenance hoặc summary
conflict làm export thất bại trước khi publish snapshot.

`model_summary` giữ n theo metric. CI Student-t của gain tính trên paired differences;
với một seed CI là NULL. Mẫu 5 seeds phản ánh run variation trên dataset cố định,
không phải uncertainty của toàn bộ population. Không báo p-value/significance.

Có manifest thì dataset counts không bị nhân đôi theo direction. Không có manifest,
cần lọc một direction khi tổng hợp dataset counts vì cùng một domain có thể xuất hiện
trong cả hai hướng. Chỉ là counts từ full split, không lấy sample sizes của diagnostics
để trình bày như dataset totals.

DuckDB được materialize thành 5 tables, nên `dashboard.duckdb` có thể di chuyển độc lập
với Parquet. Kết nối read-only khi phân tích, ví dụ:

```python
import duckdb
con = duckdb.connect('analytics/dashboard.duckdb', read_only=True)
print(con.execute('''
    SELECT direction, method, seed, phase, "lambda", pr_auc, adaptation_gain
    FROM experiment_runs
    ORDER BY phase, direction, method, seed
''').fetchall())
```

`create_views.sql` là lựa chọn khác để tạo DuckDB views đọc trực tiếp Parquet bằng
absolute paths; views cần giữ nguyên vị trí các Parquet files.
[DuckDB Parquet documentation](https://duckdb.org/docs/lts/data/parquet/overview).

Để dùng Tableau Desktop, cài DuckDB JDBC driver và DuckDB Taco connector theo
[hướng dẫn DuckDB/Tableau](https://duckdb.org/docs/lts/guides/data_viewers/tableau),
rồi chọn file database đã xuất. Đây là kết nối file embedded, không cần database server;
chỉ tạo `.duckdb` chưa tự cài connector vào Tableau.

Dashboard nên dùng filter `phase` bắt buộc, và join diagnostics bằng `run_id`.
`feature_shift` là bảng riêng ở feature grain; join trực tiếp vào experiment_runs
sẽ nhân số rows theo features. PostgreSQL có thể thêm sau như serving layer mà
không đổi 5 bảng Parquet; hiện không tạo server hay credentials PostgreSQL.

Chạy riêng tests analytics (fixtures nhỏ, không khởi động Spark):

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_proposal_analytics.py' -v
```
