"""Supplemental full-row duplicate and binary-label-conflict audit of ingested raw records."""
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from training.data_revision import file_sha256
from evaluation.proposal_data_audit import sql_string, ROOT


def run(output, work_root):
    c = duckdb.connect()
    c.execute("SET threads=1")
    c.execute("SET memory_limit='1GB'")
    c.execute("SET preserve_insertion_order=false")
    result = {'domains': {}}
    work_root = Path(work_root)
    manifest = json.loads((work_root / 'manifest.json').read_text())
    for d in ('unsw', 'cicids'):
        print(f'Full raw identity audit: {d}', flush=True)
        root = work_root / 'ingested' / d
        files = sorted(root.glob('*.parquet'))
        if not files:
            raise FileNotFoundError(root)
        schema = pq.read_schema(files[0])
        columns = [n for n in schema.names if n != '_source_file']
        quote = lambda n: '"' + n.replace('"', '""') + '"'
        full = ','.join(map(quote, columns))
        nonlabels = ','.join(map(quote, [n for n in columns if n not in ('label', 'attack_cat')]))
        c.execute(f'CREATE OR REPLACE TEMP VIEW raw_identity AS SELECT {full} FROM read_parquet({sql_string(root / "*.parquet")})')
        rows = c.execute('SELECT count(*) FROM raw_identity').fetchone()[0]
        label = ('CASE WHEN try_cast(trim(label) AS DOUBLE) IN (0,1) THEN try_cast(trim(label) AS INTEGER) END'
                 if d == 'unsw' else "CASE WHEN upper(trim(label))='BENIGN' THEN 0 WHEN trim(label)<>'' AND NOT regexp_full_match(trim(label),'[+-]?[0-9]+(\\.[0-9]+)?') THEN 1 END")
        # Hash only partitions the work; equality remains exact across all columns.
        # Conflicting labels for the same identity always belong to the same bucket.
        unique_rows = conflict_groups = conflict_rows = 0
        for bucket in range(64):
            where = f'hash(row({nonlabels})) % 64 = {bucket}'
            unique_rows += c.execute(f'SELECT count(*) FROM (SELECT DISTINCT {full} FROM raw_identity WHERE {where})').fetchone()[0]
            groups, count = c.execute(
                f'SELECT count(*),coalesce(sum(n),0) FROM (SELECT {nonlabels},count(*) n FROM raw_identity WHERE {where} GROUP BY {nonlabels} HAVING count(distinct {label})>1)'
            ).fetchone()
            conflict_groups += groups
            conflict_rows += count
        csv_rows = sum(c.execute(f'SELECT count(*) FROM read_csv({sql_string(ROOT / p)}, header=true, all_varchar=true, sample_size=20480)').fetchone()[0]
                       for p in manifest['raw_inputs'][d])
        result['domains'][d] = {
            'raw_csv_rows': csv_rows, 'ingested_rows': rows, 'unique_full_raw_rows': unique_rows,
            'exact_full_raw_duplicates': rows - unique_rows,
            'binary_label_conflicting_full_nonlabel_groups': conflict_groups,
            'rows_in_full_nonlabel_conflicts': conflict_rows,
            'invalid_binary_labels': c.execute(f'SELECT count(*) FROM raw_identity WHERE ({label}) IS NULL').fetchone()[0],
            'manifest_counts_match': rows == manifest['profiles'][d]['input_rows'] and unique_rows == manifest['profiles'][d]['rows'],
            'csv_ingested_counts_match': csv_rows == rows,
            'ingested_file_hashes': [{'path': str(p), 'sha256': file_sha256(p)} for p in files],
            'interpretation': 'Full identity uses all ingested nonlabel columns, including retained flow identifiers/time; excludes label and attack_cat. Reduced-five-feature ambiguity is a separate audit.',
        }
    c.close()
    result['quality_gate_passed'] = all(
        d['manifest_counts_match'] and d['csv_ingested_counts_match'] and not d['invalid_binary_labels']
        for d in result['domains'].values()
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved: {output}', flush=True)
    return result
