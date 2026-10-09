"""Read-only reconciliation, quality and harmonization audit before freezing data."""
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from features.common_features import COMMON_FEATURES
from training.proposal_data import split_sha256
from training.data_revision import file_sha256 as sha256

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ('train', 'val', 'test')
DOMAINS = ('unsw', 'cicids')


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def inventory(path):
    files = sorted(Path(path).glob('*.parquet'))
    if not files:
        raise FileNotFoundError(path)
    return {
        'path': str(Path(path).resolve()),
        'rows': sum(pq.ParquetFile(p).metadata.num_rows for p in files),
        'split_sha256': split_sha256(path),
        'files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in files],
        'schema': str(pq.read_schema(files[0])),
    }


def audit_common(connection, common_root, domain):
    cols = ','.join(COMMON_FEATURES)
    union = ' UNION ALL '.join(
        f"SELECT *, '{s}' AS split FROM read_parquet({sql_string(Path(common_root) / f'{domain}_{s}' / '*.parquet')})"
        for s in SPLITS
    )
    connection.execute('CREATE OR REPLACE TEMP VIEW audit_rows AS ' + union)
    grouped = f'SELECT {cols}, count(*) n, count(distinct label) labels, count(distinct split) splits FROM audit_rows GROUP BY {cols}'
    group_stats = connection.execute(
        'SELECT count(*), coalesce(sum(n-1),0), count(*) FILTER (WHERE labels>1), '
        'coalesce(sum(n) FILTER (WHERE labels>1),0), count(*) FILTER (WHERE splits>1) '
        f'FROM ({grouped})'
    ).fetchone()
    quality = {}
    for i, name in enumerate(COMMON_FEATURES):
        invalid = f'{name}<0 OR NOT isfinite({name})'
        if i:
            invalid += f' OR {name}<>floor({name})'
        missing, bad, minimum, maximum = connection.execute(
            f'SELECT count(*) FILTER (WHERE {name} IS NULL), count(*) FILTER (WHERE {invalid}), '
            f'min({name}), max({name}) FROM audit_rows'
        ).fetchone()
        quality[name] = {'missing': missing, 'invalid_nonmissing': bad, 'min': minimum, 'max': maximum}
    invalid_labels = connection.execute('SELECT count(*) FROM audit_rows WHERE label IS NULL OR label NOT IN (0,1)').fetchone()[0]
    labels = connection.execute('SELECT split,label,count(*) FROM audit_rows GROUP BY ALL ORDER BY ALL').fetchall()
    return {
        'feature_groups': group_stats[0], 'repeated_feature_rows': group_stats[1],
        'ambiguous_label_groups': group_stats[2], 'rows_in_ambiguous_label_groups': group_stats[3],
        'cross_split_feature_groups': group_stats[4], 'invalid_labels': invalid_labels,
        'feature_quality': quality,
        'class_counts': [{'split': s, 'label': label, 'rows': n} for s, label, n in labels],
        'interpretation': 'Repeated 5-feature vectors are not necessarily duplicate flows; conflicting reduced-feature labels are representation ambiguity, not proven ground-truth errors.',
    }


def audit_mapping(connection, work_root, domain, config):
    checks = {}
    for split in SPLITS:
        expressions = []
        for item in config:
            value = item[domain]
            if domain == 'cicids' and item['transform'] == 'cicids_us_to_seconds':
                value += '/1000000.0'
            expressions.append(f'{value} AS {item["canonical_name"]}')
        expected = 'SELECT ' + ','.join(expressions) + ',label FROM read_parquet(' + sql_string(Path(work_root) / 'splits' / f'{domain}_{split}' / '*.parquet') + ')'
        actual = 'SELECT ' + ','.join(COMMON_FEATURES) + ',label FROM read_parquet(' + sql_string(Path(work_root) / 'common' / f'{domain}_{split}' / '*.parquet') + ')'
        # EXCEPT ALL checks multiplicity as well as values/labels, independent of file order.
        mismatch = connection.execute(f'SELECT count(*) FROM (({expected} EXCEPT ALL {actual}) UNION ALL ({actual} EXCEPT ALL {expected}))').fetchone()[0]
        checks[split] = {'multiset_mismatches': mismatch}
    return checks


def run(output, work_root, replay_path, raw_audit_path):
    work_root = Path(work_root)
    c = duckdb.connect()
    c.execute("SET threads=2")
    c.execute("SET memory_limit='1GB'")
    config_path = ROOT / 'configs/common_features_v2.json'
    config = json.loads(config_path.read_text())
    manifest_path = work_root / 'manifest.json'
    source_manifest = json.loads(manifest_path.read_text())
    result = {
        'analysis': 'pre_training_data_audit', 'common_feature_config_sha256': sha256(config_path),
        'source_manifest': {'path': str(manifest_path.resolve()), 'sha256': sha256(manifest_path)},
        'source_manifest_config_matches': source_manifest['common_feature_config_sha256'] == sha256(config_path),
        'common_quality': {}, 'harmonization_replay': {},
    }
    result['common_inventory'] = {}
    for d in DOMAINS:
        print(f'Quality/leakage and mapping replay: {d}', flush=True)
        result['common_inventory'][d] = {s: inventory(work_root / 'common' / f'{d}_{s}') for s in SPLITS}
        result['common_quality'][d] = audit_common(c, work_root / 'common', d)
        result['harmonization_replay'][d] = audit_mapping(c, work_root, d, config)
    expected_counts = {(r['domain'], r['split']): 0 for r in source_manifest['counts']['class_counts']}
    for r in source_manifest['counts']['class_counts']:
        expected_counts[(r['domain'], r['split'])] += r['count']
    result['manifest_counts_match'] = all(
        result['common_inventory'][d][s]['rows'] == expected_counts[(d,s)] for d in DOMAINS for s in SPLITS
    )
    result['quality_gate_passed'] = (
        result['source_manifest_config_matches'] and result['manifest_counts_match']
        and all(not q['cross_split_feature_groups'] and not q['invalid_labels']
                and all(not v['invalid_nonmissing'] for v in q['feature_quality'].values())
                for q in result['common_quality'].values())
        and all(not check['multiset_mismatches'] for d in result['harmonization_replay'].values() for check in d.values())
    )
    replay_path = Path(replay_path)
    if replay_path.exists():
        replay = json.loads(replay_path.read_text())
        result['split_replay'] = {'path': str(replay_path.resolve()), 'sha256': sha256(replay_path), 'result': replay}
        result['quality_gate_passed'] &= all(
            not any(replay[d]['new_split_bucket_mismatches'].values()) for d in DOMAINS
        )
    else:
        result['quality_gate_passed'] = False
        result['split_replay_missing'] = True
    raw_audit_path = Path(raw_audit_path)
    raw_audit = json.loads(raw_audit_path.read_text())
    result['raw_identity_audit'] = {'path': str(raw_audit_path.resolve()),
                                    'sha256': sha256(raw_audit_path), 'result': raw_audit}
    result['quality_gate_passed'] &= raw_audit['quality_gate_passed']
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    c.close()
    print(f'Saved: {output}; quality gate={result["quality_gate_passed"]}', flush=True)
    return result
