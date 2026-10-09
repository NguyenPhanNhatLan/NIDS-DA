"""Create a new canonical snapshot and source-only fitted preprocessing, never overwrite old roots."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb

from features.common_features import COMMON_FEATURES
from models.proposal_pipeline import proposal_processor
from training.data_revision import file_sha256
from evaluation.proposal_data_audit import DOMAINS, SPLITS, inventory, audit_common

ROOT = Path(__file__).resolve().parents[2]


def freeze(audit_path, revision='canonical', conflict_policy='preserve'):
    audit_path = Path(audit_path)
    audit = json.loads(audit_path.read_text())
    if not audit['quality_gate_passed']:
        raise ValueError('Quality/leakage/mapping audit failed; cannot freeze')
    if conflict_policy != 'preserve':
        raise ValueError('Only explicit preservation of reduced-feature ambiguity is supported')
    if not revision or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in revision):
        raise ValueError('Revision must be a simple unique identifier')
    snapshot = ROOT / 'data/revisions' / revision
    model_root = ROOT / 'models' / revision
    result_root = ROOT / 'results' / revision
    if any(p.exists() for p in (snapshot, model_root, result_root)):
        raise FileExistsError('Use a new revision name; canonical roots are never overwritten')
    common_root, feature_root = snapshot / 'common', snapshot / 'prepared'
    common_root.mkdir(parents=True)
    frozen_files = []

    def bind(path):
        frozen_files.append({'path': str(path.resolve()), 'sha256': file_sha256(path), 'bytes': path.stat().st_size})

    for d in DOMAINS:
        for s in SPLITS:
            entry = audit['common_inventory'][d][s]
            source = Path(entry['path'])
            current = inventory(source)
            if current['split_sha256'] != entry['split_sha256']:
                raise ValueError(f'Data changed since audit: {source}')
            destination = common_root / f'{d}_{s}'
            destination.mkdir()
            for file in sorted(source.glob('*.parquet')):
                target = destination / file.name
                shutil.copyfile(file, target)
    # Float32/log/scaling can collapse different raw keys. Move every group
    # crossing a split in either direction to its furthest held-out partition.
    # This is a purity audit, not a label-based choice or performance tuning.
    arrays, labels, memberships, keep = {}, {}, {}, {}
    for d in DOMAINS:
        chunks, ys, splits = [], [], []
        for index, s in enumerate(SPLITS):
            table = pq.read_table(common_root / f'{d}_{s}', columns=[*COMMON_FEATURES, 'label'])
            chunks.append(np.column_stack([table.column(n).to_numpy() for n in COMMON_FEATURES]))
            ys.append(table.column('label').to_numpy())
            splits.append(np.full(len(table), index, dtype=np.int8))
        arrays[d] = np.concatenate(chunks)
        labels[d] = np.concatenate(ys)
        memberships[d] = np.concatenate(splits)
        keep[d] = np.ones(len(arrays[d]), dtype=bool)
    connection = duckdb.connect()
    connection.execute("SET threads=2")
    connection.execute("SET memory_limit='1GB'")
    precision_iterations = []
    processors = {}
    for iteration in range(100):
        new_memberships = {d: memberships[d].copy() for d in DOMAINS}
        for direction, source in (('unsw_to_cicids', 'unsw'), ('cicids_to_unsw', 'cicids')):
            processor = proposal_processor().fit(arrays[source][keep[source] & (memberships[source] == 0)])
            processors[direction] = processor
            for d in DOMAINS:
                indices = np.flatnonzero(keep[d])
                values = processor.transform(arrays[d][indices]).astype(np.float32)
                values[values == 0] = 0  # Normalize signed zero for byte-key comparisons.
                table = pa.table({**{n: values[:, i] for i, n in enumerate(COMMON_FEATURES)},
                                  'split': memberships[d][indices]})
                connection.register('precision_rows', table)
                cols = ','.join(COMMON_FEATURES)
                collisions = connection.execute(
                    f'SELECT {cols}, max(split) AS destination FROM precision_rows GROUP BY {cols} HAVING count(distinct split)>1'
                ).fetchnumpy()
                if len(collisions[COMMON_FEATURES[0]]):
                    collision_values = np.column_stack([collisions[n] for n in COMMON_FEATURES]).astype(np.float32)
                    row_keys = np.ascontiguousarray(values).view('V20').ravel()
                    collision_keys = np.ascontiguousarray(collision_values).view('V20').ravel()
                    order = np.argsort(collision_keys)
                    sorted_keys = collision_keys[order]
                    positions = np.searchsorted(sorted_keys, row_keys)
                    safe_positions = np.minimum(positions, len(sorted_keys) - 1)
                    matched = (positions < len(sorted_keys)) & (sorted_keys[safe_positions] == row_keys)
                    matched_rows = indices[matched]
                    target_splits = collisions['destination'][order][safe_positions[matched]]
                    new_memberships[d][matched_rows] = np.maximum(new_memberships[d][matched_rows], target_splits)
                connection.unregister('precision_rows')
        moved = {d: {s: int(np.count_nonzero((new_memberships[d] != memberships[d]) & (memberships[d] == i)))
                       for i, s in enumerate(SPLITS)} for d in DOMAINS}
        precision_iterations.append(moved)
        print(f'Precision leakage audit iteration {iteration}: moved from {moved}', flush=True)
        if not any(np.any(new_memberships[d] != memberships[d]) for d in DOMAINS):
            break
        for d in DOMAINS:
            memberships[d] = new_memberships[d]
    else:
        raise ValueError('Precision collision audit did not converge; snapshot not frozen')
    for d in DOMAINS:
        for i, s in enumerate(SPLITS):
            destination = common_root / f'{d}_{s}'
            # These are newly created snapshot copies, never original files.
            for copied in destination.glob('*.parquet'):
                copied.unlink()
            selected = keep[d] & (memberships[d] == i)
            table = pa.table({**{n: arrays[d][selected, j] for j, n in enumerate(COMMON_FEATURES)},
                              'label': labels[d][selected]})
            output = destination / 'data.parquet'
            pq.write_table(table, output, compression='snappy', row_group_size=65536)
            bind(output)
    canonical_quality = {d: audit_common(connection, common_root, d) for d in DOMAINS}
    for d in DOMAINS:
        actual_rows = sum(r['rows'] for r in canonical_quality[d]['class_counts'])
        original_rows = sum(r['rows'] for r in audit['common_inventory'][d].values())
        if actual_rows != original_rows or canonical_quality[d]['cross_split_feature_groups']:
            raise ValueError('Canonical revision must preserve every flow and keep raw feature groups in one split')
    connection.close()
    (snapshot / 'canonical_quality.json').write_text(json.dumps({
        'quality': canonical_quality, 'precision_relocation_iterations': precision_iterations,
        'precision_cross_split_groups_after_final_fit': 0,
        'policy': 'Move complete cross-split prepared float32 collision groups in either direction to max split index (test > val > train); refit source-only preprocessing to convergence; retain all flows/labels; no test metric use.',
    }, indent=2) + '\n')
    bind(snapshot / 'canonical_quality.json')
    del arrays, labels, memberships, keep
    # Verify original raw inputs and source manifest are bound to this snapshot.
    source_manifest_path = Path(audit['source_manifest']['path'])
    if file_sha256(source_manifest_path) != audit['source_manifest']['sha256']:
        raise ValueError('Source manifest changed since audit')
    source_manifest = json.loads(source_manifest_path.read_text())
    for paths in source_manifest['raw_inputs'].values():
        for name in paths:
            path = Path(name)
            bind(path if path.is_absolute() else ROOT / path)
    for path in (audit_path, source_manifest_path, ROOT / 'configs/common_features_v2.json'):
        bind(path)
    for name in ('proposal_suite_v2', 'proposal_mmd_v2', 'proposal_mkmmd_v2', 'proposal_class_aware_v2'):
        config = ROOT / 'configs' / f'{name}.json'
        if config.exists():
            bind(config)
    if 'split_replay' in audit:
        replay_path = Path(audit['split_replay']['path'])
        if file_sha256(replay_path) != audit['split_replay']['sha256']:
            raise ValueError('Split replay changed since audit')
        bind(replay_path)
    if 'raw_identity_audit' in audit:
        raw_audit = audit['raw_identity_audit']
        raw_path = Path(raw_audit['path'])
        if file_sha256(raw_path) != raw_audit['sha256'] or not raw_audit['result']['quality_gate_passed']:
            raise ValueError('Raw identity audit changed or failed')
        bind(raw_path)
    semantic_review = ROOT / 'docs/semantic_feature_review_20261009.md'
    if semantic_review.exists():
        bind(semantic_review)
    (snapshot / 'source_manifest.json').write_text(json.dumps(source_manifest, indent=2) + '\n')
    (snapshot / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    bind(snapshot / 'source_manifest.json')
    bind(snapshot / 'audit.json')
    counts, preprocessing = {}, {}
    for direction, source in (('unsw_to_cicids', 'unsw'), ('cicids_to_unsw', 'cicids')):
        print(f'Write exact source-train fitted preprocessing: {direction}', flush=True)
        processor = processors[direction]
        processor.data_revision = revision
        path = model_root / direction / 'preprocessor.joblib'
        path.parent.mkdir(parents=True)
        joblib.dump(processor, path)
        bind(path)
        preprocessing[direction] = {
            'fit_role': f'{source}_train_only', 'quantiles': 'exact sklearn median/IQR',
            'medians': processor.named_steps['imputer'].statistics_.tolist(),
            'centers': processor.named_steps['scaler'].center_.tolist(),
            'scales': processor.named_steps['scaler'].scale_.tolist(),
            'sha256': file_sha256(path),
        }
        counts[direction] = {}
        for d in DOMAINS:
            for s in SPLITS:
                destination = feature_root / direction / f'{d}_{s}'
                destination.mkdir(parents=True)
                output = destination / 'data.parquet'
                schema = pa.schema([('features', pa.list_(pa.float32(), len(COMMON_FEATURES))), ('label', pa.int64())])
                rows = 0
                with pq.ParquetWriter(output, schema, compression='snappy') as writer:
                    for file in sorted((common_root / f'{d}_{s}').glob('*.parquet')):
                        for batch in pq.ParquetFile(file).iter_batches(batch_size=65536, columns=[*COMMON_FEATURES, 'label']):
                            raw = np.column_stack([batch.column(name).to_numpy(zero_copy_only=False) for name in COMMON_FEATURES])
                            values = processor.transform(raw).astype(np.float32)
                            if not np.isfinite(values).all():
                                raise ValueError(f'Nonfinite prepared data: {d}_{s}')
                            labels = batch.column('label').to_numpy(zero_copy_only=False)
                            if not np.isin(labels, [0, 1]).all():
                                raise ValueError('Invalid label')
                            vectors = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), len(COMMON_FEATURES))
                            writer.write_table(pa.table({'features': vectors, 'label': pa.array(labels, type=pa.int64())}, schema=schema))
                            rows += len(values)
                expected = sum(pq.ParquetFile(p).metadata.num_rows for p in (common_root / f'{d}_{s}').glob('*.parquet'))
                if rows != expected:
                    raise ValueError('Prepared/common row count mismatch')
                counts[direction][f'{d}_{s}'] = rows
                bind(output)
    # Independent audit of the files written to disk, not just in-memory arrays.
    c = duckdb.connect()
    c.execute("SET threads=2")
    c.execute("SET memory_limit='1GB'")
    prepared_quality = {}
    for direction in processors:
        prepared_quality[direction] = {}
        for d in DOMAINS:
            union = ' UNION ALL '.join(
                f"SELECT features,label,'{s}' split FROM read_parquet('{feature_root / direction / f'{d}_{s}' / '*.parquet'}')"
                for s in SPLITS
            )
            c.execute('CREATE OR REPLACE TEMP VIEW prepared_audit AS ' + union)
            overlap = c.execute('SELECT count(*) FROM (SELECT features FROM prepared_audit GROUP BY features HAVING count(distinct split)>1)').fetchone()[0]
            invalid = c.execute('SELECT count(*) FROM prepared_audit WHERE label IS NULL OR label NOT IN (0,1) OR len(features)<>5 OR list_contains(list_transform(features,x -> x IS NULL OR NOT isfinite(x)),true)').fetchone()[0]
            prepared_quality[direction][d] = {'cross_split_vector_groups': overlap, 'invalid_rows': invalid}
            if overlap or invalid:
                raise ValueError(f'Prepared file quality gate failed: {direction}/{d}')
    c.close()
    quality_path = snapshot / 'prepared_quality.json'
    quality_path.write_text(json.dumps(prepared_quality, indent=2) + '\n')
    bind(quality_path)
    manifest = {
        'status': 'frozen', 'quality_gate_passed': True,
        'data_revision': revision, 'protocol': 'proposal_v2',
        'flow_counts_preserved': True,
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'common_root': str(common_root.resolve()), 'feature_root': str(feature_root.resolve()),
        'model_root': str(model_root.resolve()), 'result_root': str(result_root.resolve()),
        'feature_order': list(COMMON_FEATURES), 'split_seed': 42,
        'split_rule': 'Initial thesis_20261005 Spark xxhash64 raw five-feature keys + lit(42), buckets 70/15/15; complete precision-collision groups moved to max split index until both directions have zero prepared overlap',
        'conflict_policy': 'preserve raw flow multiplicity and original labels; reduced-feature ambiguity reported; no majority relabeling',
        'missing_policy': 'invalid raw CICIDS durations already null; source-train median imputation downstream of split',
        'precision_collision_policy': 'Move complete groups to furthest holdout; refit source-only preprocessing until zero overlap; identical rows and memberships for both directions; retain all labels/flows',
        'precision_relocation_iterations': precision_iterations,
        'canonical_quality': str((snapshot / 'canonical_quality.json').resolve()),
        'prepared_quality': str(quality_path.resolve()),
        'semantic_byte_mapping_status': 'conditional extractor equivalence; diagnostic sensitivity without bytes is required',
        'preprocessing': preprocessing, 'prepared_counts': counts,
        'profiles': source_manifest.get('profiles', {}),
        'counts': {'protocol': 'proposal_v2', 'data_revision': revision,
                   'class_counts': [
                       {'domain': d, 'split': row['split'], 'label': row['label'], 'count': row['rows']}
                       for d, quality in canonical_quality.items() for row in quality['class_counts']
                   ]},
        'dataset_statistics_stage': source_manifest.get('dataset_statistics_stage'),
        'audit': str((snapshot / 'audit.json').resolve()),
        'frozen_files': frozen_files,
    }
    path = snapshot / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    (snapshot / 'manifest.sha256').write_text(file_sha256(path) + '  manifest.json\n')
    print(f'Frozen manifest: {path}', flush=True)
    return path
