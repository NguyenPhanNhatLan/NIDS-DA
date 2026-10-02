"""Replay historical row cleaning, recovering columns without changing split membership.

Run with Spark's Python environment (spark-submit --driver-memory 6g ...).
All outputs are new, versioned directories. Existing datasets are read-only.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'data/processed/common_feature_audit/v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalized_names(names):
    seen, result = set(), []
    for name in names:
        clean = ('fwd_header_length' if name == 'fwd_header_length.1' else
                 name.strip().lower().replace(' ', '_').replace('.', '_').replace('/', '_'))
        final, index = clean, 1
        while final in seen:
            final = f'{clean}_{index}'
            index += 1
        result.append(final)
        seen.add(final)
    return result


def notebook_removals(path, original_columns, existing_columns):
    """Read literal requests and saved historical output, not a new correlation fit."""
    notebook = json.loads(Path(path).read_text())
    requested = {}
    correlation = None
    for index, cell in enumerate(notebook['cells']):
        code = ''.join(cell.get('source', []))
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ('zero_variance_cols', 'near_zero_variance_cols'):
                        requested[target.id] = ast.literal_eval(node.value)
        for output in cell.get('outputs', []):
            for line in ''.join(output.get('text', [])).splitlines():
                if line.startswith('Các cột loại bỏ:'):
                    correlation = ast.literal_eval(line.split(':', 1)[1].strip())
    if correlation is None:
        raise ValueError('Historical correlation removal output not present')
    zero, near = requested['zero_variance_cols'], requested['near_zero_variance_cols']
    original = set(original_columns)
    removed = set(zero + near + correlation) & original
    actual = original - set(existing_columns) - {'label'}
    if actual != removed:
        raise ValueError(f'Notebook/schema mismatch: {actual ^ removed}')
    return dict(constant_removed=[c for c in zero if c in original],
                constant_requested_but_not_removed=[c for c in zero if c not in original],
                near_constant_removed=[c for c in near if c in original],
                correlation_removed=correlation, other_manual_removed=[],
                original_feature_count=len(original - {'label'}),
                retained_feature_count=len(set(existing_columns) - {'label'}),
                notebook_sha256=sha256_file(path))


def attach_boundaries(raw, old_parts, key_columns, quarantine_ambiguous=False):
    """Exact typed feature-key join, with multiset and cross-boundary checks.

    Multiple raw records may share a retained key, but only if the entire group
    belongs to one old split and its multiplicity matches exactly. No positional
    matching, approximate floats, inferred labels, or rehashing of new features.
    """
    from functools import reduce
    from pyspark.sql import functions as F
    from pyspark import StorageLevel
    old = reduce(lambda a, b: a.unionByName(b), [
        frame.select(F.struct(*[F.col(c) for c in key_columns]).alias('_key'))
        .withColumn('_split', F.lit(split)) for split, frame in old_parts.items()])
    membership = old.groupBy('_key').agg(F.count('*').alias('_old_count'),
        F.min('_split').alias('_split'), F.countDistinct('_split').alias('_split_count'),
        F.sort_array(F.collect_set('_split')).alias('_possible_splits'))
    membership = membership.persist(StorageLevel.DISK_ONLY)
    ambiguous = membership.filter(F.col('_split_count') != 1)
    if not quarantine_ambiguous and ambiguous.limit(1).count():
        raise ValueError('Ambiguous provenance: retained feature key spans old splits')
    keyed = raw.withColumn('_key', F.struct(*[F.col(c) for c in key_columns]))
    counts = keyed.groupBy('_key').count().withColumnRenamed('count', '_new_count')
    comparison = counts.join(membership, '_key', 'full')
    bad = comparison.filter(F.col('_new_count').isNull() | F.col('_old_count').isNull() |
                            (F.col('_new_count') != F.col('_old_count')))
    if bad.limit(1).count():
        example = bad.select('_old_count', '_new_count').limit(5).collect()
        raise ValueError(f'Raw/old feature-key multisets differ: {example}')
    usable = membership.filter(F.col('_split_count') == 1)
    joined = keyed.join(usable.select('_key', '_split'), '_key', 'inner')
    joined = joined.withColumn('_boundary_key_sha256', F.sha2(F.to_json('_key'), 256)).drop('_key')
    quarantine = keyed.join(ambiguous.select('_key', '_possible_splits'), '_key', 'inner')
    quarantine = quarantine.withColumn('_boundary_key_sha256', F.sha2(F.to_json('_key'), 256)).drop('_key')
    summary = dict(ambiguous_keys=ambiguous.count(),
                   ambiguous_rows=int(ambiguous.agg(F.coalesce(F.sum('_old_count'), F.lit(0))).first()[0]),
                   verified_multisets_equal=True,
                   usable_counts={r['_split']: int(r['rows']) for r in usable.groupBy('_split').agg(F.sum('_old_count').alias('rows')).collect()})
    return joined, quarantine, membership, summary


def regenerate(root=ROOT, output=DEFAULT_OUTPUT, quarantine_ambiguous=False):
    from pyspark.sql import SparkSession, functions as F
    from pyspark import StorageLevel
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Versioned destination already exists: {output}')
    # A failed build remains visibly incomplete and must never be consumed.
    output.mkdir(parents=True)
    spark = (SparkSession.builder.appName('unpruned-feature-audit-v1').master('local[2]')
             .config('spark.sql.shuffle.partitions', '32')
             .config('spark.sql.adaptive.enabled', 'false')
             .config('spark.sql.session.timeZone', 'UTC').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    manifest = dict(version=1, status='building', spark_version=spark.version,
                    membership='exact retained typed feature-key join, preserving multiplicities', datasets={})
    try:
        for domain, filename in [('cicids', 'CICIDS2017.csv'), ('unsw', 'UNSW-NB15.csv')]:
            print(f'{domain}: reading original CSV', flush=True)
            source = root / 'data/raw' / filename
            raw = spark.read.option('header', True).option('inferSchema', True).csv(str(source))
            raw = raw.toDF(*normalized_names(raw.columns))
            original_names = raw.columns
            raw_count = raw.count()
            raw = raw.dropDuplicates().persist(StorageLevel.DISK_ONLY)
            dedup_count = raw.count()
            if domain == 'cicids':
                cleaned = raw.filter(
                    F.col('flow_bytes_s').isNotNull() & ~F.isnan('flow_bytes_s') &
                    (F.abs(F.col('flow_bytes_s')) != float('inf')) &
                    F.col('flow_packets_s').isNotNull() & ~F.isnan('flow_packets_s') &
                    (F.abs(F.col('flow_packets_s')) != float('inf')))
            else:
                cleaned = raw.fillna(0, subset=['ct_flw_http_mthd', 'is_ftp_login'])
                cleaned = cleaned.withColumn('is_ftp_login', F.when(F.col('is_ftp_login') > 1, 1).otherwise(F.col('is_ftp_login')))
                cleaned = cleaned.withColumn('service', F.when(F.col('service') == '-', 'unknown').otherwise(F.col('service')))
                cleaned = cleaned.withColumn('attack_cat', F.when(F.col('label') == 0, 'Normal')
                    .when(F.col('attack_cat').isNull() | (F.trim('attack_cat') == ''), 'Normal')
                    .otherwise(F.trim('attack_cat')))
                cleaned = cleaned.withColumn('ct_ftp_cmd', F.when(F.trim('ct_ftp_cmd') == '', 0)
                                             .otherwise(F.col('ct_ftp_cmd')).cast('int'))
            cleaned = cleaned.persist(StorageLevel.DISK_ONLY)
            cleaned_count = cleaned.count()
            old_parts = {part: spark.read.parquet(str(root / f'data/splits/{domain}_{part}'))
                         for part in ('train', 'val', 'test')}
            keys = [c for c in old_parts['train'].columns if c != 'label']
            # UNSW attack_cat is historical split-key provenance only; never a scored feature.
            if domain == 'cicids':
                manifest['column_removals'] = notebook_removals(root / 'notebooks/cicids.ipynb', original_names, old_parts['train'].columns)
            for field in old_parts['train'].schema.fields:
                if field.name in keys:
                    cleaned = cleaned.withColumn(field.name, F.col(field.name).cast(field.dataType))
            print(f'{domain}: {raw_count} raw, {dedup_count} deduplicated, {cleaned_count} cleaned; verifying boundaries', flush=True)
            joined, quarantine, membership, provenance = attach_boundaries(cleaned, old_parts, keys, quarantine_ambiguous)
            # Target labels were needed only to replay historical full-row deduplication.
            # They are not matched, imputed, exported, or consumed by selection.
            if domain == 'cicids':
                joined = joined.drop('label')
                quarantine = quarantine.drop('label')
            else:
                joined = joined.drop('attack_cat').withColumn('label', F.col('label').cast('int'))
                quarantine = quarantine.drop('attack_cat')
            if provenance['ambiguous_rows']:
                quarantine.write.mode('errorifexists').parquet(str(output / f'{domain}_clean_unpruned/quarantine'))
                print(f'{domain}: quarantined {provenance["ambiguous_rows"]} rows with unresolvable split membership', flush=True)
            split_counts = {}
            for part, old in old_parts.items():
                destination = output / f'{domain}_clean_unpruned/splits/{domain}_{part}'
                frame = joined.filter(F.col('_split') == part).drop('_split')
                frame.write.mode('errorifexists').parquet(str(destination))
                expected = provenance['usable_counts'][part]
                actual = spark.read.parquet(str(destination)).count()
                if actual != expected:
                    raise ValueError(f'Output count mismatch for {domain}/{part}')
                split_counts[part] = actual
                print(f'{domain}/{part}: {actual} rows preserved', flush=True)
            manifest['datasets'][domain] = dict(raw_path=str(source.relative_to(root)),
                raw_sha256=sha256_file(source), raw_rows=raw_count, deduplicated_rows=dedup_count,
                cleaned_rows=cleaned_count, boundary_columns=keys, split_counts=split_counts,
                provenance=provenance, original_split_counts={part: old.count() for part, old in old_parts.items()},
                recovered_columns=[c for c in original_names if c not in old_parts['train'].columns],
                old_splits={part: {f.name: sha256_file(f) for f in sorted((root / f'data/splits/{domain}_{part}').glob('*.parquet'))} for part in old_parts},
                output_columns=[c for c in joined.columns if c != '_split'])
            membership.unpersist()
            cleaned.unpersist()
            raw.unpersist()
        manifest['status'] = 'complete_with_quarantine' if any(d['provenance']['ambiguous_rows'] for d in manifest['datasets'].values()) else 'complete'
        manifest['builder_sha256'] = sha256_file(__file__)
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        (output / '_SUCCESS').write_text('Verified exact feature-key multisets and split counts.\n')
        print(f'Completed {output}', flush=True)
    finally:
        spark.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--quarantine-ambiguous', action='store_true', help='Exclude unresolved groups from all splits instead of aborting')
    args = parser.parse_args()
    regenerate(args.root, args.output, args.quarantine_ambiguous)


if __name__ == '__main__':
    main()
