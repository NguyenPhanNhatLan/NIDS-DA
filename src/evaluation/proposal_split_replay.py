"""Replay canonical split membership with Spark's native typed xxhash64."""
import json
from pathlib import Path

from features.common_features import DEFAULT_CONFIG


def split_bucket(frame, keys):
    from pyspark.sql import functions as F
    return F.pmod(F.xxhash64(*[F.col(name) for name in keys], F.lit(42)), F.lit(100))


def run(output, work_root):
    from pyspark.sql import functions as F
    from spark_session import get_spark
    spark = get_spark()
    result = {'engine': 'Spark native xxhash64; no custom hash implementation',
              'spark_version': spark.version, 'seed': 42}
    config = json.loads(DEFAULT_CONFIG.read_text())
    try:
        for domain in ('unsw', 'cicids'):
            keys = [item[domain] for item in config]
            checks, rows, types = {}, 0, {}
            for split in ('train', 'val', 'test'):
                path = Path(work_root) / 'splits' / f'{domain}_{split}'
                if not list(path.glob('*.parquet')):
                    raise FileNotFoundError(path)
                frame = spark.read.parquet(str(path))
                schema = {name: frame.schema[name].dataType.simpleString() for name in keys}
                if types and schema != types:
                    raise ValueError(f'Split-key types differ across {domain} splits')
                types = schema
                bucket = split_bucket(frame, keys)
                correct = bucket < 70 if split == 'train' else ((bucket >= 70) & (bucket < 85)) if split == 'val' else bucket >= 85
                counts = frame.agg(F.count('*').alias('rows'),
                                   F.sum(F.when(~correct, 1).otherwise(0)).alias('wrong')).first()
                rows += counts['rows']
                checks[split] = int(counts['wrong'] or 0)
            result[domain] = {'split_bucket_mismatches': checks,
                              'rows_replayed': rows, 'split_key_types': types}
    finally:
        spark.stop()
    result['quality_gate_passed'] = all(not any(result[d]['split_bucket_mismatches'].values())
                                        for d in ('unsw', 'cicids'))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    if not result['quality_gate_passed']:
        raise ValueError(f'Spark split replay failed; inspect {output}')
    return result
