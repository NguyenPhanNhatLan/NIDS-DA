"""Replay split membership and demonstrate type-sensitive Spark hash differences."""
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from features.common_features import DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parents[2]


def rotate(x, shift):
    return (x << np.uint64(shift)) | (x >> np.uint64(64 - shift))


def numeric_hash(values, seed, is_int=False):
    """Vectorized numeric XXH64 primitives, as specified by Apache Spark XXH64.

    Reference: Apache Spark v3.5.0 XXH64.java and expressions/hash.scala.
    Null skips the hash step; double +/-0 hashes identically.
    """
    p1, p2, p3, p4, p5 = map(np.uint64, (
        0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F,
        0x165667B19E3779F9, 0x85EBCA77C2B2AE63, 0x27D4EB2F165667C5,
    ))
    values = np.asarray(values, dtype=np.float64)
    missing = np.isnan(values)
    safe = np.where(missing | (values == 0), 0., values)
    with np.errstate(over='ignore'):
        h = seed + p5 + np.uint64(4 if is_int else 8)
        if is_int:
            bits = safe.astype(np.int64).astype(np.uint64) & np.uint64(0xffffffff)
            h ^= bits * p1
            h = rotate(h, 23) * p2 + p3
        else:
            bits = safe.view(np.uint64)
            h ^= rotate(bits * p2, 31) * p1
            h = rotate(h, 27) * p1 + p4
        h ^= h >> np.uint64(33)
        h *= p2
        h ^= h >> np.uint64(29)
        h *= p3
        h ^= h >> np.uint64(32)
    return np.where(missing, seed, h)


def row_hash(batch, keys, types):
    h = np.full(batch.num_rows, 42, dtype=np.uint64)
    for name, is_int in zip(keys, types):
        h = numeric_hash(batch.column(name).to_numpy(zero_copy_only=False), h, is_int)
    return numeric_hash(np.full(batch.num_rows, 42.), h, True)


def run(output, work_root):
    result = {'engine': 'NumPy replay of Spark numeric XXH64 primitives; null skipped, +/-0 normalized',
              'reference': 'https://github.com/apache/spark/blob/v3.5.0/sql/catalyst/src/main/java/org/apache/spark/sql/catalyst/expressions/XXH64.java'}
    for domain in ('unsw', 'cicids'):
        keys = [item[domain] for item in json.loads(DEFAULT_CONFIG.read_text())]
        domain_result = {}
        for name, root in (('new', Path(work_root) / 'splits'),):
            checks = {}
            total = 0
            for split in ('train', 'val', 'test'):
                wrong = 0
                files = sorted((root / f'{domain}_{split}').glob('*.parquet'))
                if not files:
                    raise FileNotFoundError(root / f'{domain}_{split}')
                for path in files:
                    parquet = pq.ParquetFile(path)
                    types = [pa.types.is_integer(parquet.schema_arrow.field(k).type) for k in keys]
                    for batch in parquet.iter_batches(batch_size=65536, columns=keys):
                        h = row_hash(batch, keys, types)
                        bucket = h.view(np.int64) % 100
                        correct = bucket < 70 if split == 'train' else ((bucket >= 70) & (bucket < 85)) if split == 'val' else bucket >= 85
                        wrong += int(np.count_nonzero(~correct))
                        if name == 'new':
                            total += batch.num_rows
                checks[split] = wrong
            domain_result[f'{name}_split_bucket_mismatches'] = checks
            if name == 'new':
                domain_result['rows_replayed'] = total
        result[domain] = domain_result
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result
