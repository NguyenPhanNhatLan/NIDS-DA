"""Create exact-size proportional attack-category samples from raw CSVs."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def strata(frame, domain):
    labels = frame['Label'].astype('string').str.strip()
    if labels.isna().any() or labels.eq('').any():
        raise ValueError('Missing Label')
    if domain == 'unsw':
        categories = frame['attack_cat'].astype('string').str.strip()
        normal = labels.isin(['0', '0.0'])
        categories = categories.fillna('Unknown-Attack').replace('', 'Unknown-Attack')
        return categories.where(~normal, 'Normal')
    return labels


def allocate(counts, size):
    total = sum(counts.values())
    if total < size:
        raise ValueError(f'Only {total} rows; requested {size}')
    exact = {key: count * size / total for key, count in counts.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = size - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def sample(path, domain, output, size, seed):
    csv_output = output / f'{domain}_500k.csv'
    report_output = output / f'{domain}_sampling_report.json'
    for destination in (csv_output, report_output):
        if destination.exists():
            raise FileExistsError(destination)
    columns = ['Label', 'attack_cat'] if domain == 'unsw' else ['Label']
    counts = Counter()
    for frame in pd.read_csv(path, usecols=columns, chunksize=100000, dtype=str, keep_default_na=False, low_memory=False):
        counts.update(strata(frame, domain).value_counts().to_dict())
    quotas = allocate(counts, size)
    print(f'{domain}: raw={sum(counts.values()):,}; quotas={quotas}', flush=True)
    rng = np.random.default_rng(seed)
    pools, priorities = {}, {}
    seen = 0
    for frame in pd.read_csv(path, chunksize=100000, low_memory=False, dtype=str, keep_default_na=False):
        groups = strata(frame, domain)
        for key, quota in quotas.items():
            if not quota:
                continue
            rows = frame.loc[groups.eq(key)]
            if rows.empty:
                continue
            scores = rng.random(len(rows))
            if key in pools:
                rows = pd.concat([pools[key], rows], ignore_index=True)
                scores = np.concatenate([priorities[key], scores])
            if len(rows) > quota:
                keep = np.argpartition(scores, quota - 1)[:quota]
                rows = rows.iloc[keep]
                scores = scores[keep]
            pools[key] = rows.reset_index(drop=True)
            priorities[key] = scores
        seen += len(frame)
        print(f'{domain}: scanned {seen:,}', flush=True)
    sampled = pd.concat(list(pools.values()), ignore_index=True)
    sampled = sampled.iloc[rng.permutation(len(sampled))].reset_index(drop=True)
    actual = strata(sampled, domain).value_counts().to_dict()
    assert len(sampled) == size
    assert actual == {key: quota for key, quota in quotas.items() if quota}
    output.mkdir(parents=True, exist_ok=True)
    with csv_output.open('x', encoding='utf-8', newline='') as stream:
        sampled.to_csv(stream, index=False)
    # Read back the saved labels to verify the actual deliverable.
    saved_counts = Counter()
    for frame in pd.read_csv(csv_output, usecols=columns, chunksize=100000, dtype=str, keep_default_na=False, low_memory=False):
        saved_counts.update(strata(frame, domain).value_counts().to_dict())
    assert dict(saved_counts) == actual
    digest = hashlib.sha256()
    with csv_output.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    report = {
        'source_csv': str(path), 'sample_csv': str(csv_output), 'seed': seed,
        'source_rows': sum(counts.values()), 'sample_rows': size,
        'method': 'uniform without replacement within label/attack-category strata',
        'quota_method': 'proportional largest remainder (rounding to integer counts)',
        'sample_sha256': digest.hexdigest(), 'readback_verified': True,
        'strata': [{'category': key, 'raw_count': counts[key],
                    'raw_ratio': counts[key]/sum(counts.values()),
                    'sample_count': quotas[key], 'sample_ratio': quotas[key]/size}
                   for key in sorted(counts)],
    }
    with report_output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(f'Saved and verified: {csv_output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=500000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error('--rows must be positive')
    for domain, filename in [('unsw', 'UNSW-NB15.csv'), ('cicids', 'CICIDS2017.csv')]:
        sample(ROOT / 'data/raw' / filename, domain, args.output_dir,
               args.rows, args.seed)


if __name__ == '__main__':
    main()
