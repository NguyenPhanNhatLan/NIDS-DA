"""Development MLP baseline: streamed CIC-UNSW -> local CICIDS flow CSV."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from evaluation.baseline import collect_scores, compute_metrics, select_f1_threshold
from features.kaggle_stream import DEFAULT_DATASET, iter_kaggle_csv
from models.baseline import BaselineMLP
from models.proposal_pipeline import proposal_processor
from training.baseline import set_seed, train_baseline

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ['flow_duration_us', 'fwd_packets', 'bwd_packets', 'fwd_bytes', 'bwd_bytes']
ALIASES = [
    ['Flow Duration'],
    ['Total Fwd Packet', 'Total Fwd Packets'],
    ['Total Bwd packets', 'Total Backward Packets'],
    ['Total Length of Fwd Packet', 'Total Length of Fwd Packets'],
    ['Total Length of Bwd Packet', 'Total Length of Bwd Packets'],
]


def flow_table(frame):
    """Same CICFlowMeter flow quantities in both domains; duration stays in us."""
    frame = frame.rename(columns=lambda c: c.strip())
    columns = []
    for aliases in ALIASES:
        matches = [name for name in aliases if name in frame.columns]
        if len(matches) != 1:
            raise ValueError(f'Expected one feature from {aliases}; found {matches}')
        columns.append(matches[0])
    if 'Label' not in frame:
        raise ValueError('Missing Label column')
    labels = frame['Label'].astype('string').str.strip().str.casefold()
    if labels.isna().any() or labels.eq('').any():
        raise ValueError('Missing labels; refusing to treat them as attacks')
    normal = labels.isin(['normal', 'benign', '0', '0.0'])
    # Named non-normal attack categories (e.g. Exploits) map to binary Attack.
    values = frame[columns].apply(pd.to_numeric, errors='coerce')
    values.columns = FEATURES
    values = values.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    values['label'] = (~normal).astype(np.int64)
    return values


def sample_chunks(chunks, max_rows, seed):
    """Uniform priority reservoir across the entire stream, not its first rows."""
    rng = np.random.default_rng(seed)
    sample = None
    priorities = np.empty(0)
    seen = 0
    try:
        for index, chunk in enumerate(chunks, 1):
            table = flow_table(chunk)
            seen += len(table)
            sample = table if sample is None else pd.concat([sample, table], ignore_index=True)
            priorities = np.concatenate((priorities, rng.random(len(table))))
            if len(sample) > max_rows:
                keep = np.argpartition(priorities, max_rows - 1)[:max_rows]
                sample = sample.iloc[keep].reset_index(drop=True)
                priorities = priorities[keep]
            if index % 10 == 0:
                print(f'Streamed {seen:,} rows; RAM sample {len(sample):,}', flush=True)
    finally:
        close = getattr(chunks, 'close', None)
        if close:
            close()
    if sample is None or sample.empty:
        raise ValueError('Empty dataset')
    print(f'Stream complete: {seen:,} rows; sampled {len(sample):,}', flush=True)
    return sample, seen


def split_source(table, seed):
    # Identical selected feature vectors stay together, even with differing labels.
    groups = pd.util.hash_pandas_object(table[FEATURES], index=False).to_numpy()
    train, val = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
                      .split(table, groups=groups))
    for indices in (train, val):
        if table.iloc[indices]['label'].nunique() != 2:
            raise ValueError('Source split lacks one class; increase --sample-rows')
    return train, val


def loader(x, y, batch_size, train=False):
    return DataLoader(TensorDataset(torch.from_numpy(np.array(x, dtype=np.float32, copy=True)),
                                   torch.from_numpy(np.array(y, dtype=np.int64, copy=True))),
                      batch_size=batch_size, shuffle=train, drop_last=train)


def fingerprint(table):
    return hashlib.sha256(pd.util.hash_pandas_object(table, index=False)
                          .to_numpy().tobytes()).hexdigest()


def run(args):
    if args.sample_rows < 100 or args.epochs < 1 or args.batch_size < 2:
        raise ValueError('Require sample-rows >= 100, epochs >= 1, batch-size >= 2')
    if args.output_dir.exists():
        raise FileExistsError(f'Output exists: {args.output_dir}; choose another --output-dir')
    if not args.cicids_csv.is_file():
        raise FileNotFoundError(args.cicids_csv)
    set_seed(args.seed)
    source, source_rows = sample_chunks(
        iter_kaggle_csv('CICFlowMeter.csv', chunksize=50000), args.sample_rows, args.seed)
    target, target_rows = sample_chunks(
        pd.read_csv(args.cicids_csv, chunksize=50000), args.sample_rows, args.seed + 1)
    train_idx, val_idx = split_source(source, args.seed)
    train, val = source.iloc[train_idx], source.iloc[val_idx]
    if len(train) < args.batch_size:
        raise ValueError('Not enough source training rows for a full batch')
    if target['label'].nunique() != 2:
        raise ValueError('Target sample lacks one class; increase --sample-rows')
    processor = proposal_processor()
    train_x = np.asarray(processor.fit_transform(train[FEATURES]), dtype=np.float32)
    val_x = np.asarray(processor.transform(val[FEATURES]), dtype=np.float32)
    target_x = np.asarray(processor.transform(target[FEATURES]), dtype=np.float32)
    for x in (train_x, val_x, target_x):
        if not np.isfinite(x).all():
            raise ValueError('Non-finite preprocessed features')
    counts = np.bincount(train['label'], minlength=2).tolist()
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'CIC-UNSW→CICIDS | Device={device} | Features={len(FEATURES)} | '
          f'Source train={len(train):,} val={len(val):,} | Counts={counts}', flush=True)
    source_val_loader = loader(val_x, val['label'].to_numpy(), 1024)
    model, best_epoch, best_ap = train_baseline(
        BaselineMLP(len(FEATURES)).to(device), counts,
        loader(train_x, train['label'].to_numpy(), args.batch_size, True),
        source_val_loader, epochs=args.epochs, patience=args.patience)
    labels, scores = collect_scores(model, source_val_loader)
    threshold = select_f1_threshold(labels, scores)
    source_metrics = compute_metrics(labels, scores, threshold)
    target_labels, target_scores = collect_scores(
        model, loader(target_x, target['label'].to_numpy(), 1024))
    target_metrics = compute_metrics(target_labels, target_scores, threshold)
    result = {
        'protocol': 'cicflow_source_only_development_v1',
        'direction': 'cic_unsw_to_cicids', 'seed': args.seed,
        'features': FEATURES, 'feature_count': len(FEATURES),
        'extractor_family': 'CICFlowMeter', 'same_extractor_version_verified': False,
        'extractor_note': 'Published flow CSVs; not re-extracted with local pinned commit.',
        'source_dataset': DEFAULT_DATASET, 'source_file': 'CICFlowMeter.csv',
        'target_csv': str(args.cicids_csv),
        'source_rows_streamed': source_rows, 'target_rows_streamed': target_rows,
        'sampling': 'uniform priority reservoir across whole dataset',
        'source_sample_rows': len(source), 'target_sample_rows': len(target),
        'source_sample_sha256': fingerprint(source), 'target_sample_sha256': fingerprint(target),
        'source_train_rows': len(train), 'source_val_rows': len(val),
        'source_split': '80/20 feature-group holdout; identical vectors stay together',
        'source_train_counts': counts, 'best_epoch': best_epoch,
        'best_source_val_ap': float(best_ap), 'threshold_from_source_val': float(threshold),
        'target_labels_used_training': False, 'target_labels_used_selection': False,
        'evaluation_role': 'development_only; source val used for model/threshold selection',
        'source_val': source_metrics, 'target_development': target_metrics,
    }
    args.output_dir.mkdir(parents=True)
    joblib.dump(processor, args.output_dir / 'preprocessor.joblib')
    torch.save({'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'input_dim': len(FEATURES), 'features': FEATURES,
                'threshold': float(threshold), 'seed': args.seed}, args.output_dir / 'model.pt')
    with (args.output_dir / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('\nMetric | Source val | CICIDS development')
    for metric in ['pr_auc', 'roc_auc', 'macro_f1', 'recall', 'fpr']:
        print(f'{metric:10s} | {source_metrics[metric]:.6f} | {target_metrics[metric]:.6f}')
    print(f'Saved model, preprocessor and result: {args.output_dir}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cicids-csv', type=Path, default=ROOT / 'data/raw/CICIDS2017.csv')
    parser.add_argument('--sample-rows', type=int, default=100000)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/experiment_b/cicflow_baseline/seed42')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
