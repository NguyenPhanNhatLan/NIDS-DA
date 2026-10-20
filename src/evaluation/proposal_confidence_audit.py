"""Audit source-only pseudo-label coverage on target train without reading labels."""

import argparse
import json
from pathlib import Path

from project_utils.result_files import result_file

import torch

from experiments.proposal_protocol import DIRECTIONS, SEEDS
from features.common_features import COMMON_FEATURES
from models.baseline import BaselineMLP
from training.proposal_class_aware import pseudo_label_counts
from training.data_revision import file_sha256, revision_path, verify_revision
from training.proposal_data import ParquetBatchStream


def batch_counts(logits, threshold):
    return pseudo_label_counts(logits, threshold)[2]


@torch.inference_mode()
def audit(direction, seed, thresholds, device):
    source, target = direction.split('_to_')
    checkpoint = revision_path('model_root', None) / 'source_only_target_val' / direction / f'seed{seed}.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    revision = verify_revision()
    if any(state.get(key) != value for key, value in revision.items()):
        raise ValueError(f'Checkpoint revision mismatch: {checkpoint}')
    if state['direction'] != direction or state['seed'] != seed or state['features'] != list(COMMON_FEATURES):
        raise ValueError(f'Checkpoint identity mismatch: {checkpoint}')
    model = BaselineMLP(len(COMMON_FEATURES)).to(device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    base = revision_path('feature_root', None) / direction

    source_loader = ParquetBatchStream(base / f'{source}_train', 256, True, seed, True, drop_last=True)
    target_path = base / f'{target}_train'
    target_loader = ParquetBatchStream(target_path, 256, True, seed + 1000, False, drop_last=True)
    rows = {t: {'accepted_per_class': [0, 0], 'eligible_batches': 0} for t in thresholds}
    iterator = iter(target_loader)
    batches = 0
    for _, source_labels in source_loader:
        try:
            features = next(iterator)
        except StopIteration:
            iterator = iter(target_loader)
            features = next(iterator)
        _, logits = model(features.to(device))
        source_counts = torch.bincount(source_labels, minlength=2).tolist()
        for threshold in thresholds:
            counts = batch_counts(logits, threshold)
            row = rows[threshold]
            row['accepted_per_class'] = [a + b for a, b in zip(row['accepted_per_class'], counts)]
            row['eligible_batches'] += int(all(s >= 2 and t >= 2 for s, t in zip(source_counts, counts)))
        batches += 1
    for row in rows.values():
        row['accepted_rate'] = sum(row['accepted_per_class']) / (batches * 256)
        row['eligible_batch_rate'] = row['eligible_batches'] / batches
    return {
        **revision, 'direction': direction, 'seed': seed,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha256(checkpoint),
        'device': str(device), 'comparison': 'confidence >= threshold',
        'scope': 'source-only model; simulated first adaptation epoch; counts are sample exposures, not unique target rows',
        'batch_size': 256, 'batches': batches, 'target_labels_read': False,
        'thresholds': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directions', nargs='+', choices=DIRECTIONS, default=list(DIRECTIONS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.8, 0.85, 0.9, 0.95])
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output:
        args.output = result_file(args.output)
    if not args.thresholds or any(not 0 <= t <= 1 for t in args.thresholds):
        parser.error('thresholds must be in [0, 1]')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    results = []
    for direction in args.directions:
        for seed in args.seeds:
            result = audit(direction, seed, args.thresholds, torch.device(args.device))
            results.append(result)
            print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump({'analysis': 'development_confidence_audit', 'runs': results}, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
