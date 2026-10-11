"""Separate FunnelMLP source-only/MK-MMD experiment on frozen canonical data."""
import argparse
import json
import math
from pathlib import Path

import torch

from evaluation.baseline import collect_scores, compute_metrics, select_f1_threshold
from features.common_features import COMMON_FEATURES
from models.funnel import NIDSFunnelMLP
from training.baseline import set_seed, train_baseline
from training.data_revision import verify_revision, revision_path, require_development_open
from training.proposal_data import ParquetBatchStream, split_sha256
from training.proposal_mmd import train_mmd, load_config, sha256

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/proposal_funnel_mkmmd.json'


def run(direction, seed, stage):
    require_development_open()
    revision = verify_revision()
    if not revision:
        raise ValueError('Funnel experiments require a frozen data revision')
    source, target = ('unsw', 'cicids') if direction == 'unsw_to_cicids' else ('cicids', 'unsw')
    base = revision_path('feature_root', None) / direction
    model_root = revision_path('model_root', None) / 'funnel_5d'
    result_root = revision_path('result_root', None) / 'funnel_5d'
    checkpoint_path = model_root / stage / direction / f'seed{seed}.pt'
    result_path = result_root / stage / direction / f'seed{seed}.json'
    for path in (checkpoint_path, result_path):
        if path.exists():
            raise FileExistsError(f'Output already exists: {path}')
    config = load_config(CONFIG)
    settings = config['training']
    batch_size = settings['batch_size']
    paths = {
        'source_train': base / f'{source}_train',
        'source_val': base / f'{source}_val',
        'target_train': base / f'{target}_train',
        'target_val': base / f'{target}_val',
    }
    hashes = {name: split_sha256(path) for name, path in paths.items()}
    config_hash = sha256(CONFIG)
    counts = torch.zeros(2, dtype=torch.long)
    for _, labels in ParquetBatchStream(paths['source_train'], 4096, False, seed, True):
        counts += torch.bincount(labels, minlength=2)
    if (counts == 0).any():
        raise ValueError('Source training needs both classes')
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NIDSFunnelMLP(len(COMMON_FEATURES)).to(device)
    train = ParquetBatchStream(paths['source_train'], batch_size, True, seed, True, drop_last=True)
    val = ParquetBatchStream(paths['source_val'], 1024, False, seed, True)
    history = []
    source_hash = None
    if stage == 'source-only':
        model, best_epoch, best_ap = train_baseline(
            model, counts.tolist(), train, val, epochs=settings['epochs'],
            lr=settings['learning_rate'], patience=settings['patience'],
            min_delta=settings['min_delta'],
        )
    else:
        source_path = model_root / 'source-only' / direction / f'seed{seed}.pt'
        source_hash = sha256(source_path)
        checkpoint = torch.load(source_path, map_location='cpu', weights_only=True)
        if (checkpoint['direction'] != direction or checkpoint['seed'] != seed
                or checkpoint['prepared_split_sha256'] != hashes
                or checkpoint['config_sha256'] != config_hash
                or checkpoint['data_manifest_sha256'] != revision['data_manifest_sha256']):
            raise ValueError('Funnel source checkpoint provenance differs from this run')
        model.load_state_dict(checkpoint['model_state_dict'])
        target_train = ParquetBatchStream(
            paths['target_train'], batch_size, True, seed, False, drop_last=True,
        )
        model, best_epoch, best_ap, history = train_mmd(
            model, counts.tolist(), train, target_train, val, config,
            expected_source_ap=checkpoint['best_source_val_ap'],
        )
    labels, scores = collect_scores(model, val)
    threshold = select_f1_threshold(labels, scores)
    source_metrics = compute_metrics(labels, scores, threshold)
    target_labels, target_scores = collect_scores(model, ParquetBatchStream(
        paths['target_val'], 1024, False, seed, True,
    ))
    metadata = {
        **revision, 'architecture': 'NIDSFunnelMLP', 'input_dim': len(COMMON_FEATURES),
        'features': list(COMMON_FEATURES), 'latent_dim': 5, 'latent_l2_normalized': True,
        'direction': direction, 'seed': seed, 'method': stage,
        'config_sha256': config_hash, 'prepared_split_sha256': hashes,
        'source_checkpoint_sha256': source_hash,
        'best_epoch': best_epoch, 'best_source_val_ap': best_ap,
        'selected_stage': ('source_pretrained' if best_epoch == 0 else 'adaptation')
                          if stage == 'mk-mmd' else 'source_only',
        'threshold': threshold, 'threshold_selection': 'source validation F1',
        'checkpoint_selection': 'source validation AP',
        'target_labels_used_training': False, 'target_labels_used_selection': False,
        'data_role': 'target_validation_development_only',
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**metadata, 'model_state_dict': model.state_dict(), 'history': history}, checkpoint_path)
    result = {**metadata, 'source_val': source_metrics,
              'target_development': compute_metrics(target_labels, target_scores, threshold),
              'checkpoint': str(checkpoint_path), 'history': history}
    with result_path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f'Saved: {result_path}')


def evaluate(direction, seed):
    """Replay both saved checkpoints on validation only; never retrain."""
    from evaluation.proposal_score_diagnostic import score_diagnostics

    revision = verify_revision()
    if not revision:
        raise ValueError('Funnel evaluation requires a frozen data revision')
    source, target = ('unsw', 'cicids') if direction == 'unsw_to_cicids' else ('cicids', 'unsw')
    base = revision_path('feature_root', None) / direction
    model_root = revision_path('model_root', None) / 'funnel_5d'
    output = revision_path('result_root', None) / 'funnel_5d/evaluation' / direction / f'seed{seed}.json'
    if output.exists():
        raise FileExistsError(f'Evaluation already exists: {output}')
    hashes = {name: split_sha256(base / f'{domain}_{split}')
              for name, domain, split in (
                  ('source_train', source, 'train'), ('source_val', source, 'val'),
                  ('target_train', target, 'train'), ('target_val', target, 'val'))}
    checkpoints = {}
    checkpoint_paths = {}
    # Validate both artifacts before accessing target validation labels.
    for stage in ('source-only', 'mk-mmd'):
        path = model_root / stage / direction / f'seed{seed}.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        expected = {
            'architecture': 'NIDSFunnelMLP', 'input_dim': len(COMMON_FEATURES),
            'features': list(COMMON_FEATURES), 'latent_dim': 5, 'latent_l2_normalized': True,
            'direction': direction, 'seed': seed, 'method': stage,
            'data_manifest_sha256': revision['data_manifest_sha256'],
            'config_sha256': sha256(CONFIG), 'prepared_split_sha256': hashes,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f'Checkpoint provenance mismatch: {path}: {key}')
        checkpoints[stage], checkpoint_paths[stage] = checkpoint, path
    if checkpoints['mk-mmd']['source_checkpoint_sha256'] != sha256(checkpoint_paths['source-only']):
        raise ValueError('MK-MMD was initialized from a different source-only checkpoint')
    models, source_results = {}, {}
    for stage, checkpoint in checkpoints.items():
        model = NIDSFunnelMLP(len(COMMON_FEATURES))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        labels, scores = collect_scores(model, ParquetBatchStream(
            base / f'{source}_val', 1024, False, seed, True))
        threshold = checkpoint['threshold']
        if not math.isclose(select_f1_threshold(labels, scores), threshold, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f'Source threshold replay differs: {stage}')
        metrics = compute_metrics(labels, scores, threshold)
        if not math.isclose(metrics['pr_auc'], checkpoint['best_source_val_ap'], rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(f'Source validation AP replay differs: {stage}')
        models[stage], source_results[stage] = model, metrics
    result = {
        **revision, 'direction': direction, 'seed': seed, 'architecture': 'NIDSFunnelMLP',
        'data_role': 'target_validation_development_only', 'runs': {},
    }
    for stage, model in models.items():
        checkpoint = checkpoints[stage]
        labels, scores = collect_scores(model, ParquetBatchStream(
            base / f'{target}_val', 1024, False, seed, True))
        metrics = compute_metrics(labels, scores, checkpoint['threshold'])
        result['runs'][stage] = {
            'checkpoint': str(checkpoint_paths[stage]),
            'checkpoint_sha256': sha256(checkpoint_paths[stage]),
            'selected_stage': checkpoint['selected_stage'],
            'threshold': checkpoint['threshold'], 'source_val': source_results[stage],
            'target_development': metrics,
            'score_analysis': score_diagnostics(labels, scores, checkpoint['threshold']),
        }
        print(f"{stage}: target AP={metrics['pr_auc']:.6f}, recall={metrics['recall']:.6f}, FPR={metrics['fpr']:.6f}")
    before = result['runs']['source-only']['target_development']
    after = result['runs']['mk-mmd']['target_development']
    result['target_delta'] = {key: after[key] - before[key]
                              for key in ('pr_auc', 'roc_auc', 'macro_f1', 'recall', 'fpr')}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"Delta target AP={result['target_delta']['pr_auc']:+.6f}")
    print(f'Saved: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction', required=True, choices=['unsw_to_cicids', 'cicids_to_unsw'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stage', required=True, choices=['source-only', 'mk-mmd', 'evaluate'])
    args = parser.parse_args()
    if args.stage == 'evaluate':
        evaluate(args.direction, args.seed)
    else:
        run(args.direction, args.seed, args.stage)


if __name__ == '__main__':
    main()
