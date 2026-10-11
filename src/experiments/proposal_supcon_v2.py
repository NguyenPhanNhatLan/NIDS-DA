"""Projected SupCon E1/E2 extension, separate from frozen proposal_v2 and v1."""
import argparse
import csv
import json
import math
from pathlib import Path

import torch

from evaluation.baseline import collect_scores, compute_metrics, select_f1_threshold
from experiments.proposal_supcon_extension import reference, provenance
from features.common_features import COMMON_FEATURES
from models.baseline import BaselineMLP
from models.contrastive_projection import ContrastiveProjection
from training.data_revision import file_sha256, require_development_open, revision_path
from training.proposal_data import ParquetBatchStream
from training.supcon_adaptation import initialize, source_evaluation, train as train_model

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'supcon_mkmmd_ext_v2'
CONFIGS = {'E1': ROOT / 'configs/proposal_supcon.json', 'E2': ROOT / 'configs/proposal_supcon_mkmmd.json'}
DIRECTIONS = ('unsw_to_cicids', 'cicids_to_unsw')
METRICS = ('pr_auc', 'roc_auc', 'macro_f1', 'precision', 'recall', 'fpr')


def load_config(path, experiment):
    config = json.loads(path.read_text())
    for key, value in {'extension_id': VERSION, 'experiment': experiment, 'architecture': 'BaselineMLP',
                       'feature_count': 5, 'latent_dim': 168, 'projection_dim': 64}.items():
        if config.get(key) != value:
            raise ValueError(f'Unsupported {key}: {path}')
    for key in ('alpha_supcon', 'lambda_mmd'):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'Invalid {key}')
    if config['temperature'] <= 0 or not math.isfinite(config['temperature']):
        raise ValueError('Invalid temperature')
    if experiment == 'E1' and config['lambda_mmd'] != 0:
        raise ValueError('E1 must not align domains')
    if experiment == 'E2' and config['lambda_mmd'] <= 0:
        raise ValueError('E2 requires positive MK-MMD weight')
    if type(config['mmd_warmup_epochs']) is not int or config['mmd_warmup_epochs'] < 0:
        raise ValueError('Warm-up must be nonnegative; 0 disables it')
    if not config['kernel_scales'] or any(not math.isfinite(s) or s <= 0 for s in config['kernel_scales']):
        raise ValueError('Invalid kernel scales')
    for key, minimum in (('batch_size', 2), ('epochs', 1), ('patience', 1)):
        if type(config['training'][key]) is not int or config['training'][key] < minimum:
            raise ValueError(f'Invalid training {key}')
    for key in ('learning_rate', 'weight_decay', 'min_delta'):
        if not math.isfinite(config['training'][key]) or config['training'][key] < 0:
            raise ValueError(f'Invalid training {key}')
    if config['training']['learning_rate'] == 0:
        raise ValueError('Learning rate must be positive')
    for key, minimum in (('sample_rows', 2), ('mmd_sample', 2), ('mmd_repeats', 1), ('cv_folds', 2), ('gradient_batches', 1)):
        if type(config['diagnostics'][key]) is not int or config['diagnostics'][key] < minimum:
            raise ValueError(f'Invalid diagnostic {key}')
    return config


def device_for(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is not available; use --device cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    return torch.device(name)


def artifact_paths(experiment, direction, seed):
    return (revision_path('model_root', None) / 'extensions' / VERSION / experiment / direction / f'seed{seed}.pt',
            revision_path('result_root', None) / 'extensions' / VERSION / experiment / direction / f'seed{seed}.json')


def meta_for(validated, source, direction, seed, config_path):
    meta = provenance(validated, source, direction, seed, config_path)
    meta['extension_id'] = VERSION
    meta['projection_dim'] = 64
    return meta


def split_paths(validated, direction):
    source = 'unsw' if direction == 'unsw_to_cicids' else 'cicids'
    return {name: validated['base'] / f'{domain}_{split}' for name, domain, split in (
        ('source_train', source, 'train'), ('source_val', source, 'val'),
        ('target_train', validated['target'], 'train'), ('target_val', validated['target'], 'val'))}


def run_train(experiment, direction, seed, config_path, device_name):
    require_development_open()
    config = load_config(config_path, experiment)
    paths_out = artifact_paths(experiment, direction, seed)
    if any(path.exists() for path in paths_out):
        raise FileExistsError(f'Existing/incomplete run: {paths_out}')
    validated, checkpoint, baseline = reference(direction, seed)
    paths = split_paths(validated, direction)
    model, projection = initialize(checkpoint['model_state_dict'], len(COMMON_FEATURES), seed, device_for(device_name))
    counts = torch.zeros(2, dtype=torch.long)
    for _, y in ParquetBatchStream(paths['source_train'], 4096, False, seed, True):
        counts += torch.bincount(y, minlength=2)
    batch = config['training']['batch_size']
    source_loader = ParquetBatchStream(paths['source_train'], batch, True, seed, True, drop_last=True)
    target_loader = (ParquetBatchStream(paths['target_train'], batch, True, seed + 1000, False, drop_last=True)
                     if experiment == 'E2' else None)
    validation = ParquetBatchStream(paths['source_val'], 1024, False, seed, True)
    epoch, ap, history = train_model(model, projection, source_loader, target_loader, validation,
                                    counts.tolist(), config, validated['source_val_ap'])
    source_metrics = source_evaluation(model, validation)
    threshold = source_metrics['threshold']
    # Post-training target validation reporting only; test is not accessed.
    ty, ts = collect_scores(model, ParquetBatchStream(paths['target_val'], 1024, False, seed, True))
    target_metrics = compute_metrics(ty, ts, threshold)
    meta = {**meta_for(validated, checkpoint, direction, seed, config_path),
            'experiment': experiment, 'config': config, 'history': history,
            'best_epoch': epoch, 'best_source_val_ap': ap, 'threshold': threshold,
            'selected_stage': 'source_pretrained' if epoch == 0 else 'adaptation',
            'source_train_counts': counts.tolist(), 'bn_running_stats_frozen': True,
            'checkpoint_selection': 'source validation AP including epoch0',
            'threshold_selection': 'source validation Attack F1',
            'target_labels_used_training': False, 'target_labels_used_checkpoint_selection': False,
            'target_labels_used_threshold_selection': False}
    model_path, result_path = paths_out
    for path in paths_out:
        path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open('xb') as stream:
        torch.save({**meta,
                    'model_state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    'projection_state_dict': {key: value.detach().cpu() for key, value in projection.state_dict().items()}}, stream)
    result = {**meta, 'checkpoint': str(model_path), 'checkpoint_sha256': file_sha256(model_path),
              'data_role': 'target_validation_development_only', 'source_val': source_metrics,
              'target_development': target_metrics, 'target_development_split': f"{validated['target']}_val",
              'paired_delta_vs_source_only': {key: target_metrics[key] - baseline['target_development'][key] for key in METRICS}}
    with result_path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f'Saved: {result_path}')


def load_run(experiment, direction, seed, config_path, validated, source):
    load_config(config_path, experiment)
    model_path, result_path = artifact_paths(experiment, direction, seed)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    result = json.loads(result_path.read_text())
    expected = {**meta_for(validated, source, direction, seed, config_path), 'experiment': experiment,
                'target_labels_used_training': False, 'target_labels_used_checkpoint_selection': False,
                'target_labels_used_threshold_selection': False}
    for artifact in (checkpoint, result):
        if any(artifact.get(key) != value for key, value in expected.items()):
            raise ValueError('Extension provenance mismatch')
    if result['checkpoint'] != str(model_path) or result['checkpoint_sha256'] != file_sha256(model_path):
        raise ValueError('Checkpoint/result hash mismatch')
    for key in ('best_epoch', 'best_source_val_ap', 'threshold', 'selected_stage'):
        if result[key] != checkpoint[key]:
            raise ValueError(f'Selection differs: {key}')
    # Classification loads only the unmodified BaselineMLP state dictionary.
    model = BaselineMLP(len(COMMON_FEATURES))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    metrics = source_evaluation(model, ParquetBatchStream(split_paths(validated, direction)['source_val'], 1024, False, seed, True))
    if (not math.isclose(metrics['pr_auc'], checkpoint['best_source_val_ap'], abs_tol=1e-3, rel_tol=0)
            or not math.isclose(metrics['threshold'], checkpoint['threshold'], abs_tol=1e-6, rel_tol=0)):
        raise ValueError('Source AP/threshold replay differs')
    return model, checkpoint, result, metrics


def run_evaluate(direction, seed, configs, output_name, diagnostics=True):
    from evaluation.proposal_score_diagnostic import score_diagnostics
    validated, source, baseline = reference(direction, seed)
    output = revision_path('result_root', None) / 'extensions' / VERSION / output_name / direction / f'seed{seed}.json'
    if output.exists():
        raise FileExistsError(f'Evaluation exists: {output}')
    checked = {exp: load_run(exp, direction, seed, configs[exp], validated, source) for exp in ('E1', 'E2')}
    models = {'source_only': validated['model'], **{exp: checked[exp][0] for exp in checked}}
    paths = split_paths(validated, direction)
    runs = {}
    for name, model in models.items():
        threshold = validated['threshold'] if name == 'source_only' else checked[name][1]['threshold']
        y, s = collect_scores(model, ParquetBatchStream(paths['target_val'], 1024, False, seed, True))
        sy, ss = collect_scores(model, ParquetBatchStream(paths['source_val'], 1024, False, seed, True))
        metrics = compute_metrics(y, s, threshold)
        runs[name] = {'source_val': compute_metrics(sy, ss, threshold), 'target_development': metrics,
                      'source_score_analysis': score_diagnostics(sy, ss, threshold),
                      'target_score_analysis': score_diagnostics(y, s, threshold),
                      'selected_stage': 'source_only' if name == 'source_only' else checked[name][1]['selected_stage'],
                      'paired_delta_ap': metrics['pr_auc'] - baseline['target_development']['pr_auc']}
        print(f"{name}: target AP={metrics['pr_auc']:.6f}, recall={metrics['recall']:.6f}, FPR={metrics['fpr']:.6f}")
    report = {**validated['revision'], 'extension_id': VERSION, 'direction': direction, 'seed': seed,
              'data_role': 'target_validation_development_only', 'runs': runs,
              'e2_minus_e1': {key: runs['E2']['target_development'][key] - runs['E1']['target_development'][key] for key in METRICS}}
    if diagnostics:
        from evaluation.supcon_diagnostics import diagnose
        projections = {}
        for exp in checked:
            head = ContrastiveProjection()
            head.load_state_dict(checked[exp][1]['projection_state_dict'])
            projections[exp] = head
        report['negative_transfer_diagnostics'] = diagnose(models, projections, paths,
            checked['E1'][1]['source_train_counts'], {exp: load_config(configs[exp], exp) for exp in checked}, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f'Saved: {output}')


def run_aggregate(directions, seeds, configs, output_name):
    from evaluation.proposal_aggregate import seed_statistics
    output = revision_path('result_root', None) / 'extensions' / VERSION / output_name
    if output.exists():
        raise FileExistsError(f'Aggregate exists: {output}')
    rows, summary = [], []
    for direction in directions:
        for seed in seeds:
            validated, source, baseline = reference(direction, seed)
            for exp in ('E1', 'E2'):
                _, _, result, _ = load_run(exp, direction, seed, configs[exp], validated, source)
                rows.append({'direction': direction, 'seed': seed, 'experiment': exp,
                             'selected_stage': result['selected_stage'],
                             **{key: result['target_development'][key] for key in METRICS},
                             'delta_ap': result['target_development']['pr_auc'] - baseline['target_development']['pr_auc']})
        for exp in ('E1', 'E2'):
            group = [row for row in rows if row['direction'] == direction and row['experiment'] == exp]
            summary.append({'direction': direction, 'experiment': exp,
                            'metrics': {key: seed_statistics([row[key] for row in group]) for key in (*METRICS, 'delta_ap')}})
        differences = []
        for seed in seeds:
            pair = {row['experiment']: row['pr_auc'] for row in rows if row['direction'] == direction and row['seed'] == seed}
            differences.append(pair['E2'] - pair['E1'])
        summary.append({'direction': direction, 'experiment': 'E2_minus_E1', 'metrics': {'delta_ap': seed_statistics(differences)}})
    output.mkdir(parents=True)
    with (output / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump({'extension_id': VERSION, 'data_role': 'target_validation_development_only',
                   'per_run': rows, 'summary': summary,
                   'caveat': 'Paired seed variation on fixed datasets; a positive mean alone does not establish improvement.'}, stream, indent=2, allow_nan=False)
    with (output / 'per_run.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['train', 'evaluate', 'aggregate', 'smoke'])
    parser.add_argument('--experiment', choices=['E1', 'E2'])
    parser.add_argument('--direction', choices=[*DIRECTIONS, 'both'], default='both')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--config', type=Path, help='Requires --experiment; other experiment retains its default config')
    parser.add_argument('--output-name')
    parser.add_argument('--no-diagnostics', action='store_true', help='Evaluate scores without latent/gradient diagnostics')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique')
    if args.config and not args.experiment:
        parser.error('--config requires --experiment')
    configs = dict(CONFIGS)
    if args.config:
        configs[args.experiment] = args.config
    if args.stage == 'smoke':
        from experiments.supcon_smoke import smoke
        smoke(device_for(args.device))
        return
    directions = DIRECTIONS if args.direction == 'both' else (args.direction,)
    name = args.output_name or ('aggregate' if args.stage == 'aggregate' else 'evaluation')
    if not name or Path(name).name != name or name in ('.', '..', 'E1', 'E2'):
        parser.error('Output name must be a distinct directory name')
    if args.stage == 'aggregate':
        run_aggregate(directions, args.seeds, configs, name)
    else:
        for direction in directions:
            for seed in args.seeds:
                if args.stage == 'evaluate':
                    run_evaluate(direction, seed, configs, name, not args.no_diagnostics)
                else:
                    for exp in (args.experiment,) if args.experiment else ('E1', 'E2'):
                        run_train(exp, direction, seed, configs[exp], args.device)


if __name__ == '__main__':
    main()
