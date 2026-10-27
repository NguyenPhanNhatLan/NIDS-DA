import argparse
import json
from pathlib import Path

from project_utils.result_files import result_file

from project_utils.statistics import seed_statistics
from experiments.proposal_protocol import DIRECTIONS, SEEDS
from training.data_revision import verify_revision
from training.proposal_mmd import load_config, output_paths, sha256, SOURCE_ONLY_ROOT

METRICS = ('pr_auc', 'roc_auc', 'macro_f1', 'recall', 'fpr')


def summarize(config, direction, seeds):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Seeds must be nonempty and unique')
    method = load_config(config)['method']
    revision = verify_revision()
    runs = []
    for seed in seeds:
        checkpoint, path = output_paths(direction, seed, config)
        result = json.loads(path.read_text())
        baseline = json.loads((SOURCE_ONLY_ROOT / direction / f'seed{seed}.json').read_text())
        if (result['direction'] != direction or result['seed'] != seed
                or result['method'] != method or result['config_sha256'] != sha256(config)
                or result['checkpoint'] != str(checkpoint) or not checkpoint.is_file()
                or any(result.get(k) != v for k, v in revision.items())):
            raise ValueError(f'Stale or mismatched run: {path}')
        if any(result[k] != baseline[k] for k in ('prepared_split_sha256', 'preprocessor_sha256')):
            raise ValueError(f'Baseline provenance mismatch: {path}')
        runs.append({'seed': seed, 'metrics': result['cross_domain'],
                     'baseline': baseline['target_development']})
    return {
        'direction': direction, 'config': str(config), 'seeds': seeds,
        'data_role': 'development_only',
        'metrics': {m: {
            'source_only': seed_statistics([r['baseline'][m] for r in runs]),
            'adapted': seed_statistics([r['metrics'][m] for r in runs]),
            'paired_delta': seed_statistics([r['metrics'][m] - r['baseline'][m] for r in runs]),
        } for m in METRICS},
    }


def main():
    parser = argparse.ArgumentParser(description='Summarize a complete paired-seed development variant without reading test data.')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--direction', choices=DIRECTIONS, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output:
        args.output = result_file(args.output)
    result = summarize(args.config, args.direction, args.seeds)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print('Metric       Source mean±std       Adapted mean±std      Paired delta')
    for name, stats in result['metrics'].items():
        source, adapted, delta = (stats[k] for k in ('source_only', 'adapted', 'paired_delta'))
        print(f"{name:10s} {source['mean']:.6f}±{source['std']:.6f}    "
              f"{adapted['mean']:.6f}±{adapted['std']:.6f}    {delta['mean']:+.6f}")


if __name__ == '__main__':
    main()
