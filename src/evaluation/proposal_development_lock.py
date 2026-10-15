"""Freeze the complete development matrix before any final-test labels are read."""
from datetime import datetime, timezone
import json
from pathlib import Path

from training.data_revision import revision_path, verify_revision, file_sha256, ROOT
from experiments.proposal_protocol import SUITE_PATH, METHOD_CONFIGS, experiment_matrix


def freeze_development():
    from evaluation.proposal_aggregate import aggregate
    from evaluation.proposal_final_test import validate_development
    revision = verify_revision()
    suite_path = SUITE_PATH
    suite = json.loads(suite_path.read_text())
    root = revision_path('result_root', None)
    path = root / 'development_lock.json'
    if path.exists() or list((root / 'final_target_test').rglob('*.json')):
        raise FileExistsError('Development is already locked or final test has started; do not relock after reading target test')
    aggregate_path = root / 'aggregate/summary.json'
    stored = json.loads(aggregate_path.read_text())
    expected = aggregate(tuple(suite['directions']), tuple(suite['seeds']))
    if stored != expected:
        raise ValueError('Development aggregate is incomplete/stale; regenerate before locking')
    expected_matrix = set(experiment_matrix())
    actual_matrix = [(r['direction'], r['seed'], r['method']) for r in stored['per_run']]
    if len(actual_matrix) != len(expected_matrix) or set(actual_matrix) != expected_matrix:
        raise ValueError('Development aggregate must contain the complete unique experiment matrix')
    artifacts = {aggregate_path.resolve(), suite_path.resolve(),
                 (ROOT / 'configs/common_features_v2.json').resolve()}
    artifacts.update(path.resolve() for path in METHOD_CONFIGS.values())
    selection = []
    for direction, seed, method in sorted(expected_matrix):
        validated = validate_development(direction, method, seed)
        artifacts.update((validated['checkpoint_path'].resolve(), validated['result_path'].resolve()))
        selection.append({'direction': direction, 'seed': seed, 'method': method,
                          'threshold': float(validated['threshold']),
                          'source_val_ap_verified': validated['source_val_ap']})
    lock = {**revision, 'status': 'development_locked',
            'locked_at_utc': datetime.now(timezone.utc).isoformat(),
            'matrix': selection, 'expected_run_count': len(expected_matrix),
            'contract': 'Configs, features, checkpoints, source-val thresholds and development aggregate fixed before target-test labels; final outputs never tune this protocol',
            'artifacts': [{'path': str(p), 'sha256': file_sha256(p)} for p in sorted(artifacts)]}
    with path.open('x', encoding='utf-8') as stream:
        json.dump(lock, stream, indent=2, allow_nan=False)
        stream.write('\n')
    path.with_suffix('.sha256').write_text(file_sha256(path) + '  development_lock.json\n')
    print(f'Locked {len(selection)} development runs: {path}')
    return lock


def require_development_lock(revision, lock_path=None):
    path = Path(lock_path) if lock_path else revision_path('result_root', None) / 'development_lock.json'
    if not path.is_file():
        raise FileNotFoundError('Run --stage aggregate then --stage lock-development before final-test')
    checksum = path.with_suffix('.sha256')
    if not checksum.is_file() or checksum.read_text().split()[0] != file_sha256(path):
        raise ValueError('Development lock checksum changed or is missing')
    lock = json.loads(path.read_text())
    if lock.get('status') != 'development_locked' or any(lock.get(k) != v for k, v in revision.items()):
        raise ValueError('Development lock belongs to another revision')
    expected_matrix = set(experiment_matrix())
    matrix = [(r['direction'], r['seed'], r['method']) for r in lock.get('matrix', [])]
    if len(matrix) != len(expected_matrix) or set(matrix) != expected_matrix or lock.get('expected_run_count') != len(expected_matrix):
        raise ValueError('Development lock matrix is incomplete')
    for artifact in lock['artifacts']:
        if file_sha256(artifact['path']) != artifact['sha256']:
            raise ValueError(f"Locked development artifact changed: {artifact['path']}")
    return lock
