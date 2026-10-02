"""Score only a frozen semantic audit against verified unpruned training splits."""
import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from features.feature_pair_scoring import run
from features.unpruned_audit_data import ROOT, sha256_file


def freeze_semantics(config_path):
    """Freeze curated decisions and schema availability, without reading values."""
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    config = json.loads(config_path.read_text())
    destination = root / config['semantic_freeze']
    if destination.exists():
        raise FileExistsError('This semantic audit is already frozen')
    manifest_path = root / config['dataset_manifest']
    if not (manifest_path.parent / '_SUCCESS').exists():
        raise ValueError('Dataset is incomplete')
    candidate_path = root / config['candidates']
    candidates = json.loads(candidate_path.read_text())['pairs']
    baseline = {p['canonical_name']: p for p in json.loads((root / 'configs/feature_pair_candidates.json').read_text())['pairs']}
    schemas = {d: pq.ParquetDataset(root / config['training_paths'][d]).schema.names for d in ('unsw','cicids')}
    rows = []
    for pair in candidates:
        if not baseline[pair['canonical_name']]['semantic_valid'] and pair['semantic_valid']:
            raise ValueError('This audit must not promote previously rejected pairs')
        checks = pair['semantic_checks']
        if pair['semantic_valid'] and not all(checks[k] is True for k in ('meaning_match','direction_match','unit_match','computation_match','aggregation_match')):
            raise ValueError('Accepted pair lacks all five semantic checks')
        rows.append(dict(canonical_name=pair['canonical_name'], unsw_feature=pair['unsw'],
            cicids_feature=pair['cicids'], available_unsw=pair['unsw'] in schemas['unsw'],
            available_cicids=pair['cicids'] in schemas['cicids'], semantic_valid=pair['semantic_valid'],
            **checks, evidence=' | '.join(pair['evidence']), decision_reason=pair['reason']))
    table_path = root / config['semantic_table']
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(table_path, index=False)
    # Binding configuration as well as decisions prevents scoring another pool.
    files = [config['candidates'], config['semantic_table'], config['dataset_manifest'],
             config['evidence_manifest'], str(config_path.relative_to(root))]
    freeze = dict(status='frozen_before_scoring', schemas=schemas,
                  files={name: sha256_file(root / name) for name in files})
    destination.write_text(json.dumps(freeze, indent=2)+'\n')
    return freeze


def verify_freeze(config_path):
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    config = json.loads(config_path.read_text())
    freeze = json.loads((root / config['semantic_freeze']).read_text())
    if freeze['status'] != 'frozen_before_scoring':
        raise ValueError('Semantic audit must be frozen before scoring')
    for filename, digest in freeze['files'].items():
        if sha256_file(root / filename) != digest:
            raise ValueError(f'Frozen audit file changed: {filename}')
    if config['candidates'] not in freeze['files']:
        raise ValueError('Candidate decisions are not frozen')
    manifest_path = root / config['dataset_manifest']
    if not (manifest_path.parent / '_SUCCESS').exists():
        raise ValueError('Unpruned dataset build is incomplete')
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] not in ('complete', 'complete_with_quarantine'):
        raise ValueError('Dataset provenance is unverified')
    for domain in ('unsw','cicids'):
        expected = manifest_path.parent / f'{domain}_clean_unpruned/splits/{domain}_train'
        if (root / config['training_paths'][domain]).resolve() != expected.resolve():
            raise ValueError('Training paths do not belong to the verified dataset')
        if not manifest['datasets'][domain]['provenance']['verified_multisets_equal']:
            raise ValueError('Historical feature-key multisets were not verified')
    return root, config, freeze


def main(config_path=None):
    if config_path is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--config', default=str(ROOT / 'configs/feature_selection_unpruned_v1.json'))
        parser.add_argument('--freeze', action='store_true', help='Freeze semantic audit only; do not score')
        args = parser.parse_args()
        config_path = args.config
        if args.freeze:
            freeze_semantics(config_path)
            print('Semantic audit frozen; no statistical scoring performed.')
            return
    root, config, freeze = verify_freeze(config_path)
    # No speculative Top-K outputs: the expanded schema may still yield <10.
    if config['top_k']:
        raise ValueError('Audit mode ranks the eligible pool only; top_k must be empty')
    scores, selected = run(config_path)
    audit = pd.read_csv(root / config['semantic_table'])
    summary = dict(A_candidates_before_semantics=len(audit),
        B_passing_semantics=int(audit.semantic_valid.sum()),
        C_available_in_both=int((audit.available_unsw & audit.available_cicids).sum()),
        C_semantic_and_available=int((audit.semantic_valid & audit.available_unsw & audit.available_cicids).sum()),
        D_surviving_quality=len(selected), E_final_eligible_pool=[r['canonical_name'] for r in selected],
        E_final_eligible_count=len(selected), top_k_generated=[],
        semantic_freeze_sha256=sha256_file(root / config['semantic_freeze']))
    out = root / config['output_dir']
    (out / 'audit_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    pd.DataFrame(selected).to_csv(out / 'eligible_pool.csv', index=False)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    main()
