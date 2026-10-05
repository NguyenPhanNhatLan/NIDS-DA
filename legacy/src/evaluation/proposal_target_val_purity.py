"""Audit saved class-aware pseudo-label rules without retraining or refitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from features.common_features import COMMON_FEATURES
from models.baseline import BaselineMLP
from training.proposal_class_aware import (
    FEATURE_ROOT, MODEL_ROOT, ROOT, audit_target_val_pseudo_labels,
    direction_domains, get_device, sha256,
)
from training.proposal_data import split_sha256


def audit_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    direction = checkpoint['direction']
    _, target = direction_domains(direction)
    teacher_path = Path(checkpoint['source_checkpoint'])
    if not teacher_path.is_file():
        teacher_path = ROOT / 'models/proposal_v2/source_only_target_val' / direction / f"seed{checkpoint['seed']}.pt"
    if sha256(teacher_path) != checkpoint['source_checkpoint_sha256']:
        raise ValueError(f'Source teacher checkpoint changed: {teacher_path}')
    base = FEATURE_ROOT / direction
    source, target = direction_domains(direction)
    for role, domain in (('source', source), ('target', target)):
        for split in ('train', 'val'):
            key = f'{role}_{split}'
            if split_sha256(base / f'{domain}_{split}') != checkpoint['prepared_split_sha256'][key]:
                raise ValueError(f'Prepared split changed: {key} ({direction})')
    if checkpoint['features'] != list(COMMON_FEATURES):
        raise ValueError('Checkpoint features do not match current features.')
    teacher_checkpoint = torch.load(teacher_path, map_location='cpu', weights_only=True)
    teacher = BaselineMLP(input_dim=checkpoint['input_dim']).to(device)
    teacher.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    audit = audit_target_val_pseudo_labels(
        teacher, base / f'{target}_val', device,
        checkpoint['q_normal'], checkpoint['q_attack'],
    )
    return {'direction': direction, 'seed': checkpoint['seed'],
            'class_aware_checkpoint': str(path),
            'source_checkpoint': str(teacher_path), **audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction', choices=['both', 'cicids_to_unsw', 'unsw_to_cicids'], default='both')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tag', default='q02_q95_lambda0001')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    directions = ['cicids_to_unsw', 'unsw_to_cicids'] if args.direction == 'both' else [args.direction]
    device = get_device()
    results = [audit_checkpoint(MODEL_ROOT / args.tag / direction / f'seed{args.seed}.pt', device)
               for direction in directions]
    print('| Direction | Group | Count | Purity |')
    print('| --- | --- | ---: | ---: |')
    names = {'cicids_to_unsw': 'CICIDS→UNSW', 'unsw_to_cicids': 'UNSW→CICIDS'}
    for result in results:
        for group in ('normal', 'attack'):
            purity = result[f'pseudo_{group}_purity']
            display = f'{purity:.2%}' if purity is not None else 'N/A'
            print(f"| {names[result['direction']]} | pseudo-{group.title()} | {result[f'pseudo_{group}_count']} | {display} |")
    for result in results:
        print(f"Coverage {names[result['direction']]}: {result['coverage']:.2%} ({result['accepted_count']}/{result['target_rows']})")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(results, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(f'Saved audit: {args.output}')


if __name__ == '__main__':
    main()
