"""V5f classifier margins on the unchanged V2 pseudo pools; no target labels."""
import argparse
import json
import math

import torch

from evaluation.calibration import calibrate_margin
from evaluation.hda_v5b_calibration import collect_margins, data_snapshot, payload_hash
from evaluation.hda_v5f import (
    calibration_path, dependencies, load_v5b_calibration, validate_affine, verify_training_data,
)
from evaluation.v5f_gradient_matrix import load_pools
from training.hda_v5f import checkpoint_path, load_context, load_student
from training.hda_v5b import file_hash
from training.thesis_protocol import resolve_path


def statistics(margins):
    m = margins.double()
    if m.numel() == 0 or not torch.isfinite(m).all():
        raise ValueError("Empty or nonfinite classifier margins")
    names = ('min', 'q05', 'q25', 'median', 'q75', 'q95', 'max')
    quantiles = torch.quantile(m, torch.tensor([0., .05, .25, .5, .75, .95, 1.], dtype=torch.float64))
    return {"count": m.numel(), **dict(zip(names, quantiles.tolist())),
            "mean": m.mean().item(), "std": m.std(unbiased=False).item(),
            "std_definition": "population (ddof=0)"}


def summarize(margins, parameters, threshold):
    m = margins.double()
    calibrated = calibrate_margin(m, parameters['a'], parameters['b'])
    raw_equivalent = (threshold - parameters['b']) / parameters['a']
    fraction = lambda mask: mask.double().mean().item()
    return {"raw_margin": statistics(m), "calibrated_margin": statistics(calibrated),
            "fraction_margin_gt_zero": fraction(m > 0),
            "fraction_margin_gt_raw_source_threshold": fraction(m > threshold),
            "fraction_calibrated_margin_gt_calibrated_threshold": fraction(calibrated > threshold),
            "fraction_margin_gt_raw_equivalent_calibrated_threshold": fraction(m > raw_equivalent),
            "comparison": "strict >"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/hda_v5f.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results/hda_v5f/diagnostics/pool_classifier_seed42.json')
    args = parser.parse_args()
    output = resolve_path(args.output)
    if output.exists():
        parser.error('Output exists; choose a fresh --output')
    config, protocol, provenance, source, v2, teacher = load_context(args.config, args.seed)
    student, checkpoint = load_student(config, provenance, source, teacher)
    verify_training_data(protocol, checkpoint)
    v5b = load_v5b_calibration(config, protocol, provenance)
    path = calibration_path(config)
    artifact = json.loads(path.read_text())
    frozen = artifact['frozen']
    deps = dependencies(config, provenance)
    if (payload_hash(frozen) != artifact['sha256']
            or frozen['dependencies'] != deps
            or frozen['v5b_calibration_payload_sha256'] != v5b['sha256']
            or frozen['source_validation_files'] != data_snapshot(resolve_path(config['source_validation']))
            or frozen['target_adaptation_files'] != data_snapshot(resolve_path(protocol['target_data']['adaptation_train']))):
        raise ValueError('Frozen calibration, dependencies or fitting data changed')
    params = frozen['parameters']['v5f']
    threshold = frozen['thresholds']['v5f']
    validate_affine(params)
    if not math.isfinite(threshold):
        raise ValueError('Nonfinite source threshold')
    for model in (student, source, v2):
        model.cpu().eval().requires_grad_(False)
    batch_size = protocol['training']['batch_size']
    _, pools, _, _ = load_pools(config, protocol, provenance, source, v2, torch.device('cpu'), batch_size)
    result = {
        'checkpoint': str(checkpoint_path(config)), 'dependencies': deps,
        'calibration': str(path), 'calibration_payload_sha256': artifact['sha256'],
        'training_seed': config['training_seed'], 'target_labels_used': False,
        'pool_policy': protocol['pseudo_labels'],
        'margin_model': 'adapted V5f private classifier: logit_attack - logit_normal',
        'raw_source_threshold': threshold, 'calibrated_threshold': threshold,
        'raw_equivalent_calibrated_threshold': (threshold - params['b']) / params['a'],
        'affine': {'a': params['a'], 'b': params['b']},
        'note': 'Same fixed V2 pools used for affine fitting; descriptive in-sample diagnostics, not independent validation.',
        'pools': {},
    }
    print(json.dumps({k: result[k] for k in ('raw_source_threshold', 'calibrated_threshold',
                                           'raw_equivalent_calibrated_threshold', 'affine')}, indent=2), flush=True)
    for label, name in ((0, 'pseudo_normal'), (1, 'pseudo_attack')):
        margins, _ = collect_margins(student, pools[label].split(batch_size))
        result['pools'][name] = summarize(margins, params, threshold)
        print(name, json.dumps(result['pools'][name], indent=2), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('Saved:', output)


if __name__ == '__main__':
    main()
