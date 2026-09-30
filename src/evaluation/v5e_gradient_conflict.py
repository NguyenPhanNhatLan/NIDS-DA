"""Adapter gradient conflict on saved fixed diagnostic batches; no training."""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from training.hda_v5b import file_hash, ranking_loss
from training.hda_v5e import checkpoint_path, load_context, load_student
from training.mkmmd import mk_mmd_loss


NAMES = ('hidden', 'normal', 'attack', 'rank')


def gradient_summary(gradients, eps=1e-12):
    """Cosine is undefined for zero/near-zero vectors; never report it as zero."""
    vectors = {name: gradients[name].detach().cpu().double().flatten() for name in NAMES}
    if len({v.numel() for v in vectors.values()}) != 1:
        raise ValueError('Gradient vectors must have matching dimensions')
    if any(not torch.isfinite(v).all() for v in vectors.values()):
        raise ValueError('Nonfinite adapter gradient')
    other = vectors['hidden'] + vectors['normal'] + vectors['rank']
    norms = {name: torch.linalg.vector_norm(value).item() for name, value in vectors.items()}
    other_norm = torch.linalg.vector_norm(other).item()
    valid = norms['attack'] > eps and other_norm > eps
    cosine = F.cosine_similarity(vectors['attack'], other, dim=0, eps=eps).item() if valid else None
    return {
        'weighted_gradient_norms': norms,
        'other_gradient_norm': other_norm,
        'cosine_attack_vs_hidden_normal_rank': cosine,
        'dot_attack_vs_others': torch.dot(vectors['attack'], other).item(),
        'attack_to_other_norm_ratio': norms['attack'] / other_norm if other_norm > eps else None,
        'cosine_status': 'defined' if valid else 'undefined: zero or near-zero gradient norm',
        'zero_norm_epsilon': eps,
    }


def measure_conflict(model, teacher, batch, scales, weights, bandwidth_mode='fixed'):
    if bandwidth_mode not in ('fixed', 'batch'):
        raise ValueError('bandwidth_mode must be fixed or batch')
    if any(name not in weights or weights[name] < 0 for name in NAMES):
        raise ValueError('Missing or negative loss weights')
    # Copies make the diagnostic independent of caller modes, buffers and .grad.
    student = copy.deepcopy(model).eval().requires_grad_(False)
    frozen_teacher = copy.deepcopy(teacher).eval().requires_grad_(False)
    student.adapter.requires_grad_(True)
    device = next(student.parameters()).device
    adapter = tuple(student.adapter.parameters())
    source_h = batch['source_hidden'].detach().to(device)
    target = batch['target'].detach().to(device)
    with torch.enable_grad():
        target_h = student.adapter(target)
        latent = student.shared_latent(target_h)
        logits = student.classifier(latent)
        target_normal = student.shared_latent(student.adapter(batch['target_normal'].detach().to(device)))
        target_attack = student.shared_latent(student.adapter(batch['target_attack'].detach().to(device)))
        losses, bandwidths = {}, {}
        pairs = {'hidden': (source_h, target_h),
                 'normal': (batch['source_normal'].detach().to(device), target_normal),
                 'attack': (batch['source_attack'].detach().to(device), target_attack)}
        for name, (s, t) in pairs.items():
            base = batch['bandwidth_squared'][name] if bandwidth_mode == 'fixed' else None
            losses[name], sigma = mk_mmd_loss(s, t, scales=scales, bandwidth_squared=base)
            bandwidths[name] = sigma.square().item()
        with torch.no_grad():
            _, teacher_logits = frozen_teacher(target)
        losses['rank'] = ranking_loss(teacher_logits[:, 1] - teacher_logits[:, 0],
                                      logits[:, 1] - logits[:, 0])
        weighted = {name: weights[name] * losses[name] for name in NAMES}
        vectors = {}
        for index, name in enumerate(NAMES):
            grads = torch.autograd.grad(weighted[name], adapter,
                                        retain_graph=index < len(NAMES) - 1, allow_unused=True)
            vectors[name] = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten().cpu()
                                       for p, g in zip(adapter, grads)])
    return {
        **gradient_summary(vectors),
        'raw_losses': {name: losses[name].item() for name in NAMES},
        'weighted_losses': {name: weighted[name].item() for name in NAMES},
        'weights': {name: weights[name] for name in NAMES},
        'bandwidth_mode': bandwidth_mode, 'bandwidth_squared': bandwidths,
        'kernel_scales': list(scales), 'bn_mode': 'eval',
        'adapter_parameter_count': sum(p.numel() for p in adapter),
        'interpretation': 'Negative cosine means opposing local directions; small Attack norm means weak local influence. Neither alone predicts AP/F1.',
        'limitations': 'One fixed batch, eval BN, raw gradients before Adam preconditioning/weight decay; not the full training trajectory.',
        'source_ce_adapter_gradient': 'Zero: source CE has no computational path to the target adapter.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/hda_v5e.json')
    parser.add_argument('--training-seed', type=int, choices=(42, 43, 44), default=42)
    parser.add_argument('--bandwidth', choices=('fixed', 'batch'), default='fixed')
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--output', help='Optional new JSON file; never overwrite')
    args = parser.parse_args()
    output = Path(args.output) if args.output else None
    if output is not None and output.exists():
        parser.error(f'Output already exists: {output}')
    config, _, provenance, source, _, teacher = load_context(args.config, args.training_seed)
    model, checkpoint = load_student(config, provenance, source, teacher)
    if 'fixed_diagnostic_batches' not in checkpoint:
        parser.error('Checkpoint has no saved fixed diagnostic batches')
    device = torch.device(args.device)
    result = measure_conflict(model.to(device), teacher.to(device), checkpoint['fixed_diagnostic_batches'],
                              checkpoint['kernel_scales'], checkpoint['loss_weights'], args.bandwidth)
    result.update(checkpoint=str(checkpoint_path(config)), checkpoint_sha256=file_hash(checkpoint_path(config)),
                  diagnostic_code_sha256=file_hash(Path(__file__)), training_seed=config['training_seed'],
                  teacher_seed=config['teacher_seed'], device=str(device))
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            stream.write(rendered + '\n')
    print(rendered)


if __name__ == '__main__':
    main()
