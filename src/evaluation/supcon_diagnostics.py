"""Bounded post-training diagnostics for BaselineMLP SupCon experiments."""
import numpy as np
import torch
from torch import nn

from training.adaptation import estimate_bandwidth_squared
from training.proposal_data import ParquetBatchStream
from training.proposal_mkmmd import multi_kernel_mmd_loss
from training.supcon_adaptation import frozen_bn_training


def fixed_mmd_comparison(latents, config, seed):
    from evaluation.proposal_domain_shift import fixed_kernel_mmd
    n = min(config['mmd_sample'], *(len(x) for pair in latents.values() for x in pair))
    if n < 2:
        raise ValueError('MMD diagnostics need at least two rows per domain')
    rng = np.random.default_rng(seed)
    pairs = [(rng.choice(len(latents['source_only'][0]), n, replace=False),
              rng.choice(len(latents['source_only'][1]), n, replace=False))
             for _ in range(config['mmd_repeats'])]
    i, j = pairs[0]
    sigma2 = estimate_bandwidth_squared(torch.from_numpy(latents['source_only'][0][i]),
                                       torch.from_numpy(latents['source_only'][1][j]))
    result = {}
    with torch.no_grad():
        for name, (source, target) in latents.items():
            values = [float(fixed_kernel_mmd(torch.from_numpy(source[i]), torch.from_numpy(target[j]), sigma2))
                      for i, j in pairs]
            result[name] = {'mmd2_mean': float(np.mean(values)),
                            'mmd2_subsample_std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                            'mmd2_runs': values}
    return {'kernel': 'single_RBF_diagnostic', 'bandwidth_squared': float(sigma2),
            'bandwidth_fit': 'source-only pooled subsample; fixed for source-only/E1/E2',
            'paired_indices': [{'source': i.tolist(), 'target': j.tolist()} for i, j in pairs],
            'std_interpretation': 'subsampling variation, not a confidence interval', 'runs': result}


def gradient_diagnostic(model, projection, paths, counts, config, seed):
    source = ParquetBatchStream(paths['source_train'], config['training']['batch_size'], False, seed, True, drop_last=True)
    target = ParquetBatchStream(paths['target_train'], config['training']['batch_size'], False, seed + 1000, False, drop_last=True)
    parameters = [*model.fc1.parameters(), *model.bn1.parameters(), *model.fc2.parameters(), *model.bn2.parameters()]
    weights = torch.tensor(counts, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.sum() / (2 * weights))
    values = {'ce_vs_mkmmd': [], 'ce_supcon_vs_mkmmd': []}
    frozen_bn_training(model)
    projection.eval()
    try:
        set_rng = torch.random.fork_rng(devices=[])
        with set_rng:
            torch.manual_seed(seed)
            for index, ((sx, sy), tx) in enumerate(zip(source, target)):
                if index >= config['diagnostics']['gradient_batches']:
                    break
                z, logits = model(sx)
                tz, _ = model(tx)
                ce = criterion(logits, sy)
                from training.supcon import supervised_contrastive_loss
                supcon, _ = supervised_contrastive_loss(projection(z), sy, config['temperature'])
                mmd = multi_kernel_mmd_loss(z, tz, config['kernel_scales'])[0]
                gradients = []
                for loss in (ce, ce + config['alpha_supcon'] * supcon, mmd):
                    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                    gradients.append(torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten()
                                                for p, g in zip(parameters, grads)]))
                for key, gradient in zip(values, gradients[:2]):
                    if gradient.norm() > 1e-12 and gradients[2].norm() > 1e-12:
                        values[key].append(float(torch.nn.functional.cosine_similarity(gradient, gradients[2], dim=0)))
    finally:
        model.eval()
    return {key: {'batches': len(series), 'mean': float(np.mean(series)) if series else None,
                   'negative_fraction': float(np.mean(np.array(series) < 0)) if series else None,
                   'status': 'ok' if series else 'no_valid_gradient_pairs'} for key, series in values.items()}


def diagnose(models, projections, paths, counts, configs, seed):
    from evaluation.proposal_negative_transfer import (
        sample_labeled, collect_representation, class_conditional_mmd,
        latent_domain_auc, bn_shift, label_prior_shift,
    )
    config = configs['E2']
    settings = config['diagnostics']
    sx, sy = sample_labeled(paths['source_val'], settings['sample_rows'], seed)
    tx, ty = sample_labeled(paths['target_val'], settings['sample_rows'], seed + 1)
    latents, reports = {}, {}
    for name, model in models.items():
        sz, source_means = collect_representation(model, sx)
        tz, target_means = collect_representation(model, tx)
        latents[name] = (sz, tz)
        report = {'domain_auc': latent_domain_auc(sz, tz, seed, settings['cv_folds']),
                  'bn_shift': bn_shift(model, source_means, target_means)}
        if all(int((labels == c).sum()) >= 2 for labels in (sy, ty) for c in (0, 1)):
            report['class_conditional'] = class_conditional_mmd(sz, sy, tz, ty, min(128, settings['mmd_sample']), seed)
        else:
            report['class_conditional'] = {'status': 'insufficient_sampled_class_counts'}
        if name in projections:
            report['gradient_cosine'] = gradient_diagnostic(model, projections[name], paths, counts, configs[name], seed)
        reports[name] = report
    fixed = fixed_mmd_comparison(latents, settings, seed)
    before = fixed['runs']['source_only']['mmd2_mean']
    for name in ('E1', 'E2'):
        after = fixed['runs'][name]['mmd2_mean']
        reports[name]['fixed_mmd_reduction_pct'] = 100 * (before - after) / before if before > 1e-12 else None
    return {'sample_rows': {'source': len(sx), 'target': len(tx)},
            'sampling': 'same ordered sampled flows for every checkpoint; reproducible from seed',
            'label_prior': label_prior_shift(sy, ty), 'fixed_bandwidth_mmd': fixed,
            'class_conditional_kernel': 'existing single-RBF diagnostic, separately estimated per checkpoint',
            'runs': reports, 'target_labels_role': 'post-training class diagnostics only'}
