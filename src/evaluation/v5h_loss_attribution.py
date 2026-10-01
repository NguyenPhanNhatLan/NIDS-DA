"""Local loss-gradient audit of bad Normal entrants and lost Attack at frozen V5h."""
import argparse
import json

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from evaluation.hda_v5b_calibration import data_snapshot
from evaluation.hda_v5f import verify_training_data
from evaluation.v5f_gradient_matrix import load_pools
from evaluation.v5h_tail_audit import score_pair, recall_at_fpr
from training import hda_v5f, hda_v5h
from training.hda_v4 import sample_pool
from training.hda_v5b import file_hash, ranking_loss
from training.mkmmd import MKMMDLoss
from training.weighted_mkmmd import WeightedMKMMDLoss
from training.v5e_performance import cached_pools
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader, make_teacher_loader

NAMES = ('hidden', 'normal', 'attack', 'rank', 'source')


def select_groups(old, new, labels):
    old, new, labels = np.asarray(old), np.asarray(new), np.asarray(labels)
    if (old.ndim != 1 or old.shape != new.shape or old.shape != labels.shape
            or not np.isfinite(old).all() or not np.isfinite(new).all()
            or set(np.unique(labels)) != {0, 1}):
        raise ValueError('Need aligned finite scores and both binary development classes')
    k = int(np.ceil(len(labels) * .10))
    f = np.zeros(len(labels), dtype=bool)
    h = f.copy()
    f[np.argsort(-old, kind='stable')[:k]] = True
    h[np.argsort(-new, kind='stable')[:k]] = True
    operating = {name: recall_at_fpr(scores, labels)['5%']
                 for name, scores in (('v5f', old), ('v5h', new))}
    def detected(scores, row):
        if row['predict_none']:
            return np.zeros(len(scores), dtype=bool)
        return scores >= row['development_diagnostic_threshold']
    groups = {
        'top10_bad_normal': np.flatnonzero(h & ~f & (labels == 0)),
        'top10_lost_attack': np.flatnonzero(f & ~h & (labels == 1)),
        'fpr5_lost_attack': np.flatnonzero(
            detected(old, operating['v5f']) & ~detected(new, operating['v5h']) & (labels == 1)),
    }
    expected = {'top10_bad_normal': 2484, 'top10_lost_attack': 1055, 'fpr5_lost_attack': 4857}
    metadata = {
        'top_k': k, 'development_fpr5_operating_points': operating,
        'expected_count_comparison': {
            name: {'observed': len(ids), 'expected': expected[name],
                   'expected_is_approximate': name == 'fpr5_lost_attack',
                   'difference': len(ids) - expected[name],
                   'matches_exactly': len(ids) == expected[name]}
            for name, ids in groups.items()},
        'attack_group_overlap': int(np.intersect1d(groups['top10_lost_attack'], groups['fpr5_lost_attack']).size),
        'fpr_policy': 'Separate raw-margin thresholds for V5f/V5h: maximum empirical recall at FPR <= 0.05; lowest FPR breaks recall ties. Whole score ties, >= comparison, no interpolation. Development diagnostic only, not saved deployment thresholds.',
    }
    return groups, metadata


def trainable_parameters(model):
    return tuple(model.adapter.parameters()) + tuple(model.classifier.parameters())


def gradient_vector(value, params, retain_graph=False):
    grads = torch.autograd.grad(value, params, retain_graph=retain_graph, allow_unused=True)
    vector = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten().double()
                        for p, g in zip(params, grads)])
    if not torch.isfinite(vector).all():
        raise ValueError('Nonfinite gradient')
    return vector


def loss_gradients(student, teacher, sx, sy, tx, source_pools, hard, middle, config, settings, class_weights):
    """All five weighted losses, eval BN; source labels only, no development labels."""
    kernel = MKMMDLoss(config['kernel_scales']).float()
    weighted = WeightedMKMMDLoss(config['kernel_scales']).float()
    sh, sz = student.source_representations(sx)
    th = student.adapter(tx)
    tz = student.shared_latent(th)
    logits = student.classifier(tz)
    with torch.no_grad():
        _, teacher_logits = teacher(tx)
    n = settings['class_batch_size']
    nx, nw, ax = hda_v5h.sample_conditional_inputs(hard, middle, n, config['conditional_anchor_mass'], 'cpu')
    z = student.shared_latent(student.adapter(torch.cat((nx, ax))))
    losses = {
        'hidden': kernel(sh, th)[0],
        'normal': weighted(sample_pool(source_pools[0], n, 'cpu'), z[:n], nw)[0],
        'attack': kernel(sample_pool(source_pools[1], n, 'cpu'), z[n:])[0],
        'rank': ranking_loss(teacher_logits[:,1]-teacher_logits[:,0], logits[:,1]-logits[:,0]),
        'source': F.cross_entropy(student.classifier(sz), sy, weight=class_weights),
    }
    params = trainable_parameters(student)
    return {name: gradient_vector(config['loss_weights'][name] * losses[name], params,
                                  retain_graph=i < len(NAMES)-1) for i, name in enumerate(NAMES)}


def directional_effects(student, features, mean_gradients, config):
    """First-order raw-margin changes for a hypothetical LR-scaled gradient step."""
    params = trainable_parameters(student)
    adapter_size = sum(p.numel() for p in student.adapter.parameters())
    rates = torch.cat([torch.full((p.numel(),), config['adapter_lr'], dtype=torch.float64)
                       for p in student.adapter.parameters()] +
                      [torch.full((p.numel(),), config['classifier_lr'], dtype=torch.float64)
                       for p in student.classifier.parameters()])
    rows = []
    for x in features:
        _, logits = student(x.unsqueeze(0))
        margin = logits[0,1]-logits[0,0]
        probe = gradient_vector(margin, params)
        effects = {}
        for name, loss_grad in mean_gradients.items():
            products = -rates * probe * loss_grad
            adapter = products[:adapter_size].sum().item()
            classifier = products[adapter_size:].sum().item()
            effects[name] = {'adapter': adapter, 'classifier': classifier, 'total': adapter + classifier}
        rows.append({'v5h_margin': margin.item(), 'effects': effects,
                     'total_predicted_margin_change': sum(v['total'] for v in effects.values())})
    return rows


def exact_middle_cache(config, checkpoint, provenance):
    """Use the frozen geometry weights that produced this checkpoint, not a refit."""
    found = []
    directory = resolve_path(config['geometry_cache_dir'])
    for manifest in sorted(directory.glob('*/manifest.json')):
        dep = json.loads(manifest.read_text())['dependencies']
        if (dep.get('kind') == 'v5h_v2_middle_geometry_weights'
                and dep.get('target_data') == checkpoint['training_data']['target']
                and dep.get('source_data') == checkpoint['training_data']['source']
                and dep.get('v2_checkpoint') == provenance['teacher_checkpoint_sha256']
                and dep.get('source_checkpoint') == provenance['source_checkpoint_sha256']
                and dep.get('producer_code', {}).get('src/training/hda_v5h.py') == checkpoint['code_sha256']['src/training/hda_v5h.py']):
            def no_build():
                raise ValueError('Required geometry cache missing')
            pool = cached_pools(directory, dep, no_build)
            if pool['metadata'] == checkpoint['conditional_alignment']['middle_metadata']:
                found.append((pool, str(manifest.parent)))
    if len(found) != 1:
        raise ValueError(f'Expected one exact training geometry cache, found {len(found)}')
    return found[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v5f-config', default='configs/hda_v5f.json')
    parser.add_argument('--v5h-config', default='configs/hda_v5h.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-seed', type=int, default=2026)
    parser.add_argument('--top-fraction', type=float, choices=(.10,), default=.10,
                        help='Fixed at 0.10 for the top10 cohorts')
    parser.add_argument('--batches', type=int, default=20)
    parser.add_argument('--max-per-group', type=int, default=64)
    parser.add_argument('--output', default='results/hda_v5h/diagnostics/loss_attribution_top10_fpr5_seed42.json')
    args = parser.parse_args()
    if not 0 < args.top_fraction < 1 or args.batches < 1 or args.max_per_group < 1:
        parser.error('Require 0 < top-fraction < 1 and positive batches/max-per-group')
    output = resolve_path(args.output)
    if output.exists():
        parser.error('Output exists; choose a fresh --output')
    models, checkpoints, deps, contexts = {}, {}, {}, {}
    for name, module, path in (('v5f', hda_v5f, args.v5f_config), ('v5h', hda_v5h, args.v5h_config)):
        context = module.load_context(path, args.seed)
        config, protocol, provenance, source, v2, teacher = context
        model, checkpoint = module.load_student(config, provenance, source, teacher)
        verify_training_data(protocol, checkpoint)
        ref = hda_v5f.load_v5d_reference(config, protocol, provenance)
        models[name] = model.cpu().eval().requires_grad_(False)
        checkpoints[name] = checkpoint
        contexts[name] = context
        deps[name] = {'checkpoint': str(module.checkpoint_path(config)),
                      'sha256': file_hash(module.checkpoint_path(config)), 'provenance': provenance,
                      'code_sha256': module.code_hashes(), 'development_files': ref['development_files']}
    config, protocol, provenance, source, v2, teacher = contexts['v5h']
    target = evaluation_target(protocol, 'development')
    if target.resolve() != evaluation_target(contexts['v5f'][1], 'development').resolve():
        raise ValueError('Development paths differ')
    snapshot = data_snapshot(target)
    if any(d['development_files'] != snapshot for d in deps.values()):
        raise ValueError('Development snapshot changed')
    bs = protocol['training']['batch_size']
    expected = sum(pq.read_metadata(p).num_rows for p in sorted(target.glob('*.parquet')))
    old, new, labels = score_pair(models['v5f'], models['v5h'], make_loader(target, provenance['target_dim'], bs), expected)
    groups, selection = select_groups(old, new, labels)
    print('Cohort counts and development FPR5 thresholds:', json.dumps(selection, indent=2), flush=True)
    rng = np.random.default_rng(args.sample_seed)
    indices = {name: np.sort(rng.choice(ids, min(len(ids), args.max_per_group), replace=False)) for name, ids in groups.items()}
    print('Selected groups:', {n: {'population':len(groups[n]), 'sampled':len(ids)} for n,ids in indices.items()}, flush=True)
    features = {name: [] for name in groups}
    offset = 0
    # Read feature-only development batches for selected row indices.
    for x in make_teacher_loader(target, provenance['target_dim'], bs):
        for name, ids in indices.items():
            local = ids[(ids >= offset) & (ids < offset + len(x))] - offset
            if len(local): features[name].append(x[torch.from_numpy(local)])
        offset += len(x)
    if offset != expected:
        raise ValueError('Development row count changed')
    student = models['v5h']
    before = {k:v.clone() for k,v in student.state_dict().items()}
    summaries, sample_rows = {}, {}
    if any(len(ids) for ids in indices.values()):
        for model in (source, v2, teacher): model.cpu().eval().requires_grad_(False)
        source_pools, hard, source_path, adaptation_path = load_pools(config, protocol, provenance, source, v2, torch.device('cpu'), bs)
        middle, middle_path = exact_middle_cache(config, checkpoints['v5h'], provenance)
        student.adapter.requires_grad_(True)
        student.classifier.requires_grad_(True)
        torch.manual_seed(args.sample_seed)
        si = iter(make_loader(source_path, provenance['source_dim'], bs))
        ti = iter(make_teacher_loader(adaptation_path, provenance['target_dim'], bs))
        averaged = {}
        for i in range(args.batches):
            try: sx, sy = next(si); tx = next(ti)
            except StopIteration: raise ValueError(f'Only {i} paired training batches available') from None
            grads = loss_gradients(student, teacher, sx, sy, tx, source_pools, hard, middle,
                                   config, protocol['training'], torch.tensor(checkpoints['v5h']['source_class_weights']))
            for name, g in grads.items(): averaged[name] = averaged.get(name, torch.zeros_like(g)) + g / args.batches
            print(f'Loss-gradient batch {i+1}/{args.batches}', flush=True)
        for group, ids in indices.items():
            if not len(ids):
                summaries[group] = {'population':len(groups[group]), 'sampled':0, 'per_loss':None}
                continue
            rows = directional_effects(student, torch.cat(features[group]), averaged, config)
            for row, idx in zip(rows, ids):
                row.update(development_row_index=int(idx), true_label=int(labels[idx]), v5f_margin=float(old[idx]))
            sample_rows[group] = rows
            per_loss = {}
            for name in NAMES:
                values = np.array([r['effects'][name]['total'] for r in rows])
                per_loss[name] = {'mean_predicted_margin_change':float(values.mean()),
                                  'median_predicted_margin_change':float(np.median(values)),
                                  'fraction_margin_up':float(np.mean(values>0)),
                                  'fraction_harmful_direction':float(np.mean(values>0 if group=='top10_bad_normal' else values<0)),
                                  'mean_adapter_effect':float(np.mean([r['effects'][name]['adapter'] for r in rows])),
                                  'mean_classifier_effect':float(np.mean([r['effects'][name]['classifier'] for r in rows]))}
            summaries[group] = {'population':len(groups[group]), 'sampled':len(rows), 'per_loss':per_loss}
    else:
        middle_path = None
        summaries = {name:{'population':0,'sampled':0,'per_loss':None} for name in groups}
    if any(not torch.equal(v,before[k]) for k,v in student.state_dict().items()):
        raise ValueError('Diagnostic changed model parameters or buffers')
    if data_snapshot(target) != snapshot:
        raise ValueError('Development changed during diagnostic')
    for name,module in (('v5f',hda_v5f),('v5h',hda_v5h)):
        if file_hash(resolve_path(deps[name]['checkpoint'])) != deps[name]['sha256'] or module.code_hashes()!=deps[name]['code_sha256']:
            raise ValueError('Checkpoint/code changed during diagnostic')
    result = {'phase':'post-hoc development', 'dependencies':deps, 'development_path':str(target),
              'development_snapshot':snapshot, 'top_fraction':args.top_fraction, 'sample_seed':args.sample_seed,
              'loss_gradient_batches':args.batches, 'geometry_cache':middle_path,
              'groups':summaries, 'samples':sample_rows, 'selection':selection,
              'definition':'At final V5h, average weighted loss gradients across paired source/adaptation batches; effect = -grad(probe raw margin) dot block_LR * mean_grad(weighted loss). Adapter and classifier included.',
              'group_policy':'Top10 uses ceil(N*0.10), descending score with stable original-row tie breaking. top10_bad_normal: Y=0 in V5h top10 only; top10_lost_attack: Y=1 in V5f top10 only. fpr5_lost_attack: Y=1 detected only by V5f at separate development FPR<=5% thresholds. Attack groups can overlap; expected counts are checks, not selection targets.',
              'limitations':'Eval BN, CPU, batch-estimated bandwidths, first distinct natural train batches and random conditional sampling. This is a local hypothetical LR-scaled SGD direction, not Adam, weight decay, rank/AP derivative, or historical causal attribution. Development labels select probes only, never training losses. Source CE can affect probes through classifier.',
              'model_unchanged':True}
    print(json.dumps(summaries,indent=2),flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    print('Saved:',output)


if __name__ == '__main__':
    main()
