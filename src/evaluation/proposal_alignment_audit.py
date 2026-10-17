"""Five post-hoc validation checks for a class-aware development variant."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.proposal_aggregate import seed_statistics
from evaluation.proposal_latent_knn_audit import agreements
from evaluation.proposal_negative_transfer import load_models, sample_labeled, FEATURE_ROOT
from experiments.proposal_protocol import DIRECTIONS, SEEDS
from training.adaptation import estimate_bandwidth_squared, rbf_kernel
from training.data_revision import verify_revision
from training.proposal_mmd import load_config
from training.proposal_class_aware import confidence_setting


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def predict(model, x):
    latent, probabilities = [], []
    for start in range(0, len(x), 1024):
        z, logits = model(torch.from_numpy(x[start:start+1024]))
        latent.append(z.numpy())
        probabilities.append(logits.softmax(1).numpy())
    return np.concatenate(latent), np.concatenate(probabilities)


def pseudo_quality(probabilities, labels, threshold):
    predicted = probabilities.argmax(1)
    if isinstance(threshold, dict):
        threshold = np.array([threshold["0"], threshold["1"]])[predicted]
    accepted = probabilities.max(1) >= threshold
    return prediction_quality(predicted, accepted, labels)


def prediction_quality(predicted, accepted, labels):
    attack = accepted & (predicted == 1)
    tp = int(np.sum(attack & (labels == 1)))
    return {
        'sample_rows': len(labels), 'true_attacks': int(np.sum(labels == 1)),
        'accepted': int(accepted.sum()), 'coverage': float(accepted.mean()),
        'accepted_accuracy': float(np.mean(predicted[accepted] == labels[accepted])) if accepted.any() else None,
        'accepted_attack': int(attack.sum()), 'correct_accepted_attack': tp,
        'attack_precision': tp / int(attack.sum()) if attack.any() else None,
        'attack_recall': tp / int(np.sum(labels == 1)) if np.any(labels == 1) else None,
    }


def balanced_indices(labels, per_class, rng):
    n = min(per_class, *(int(np.sum(labels == c)) for c in (0, 1)))
    if n < 2:
        raise ValueError('Validation sample needs both classes; increase --sample-rows')
    return np.concatenate([rng.choice(np.flatnonzero(labels == c), n, replace=False) for c in (0, 1)])


@torch.inference_mode()
def paired_mmd(before_s, before_t, after_s, after_t, sample_size, repeats, seed):
    rng = np.random.default_rng(seed)
    n = min(sample_size, len(before_s), len(before_t))
    before, after = [], []
    for _ in range(repeats):
        si = rng.choice(len(before_s), n, replace=False)
        ti = rng.choice(len(before_t), n, replace=False)
        s, t, a, b = [torch.from_numpy(x) for x in
                      (before_s[si], before_t[ti], after_s[si], after_t[ti])]
        # One baseline bandwidth per paired subsample, shared before and after.
        bandwidth = estimate_bandwidth_squared(s, t)
        for left, right, values in ((s,t,before), (a,b,after)):
            value = (rbf_kernel(left,left,bandwidth).mean()
                     + rbf_kernel(right,right,bandwidth).mean()
                     - 2*rbf_kernel(left,right,bandwidth).mean())
            values.append(float(value))
    return {'before': float(np.mean(before)), 'after': float(np.mean(after)),
            'paired_delta': float(np.mean(np.array(after)-before)),
            'repeats': repeats, 'sample_size_per_domain': n}


def run(args):
    config = load_config(args.config)
    revision = verify_revision()
    # Complete matrix and provenance preflight happens before creating output.
    for seed in args.seeds:
        load_models(args.direction, seed, args.config)
    source, target = args.direction.split('_to_')
    base = FEATURE_ROOT / args.direction
    xs, ys = sample_labeled(base / f'{source}_val', args.sample_rows, args.sampling_seed)
    xt, yt = sample_labeled(base / f'{target}_val', args.sample_rows, args.sampling_seed+1)
    rng = np.random.default_rng(args.sampling_seed)
    si = balanced_indices(ys, args.per_class, rng)
    ti = balanced_indices(yt, args.per_class, rng)
    if args.k >= min(len(si), len(ti)):
        raise ValueError('k must be smaller than each balanced domain sample')
    metrics, history, pseudo, knn, mmd, provenance = [], [], [], [], [], []
    for seed in args.seeds:
        baseline, adapted, br, ar, paths = load_models(args.direction, seed, args.config)
        checkpoint = torch.load(paths['adapted_checkpoint'], map_location='cpu', weights_only=True)
        for artifact in (br, ar, checkpoint):
            if any(artifact.get(k) != v for k,v in revision.items()):
                raise ValueError('Artifact frozen revision mismatch')
        if checkpoint['config_sha256'] != ar['config_sha256']:
            raise ValueError('Checkpoint config mismatch')
        provenance.append({'seed':seed, **paths})
        metrics.append({'seed':seed, **{
            f'{m}_{phase}': float(values[m])
            for m in ('pr_auc','roc_auc')
            for phase,values in (('before',br['target_development']), ('after',ar['cross_domain']))},
            **{f'{m}_delta': float(ar['cross_domain'][m]-br['target_development'][m]) for m in ('pr_auc','roc_auc')}})
        for row in checkpoint['history']:
            if not row.get('class_separation'):
                continue
            sep = row['class_separation']
            align = row['class_aware_alignment']
            batches = align['aligned_batches']+align['skipped_batches']
            weighted = sep['weight']*sep['loss']
            history.append({'seed':seed, 'epoch':row['epoch'], 'selected_epoch':checkpoint['best_epoch'],
                            'separation_loss':sep['loss'], 'weighted_separation_loss':weighted,
                            'total_loss':row['loss'], 'weighted_fraction_of_total':weighted/row['loss'] if row['loss'] else None,
                            'eligible_batches':sep['eligible_batches'], 'total_batches':batches,
                            'eligible_rate':sep['eligible_batches']/batches,
                            'mmd2_training':row['mmd2'], 'source_ce':row['source_ce'],
                            'weighted_source_ce':row.get('loss_contributions',{}).get('weighted_source_ce'),
                            'weighted_class_mmd':row.get('loss_contributions',{}).get('weighted_class_mmd'),
                            'positive_separation_batches':sep.get('positive_loss_batches'),
                            'gradient_sampled_batches':row.get('gradient_diagnostics',{}).get('sampled_batches'),
                            **{f'gradient_l2_{name}':row.get('gradient_diagnostics',{}).get(
                                'mean_weighted_parameter_gradient_l2',{}).get(name)
                               for name in ('source_ce','class_mmd','class_separation')}})
        representations = {}
        for name,model in (('before',baseline), ('after',adapted)):
            zs,_ = predict(model,xs)
            zt,probabilities = predict(model,xt)
            representations[name] = (zs,zt)
            thresholds = list(args.thresholds)
            configured_threshold = confidence_setting(config)
            if configured_threshold not in thresholds:
                thresholds.append(configured_threshold)
            for threshold in thresholds:
                pseudo.append({'seed':seed,'model':name,'selection':'confidence_only','threshold':json.dumps(threshold) if isinstance(threshold,dict) else threshold,
                               **pseudo_quality(probabilities,yt,threshold)})
            for task,zq,zr,yq,yr,same in (
                ('target_within',zt[ti],zt[ti],yt[ti],yt[ti],True),
                ('source_within',zs[si],zs[si],ys[si],ys[si],True),
                ('target_to_source',zt[ti],zs[si],yt[ti],ys[si],False)):
                values = agreements(zq,zr,yq,yr,[args.k],'euclidean',same)[args.k]
                for label in (0,1):
                    selected = values[yq==label]
                    knn.append({'seed':seed,'model':name,'task':task,'class':label,'k':args.k,
                                'n_query':len(selected),'agreement':float(selected.mean()),'mixing':float(1-selected.mean())})
        bs,bt = representations['before']; a_s,at = representations['after']
        for label in (None,0,1):
            mask_s = np.ones(len(ys),bool) if label is None else ys==label
            mask_t = np.ones(len(yt),bool) if label is None else yt==label
            if min(int(mask_s.sum()),int(mask_t.sum())) < 2:
                raise ValueError('Too few sampled rows for class MMD')
            mmd.append({'seed':seed,'class':'marginal' if label is None else label,
                        **paired_mmd(bs[mask_s],bt[mask_t],a_s[mask_s],at[mask_t],
                                     args.mmd_sample,args.mmd_repeats,args.sampling_seed)})
        print(f'Audited seed {seed}: AP delta={metrics[-1]["pr_auc_delta"]:+.6f}',flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    for name,rows in (('per_seed_metrics',metrics),('training_contribution',history),
                      ('pseudo_label_quality',pseudo),('latent_knn',knn),('paired_mmd',mmd)):
        if rows:
            write_csv(args.output / f'{name}.csv',rows)
    summary = {'data_role':'post_hoc_validation_only', 'target_labels_used_for_diagnostics':True,
               'target_test_read':False, 'seeds':args.seeds,
               'metrics':{m:{'improved_seeds':sum(r[f'{m}_delta']>0 for r in metrics),
                             'paired_delta':seed_statistics([r[f'{m}_delta'] for r in metrics])}
                          for m in ('pr_auc','roc_auc')},
               'separation_epochs_with_positive_loss':sum(r['separation_loss']>0 for r in history),
               'separation_epochs_audited':len(history)}
    manifest = {**revision,'config':str(args.config),'sampling_seed':args.sampling_seed,
                'source_sample_rows':len(ys),'target_sample_rows':len(yt),
                'parameters':{k:v for k,v in vars(args).items() if k not in ('config','output')},
                'target_sample_attack_prevalence':float(np.mean(yt==1)),
                'same_samples_across_seeds_and_models':True,'provenance':provenance,
                'limitations':[
                    'Pseudo-label quality uses a natural-prevalence validation subsample, not training labels or online training predictions.',
                    'kNN uses class-balanced subsets and Euclidean original latent space; mixing=1-agreement. Self rows excluded; other duplicates retained.',
                    'MMD uses paired samples and the same source-only bandwidth before/after; marginal and true-class conditional values are post-hoc diagnostics.',
                    'Latent scaling changes can affect MMD and kNN; decreasing MMD alone does not establish class preservation.',
                    'Positive weighted separation loss shows objective contribution, not causal benefit. Use ablation to assess benefit.',
                    'Seed confidence intervals describe run variation conditional on fixed validation samples.']}
    for name,value in (('summary',summary),('manifest',manifest)):
        (args.output / f'{name}.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2)); print(f'Saved: {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--direction',choices=DIRECTIONS,default='unsw_to_cicids')
    parser.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    parser.add_argument('--sample-rows',type=int,default=50000)
    parser.add_argument('--sampling-seed',type=int,default=42)
    parser.add_argument('--per-class',type=int,default=500)
    parser.add_argument('--k',type=int,default=15)
    parser.add_argument('--thresholds',nargs='+',type=float,default=[.8,.9,.95])
    parser.add_argument('--mmd-sample',type=int,default=256)
    parser.add_argument('--mmd-repeats',type=int,default=10)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a fresh directory')
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or min(args.sample_rows,args.per_class,args.mmd_sample)<2
            or args.k<1 or args.mmd_repeats<1
            or any(not 0<=t<=1 for t in args.thresholds)):
        parser.error('Invalid seeds, sample sizes, k, repeats or thresholds')
    torch.set_num_threads(4)
    run(args)


if __name__ == '__main__':
    main()
