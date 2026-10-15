"""Post-hoc validation latent UMAP export; labels never enter model or UMAP fitting."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import umap

from evaluation.proposal_negative_transfer import (
    FEATURE_ROOT, ROOT, collect_representation, domains, load_models, sample_labeled,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction', choices=['unsw_to_cicids', 'cicids_to_unsw'], default='unsw_to_cicids')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sampling-seed', type=int, default=42)
    parser.add_argument('--sample-per-domain', type=int, default=3000)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/analysis/tableau_class_diagnostics/latent_umap_seed42')
    args = parser.parse_args()
    if args.sample_per_domain < 30:
        parser.error('sample-per-domain must be at least 30')
    if args.output.exists():
        raise FileExistsError('Use a new output folder')
    source, target = domains(args.direction)
    xs, ys = sample_labeled(FEATURE_ROOT / args.direction / f'{source}_val', args.sample_per_domain, args.sampling_seed)
    xt, yt = sample_labeled(FEATURE_ROOT / args.direction / f'{target}_val', args.sample_per_domain, args.sampling_seed + 1)
    labels = np.concatenate([ys, yt])
    x = np.concatenate([xs, xt])
    if set(np.unique(ys)) != {0, 1} or set(np.unique(yt)) != {0, 1}:
        raise ValueError('Both classes must occur in each validation sample')
    models = {}
    provenance = {}
    for name, config in [('MMD', 'proposal_mmd_v2'), ('MK-MMD', 'proposal_mkmmd_v2'),
                         ('Class-aware MMD', 'proposal_class_aware_v2')]:
        baseline, adapted, _, _, paths = load_models(args.direction, args.seed, ROOT / 'configs' / f'{config}.json')
        models.setdefault('Source-only', baseline)
        models[name] = adapted
        provenance[name] = paths
    args.output.mkdir(parents=True)
    with (args.output / 'latent_umap.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['direction', 'model', 'training_seed', 'sampling_seed',
                                                   'point_id', 'domain', 'role', 'class', 'umap_1', 'umap_2'])
        writer.writeheader()
        for name, model in models.items():
            print(f'Extracting and embedding {name}...', flush=True)
            latent, _ = collect_representation(model, x)
            # Independent unsupervised layouts: compare class structure within panels only.
            coords = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2,
                               metric='euclidean', random_state=42, n_jobs=1).fit_transform(latent)
            if not np.isfinite(coords).all():
                raise ValueError('Nonfinite UMAP coordinates')
            np.savez_compressed(args.output / f'{name.replace(" ", "_")}_latent.npz',
                                latent=latent, coordinates=coords, labels=labels)
            for i, (u, v) in enumerate(coords):
                role = 'Source' if i < len(xs) else 'Target'
                domain = source if i < len(xs) else target
                writer.writerow(dict(direction=args.direction, model=name, training_seed=args.seed,
                                     sampling_seed=args.sampling_seed, point_id=f'{role}_{i}',
                                     domain=domain, role=role, **{'class': 'Benign' if labels[i] == 0 else 'Attack'},
                                     umap_1=float(u), umap_2=float(v)))
    manifest = dict(direction=args.direction, training_seed=args.seed, sampling_seed=args.sampling_seed,
                    source_rows=len(xs), target_rows=len(xt), models=list(models),
                    class_counts={role: {str(label): int((y == label).sum()) for label in [0, 1]}
                                  for role, y in [('Source', ys), ('Target', yt)]},
                    labels_used_training=False, labels_used_umap_fit=False,
                    data_role='post_hoc_validation_diagnostic_only', checkpoints=provenance,
                    umap=dict(n_neighbors=30, min_dist=0.1, metric='euclidean', random_state=42),
                    caveat='Independent UMAP layouts per model; absolute coordinates, cluster sizes and gaps across panels are not quantitatively comparable. Same sampled flows and hyperparameters in all panels.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {len(models) * len(x)} rows: {args.output / "latent_umap.csv"}', flush=True)


if __name__ == '__main__':
    main()
