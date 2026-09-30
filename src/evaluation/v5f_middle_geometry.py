"""Post-hoc geometry audit inside the V2 middle region.

Purpose
-------
Check whether source-class geometry can separate true Normal vs Attack
among target DEVELOPMENT samples that were not covered by the frozen
hard conditional pseudo-label policy:

    q02 < V2 margin < q95

This script:
- does NOT train;
- does NOT update checkpoints;
- does NOT fit thresholds;
- uses DEVELOPMENT labels only for post-hoc diagnosis;
- uses frozen adaptation-train q02/q95/q98 from the V5f checkpoint;
- uses the frozen V2 latent representation, NOT the adapted V5f latent.
"""

import argparse
import json

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation.v5f_gradient_matrix import load_pools
from evaluation.hda_v5b_calibration import data_snapshot
from evaluation.hda_v5f import dependencies, verify_training_data
from training.hda_v5f import load_context, load_student, checkpoint_path, load_v5d_reference
from training.hda_v5b import file_hash
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader


def format_value(value, spec):
    return format(value, spec) if value is not None else 'undefined'


def distribution(values):
    """Simple descriptive statistics."""
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return None
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite diagnostic values')

    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std_population": float(np.std(values)),
    }


def summarize_group(mask, labels, margins, geometry_score, d_normal, d_attack):
    n = int(mask.sum())

    if n == 0:
        return {
            "count": 0,
            "fraction": None,
            "attack_count": 0,
            "attack_prevalence": None,
        }

    return {
        "count": n,
        "attack_count": int(labels[mask].sum()),
        "attack_prevalence": float(labels[mask].mean()),
        "v2_margin": distribution(margins[mask]),
        "geometry_score": distribution(geometry_score[mask]),
        "distance_to_source_normal": distribution(d_normal[mask]),
        "distance_to_source_attack": distribution(d_attack[mask]),
    }


def geometry_bins(score, labels, bins=5):
    """Attack prevalence from low -> high geometry-Attack confidence.

    geometry_score = d(source Normal) - d(source Attack)

    Larger score:
        target sample is relatively closer to Source Attack.

    Smaller / negative score:
        relatively closer to Source Normal.
    """
    if bins < 2:
        raise ValueError("geometry bins must be >= 2")
    if not len(score) or not np.isfinite(score).all():
        raise ValueError('Empty or nonfinite geometry scores')

    cuts = np.quantile(
        score,
        np.linspace(0, 1, bins + 1),
    )

    rows = []

    for i in range(bins):
        if i == 0:
            mask = score <= cuts[i + 1]
        elif i == bins - 1:
            mask = score > cuts[i]
        else:
            mask = (score > cuts[i]) & (score <= cuts[i + 1])

        n = int(mask.sum())

        rows.append({
            "bin": i + 1,
            "range_low": float(cuts[i]),
            "range_high": float(cuts[i + 1]),
            "count": n,
            "attack_count": int(labels[mask].sum()) if n else 0,
            "attack_prevalence": float(labels[mask].mean()) if n else None,
            "median_geometry_score": (
                float(np.median(score[mask])) if n else None
            ),
        })

    return {
        "definition": (
            "geometry_score = distance_to_SourceNormal "
            "- distance_to_SourceAttack; larger means more Attack-like"
        ),
        "quantile_bins": rows,
    }


@torch.no_grad()
def score_development(
    v2,
    loader,
    source_normal_centroid,
    source_attack_centroid,
    q02,
    q95,
):
    margins = []
    labels = []
    geometry_scores = []
    distances_normal = []
    distances_attack = []

    seen = 0
    middle_seen = 0

    for batch_index, (x, y) in enumerate(loader, 1):
        # Frozen V2 latent + logits.
        latent, logits = v2(x)

        margin = logits[:, 1] - logits[:, 0]
        if not torch.isfinite(margin).all() or not torch.isfinite(latent).all():
            raise ValueError('Nonfinite V2 margins or latent values')

        # IMPORTANT:
        # Exact numeric thresholds frozen from ADAPTATION-TRAIN.
        middle = (margin > q02) & (margin < q95)

        seen += len(x)

        if not middle.any():
            continue

        z = latent[middle].double()

        sn = source_normal_centroid.to(
            device=z.device,
            dtype=z.dtype,
        )

        sa = source_attack_centroid.to(
            device=z.device,
            dtype=z.dtype,
        )

        # Euclidean distance to frozen source class centroids.
        d_n = torch.linalg.vector_norm(
            z - sn,
            dim=1,
        )

        d_a = torch.linalg.vector_norm(
            z - sa,
            dim=1,
        )

        # Positive = closer to Source Attack.
        geometry_score = d_n - d_a

        margins.append(
            margin[middle].cpu().double().numpy()
        )

        labels.append(
            y[middle].cpu().numpy()
        )

        geometry_scores.append(
            geometry_score.cpu().numpy()
        )

        distances_normal.append(
            d_n.cpu().numpy()
        )

        distances_attack.append(
            d_a.cpu().numpy()
        )

        middle_seen += int(middle.sum())

        if batch_index % 200 == 0:
            print(
                f"Scored {seen:,} development rows | "
                f"middle={middle_seen:,}",
                flush=True,
            )

    if not margins:
        raise ValueError(
            "No development samples found inside q02 < margin < q95"
        )

    return (
        np.concatenate(margins),
        np.concatenate(labels).astype(np.int64),
        np.concatenate(geometry_scores),
        np.concatenate(distances_normal),
        np.concatenate(distances_attack),
        seen,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--config",
        default="configs/hda_v5f.json",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--geometry-bins",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        default=(
            "results/hda_v5f/diagnostics/"
            "middle_geometry_seed42.json"
        ),
    )

    args = parser.parse_args()
    if args.geometry_bins < 2:
        parser.error('--geometry-bins must be >= 2')

    output = resolve_path(args.output)

    if output.exists():
        parser.error(
            f"Output already exists: {output}. "
            "Choose a fresh --output path."
        )

    # --------------------------------------------------
    # 1. Load frozen context
    # --------------------------------------------------

    (
        config,
        protocol,
        provenance,
        source,
        v2,
        teacher,
    ) = load_context(
        args.config,
        training_seed=args.seed,
    )

    # Load V5f only to validate provenance/checkpoint
    # and recover pseudo_metadata.
    _, checkpoint = load_student(
        config,
        provenance,
        source,
        teacher,
    )
    verify_training_data(protocol, checkpoint)
    reference = load_v5d_reference(config, protocol, provenance)
    initial_dependencies = dependencies(config, provenance)

    pseudo = checkpoint["pseudo_metadata"]

    q02 = float(pseudo["q02"])
    q95 = float(pseudo["q95"])
    q98 = float(pseudo["q98"])
    if not np.isfinite([q02, q95, q98]).all() or not q02 < q95 <= q98:
        raise ValueError('Invalid frozen pseudo-pool quantiles')

    print("\n=== FROZEN ADAPTATION-TRAIN THRESHOLDS ===")
    print(f"q02 = {q02:.6f}")
    print(f"q95 = {q95:.6f}")
    print(f"q98 = {q98:.6f}")

    # --------------------------------------------------
    # 2. Frozen models on CPU / eval mode
    # --------------------------------------------------

    device = torch.device("cpu")

    source = source.to(device).eval().requires_grad_(False)
    v2 = v2.to(device).eval().requires_grad_(False)

    batch_size = protocol["training"]["batch_size"]

    # --------------------------------------------------
    # 3. Reuse EXACT source latent pool definition
    #    used by V5f training.
    # --------------------------------------------------

    source_pools, _, _, _ = load_pools(
        config,
        protocol,
        provenance,
        source,
        v2,
        device,
        batch_size,
    )

    source_normal_centroid = (
        source_pools[0]
        .double()
        .mean(dim=0)
    )

    source_attack_centroid = (
        source_pools[1]
        .double()
        .mean(dim=0)
    )

    print("\nSource latent pools:")
    print(f"Normal: {len(source_pools[0]):,}")
    print(f"Attack: {len(source_pools[1]):,}")

    centroid_distance = torch.linalg.vector_norm(
        source_normal_centroid -
        source_attack_centroid
    ).item()

    print(
        "Source centroid distance:",
        f"{centroid_distance:.6f}",
    )

    # --------------------------------------------------
    # 4. Score DEVELOPMENT
    # --------------------------------------------------

    target_path = evaluation_target(
        protocol,
        "development",
    )
    development_snapshot = data_snapshot(target_path)
    if development_snapshot != reference['development_files']:
        raise ValueError('Development data differs from pinned V5d reference')
    expected_rows = sum(pq.read_metadata(p).num_rows for p in sorted(target_path.glob('*.parquet')))

    loader = make_loader(
        target_path,
        provenance["target_dim"],
        batch_size,
    )

    (
        margins,
        labels,
        geometry_score,
        d_normal,
        d_attack,
        total_seen,
    ) = score_development(
        v2,
        loader,
        source_normal_centroid,
        source_attack_centroid,
        q02,
        q95,
    )
    if total_seen != expected_rows:
        raise ValueError(f'Incomplete development scoring: {total_seen}/{expected_rows}')
    if (data_snapshot(target_path) != development_snapshot
            or dependencies(config, provenance) != initial_dependencies):
        raise ValueError('Inputs changed during diagnostic')

    middle_n = len(labels)

    print("\n=== MIDDLE REGION ===")
    print(
        f"q02 < V2 margin < q95: "
        f"N={middle_n:,} / {total_seen:,} "
        f"({middle_n / total_seen:.2%})"
    )

    print(
        "True Attack prevalence:",
        f"{labels.mean():.2%}",
    )

    # --------------------------------------------------
    # 5. Nearest source centroid
    # --------------------------------------------------

    closer_normal = d_normal < d_attack
    closer_attack = d_attack < d_normal
    ties = d_attack == d_normal

    normal_stats = summarize_group(
        closer_normal,
        labels,
        margins,
        geometry_score,
        d_normal,
        d_attack,
    )

    attack_stats = summarize_group(
        closer_attack,
        labels,
        margins,
        geometry_score,
        d_normal,
        d_attack,
    )

    tie_stats = summarize_group(
        ties,
        labels,
        margins,
        geometry_score,
        d_normal,
        d_attack,
    )

    for row in (
        normal_stats,
        attack_stats,
        tie_stats,
    ):
        row["fraction_of_middle"] = (
            row["count"] / middle_n
            if middle_n
            else None
        )

    # --------------------------------------------------
    # 6. Does geometry contain class information?
    # --------------------------------------------------

    if len(np.unique(labels)) == 2:
        geometry_roc_auc = float(
            roc_auc_score(
                labels,
                geometry_score,
            )
        )

        geometry_ap = float(
            average_precision_score(
                labels,
                geometry_score,
            )
        )

    else:
        geometry_roc_auc = None
        geometry_ap = None

    bins = geometry_bins(
        geometry_score,
        labels,
        args.geometry_bins,
    )

    # --------------------------------------------------
    # 7. Print result
    # --------------------------------------------------

    print("\n=== NEAREST SOURCE CENTROID ===")

    print(
        "\nCloser to Source Normal:"
    )
    print(
        f"N={normal_stats['count']:,} "
        f"({normal_stats['fraction_of_middle']:.2%})"
    )
    print(
        "Attack prevalence:",
        format_value(normal_stats['attack_prevalence'], '.2%'),
    )

    print(
        "\nCloser to Source Attack:"
    )
    print(
        f"N={attack_stats['count']:,} "
        f"({attack_stats['fraction_of_middle']:.2%})"
    )
    print(
        "Attack prevalence:",
        format_value(attack_stats['attack_prevalence'], '.2%'),
    )

    print(
        "\nCentroid ties:",
        f"{tie_stats['count']:,}",
    )

    if (
        normal_stats["attack_prevalence"] is not None
        and attack_stats["attack_prevalence"] is not None
    ):
        difference = (
            attack_stats["attack_prevalence"]
            - normal_stats["attack_prevalence"]
        )

        print(
            "\nAttack prevalence difference "
            "(closer Attack - closer Normal):",
            f"{difference:+.2%}",
        )

    print("\n=== GEOMETRY SCORE PERFORMANCE ===")
    print(
        "ROC-AUC:",
        geometry_roc_auc,
    )
    print(
        "AP:",
        geometry_ap,
    )

    print(
        f"\nGeometry quantile bins ({args.geometry_bins}) "
        "(higher = more Source-Attack-like):"
    )

    print(
        f"{'Bin':>5} "
        f"{'N':>10} "
        f"{'Attack prev':>15} "
        f"{'Median g':>15}"
    )

    for row in bins["quantile_bins"]:
        prevalence = row["attack_prevalence"]

        print(
            f"{row['bin']:>5} "
            f"{row['count']:>10,d} "
            f"{format_value(prevalence, '.2%'):>14} "
            f"{format_value(row['median_geometry_score'], '.4f'):>15}"
        )

    # --------------------------------------------------
    # 8. Save reproducible report
    # --------------------------------------------------

    result = {
        "dependencies": initial_dependencies,
        "development_path": str(target_path),
        "development_snapshot": development_snapshot,
        "expected_development_rows": expected_rows,
        "source_pool_counts": {"normal": len(source_pools[0]), "attack": len(source_pools[1])},
        "source_centroid_vectors": {"normal": source_normal_centroid.tolist(), "attack": source_attack_centroid.tolist()},
        "phase": "post-hoc development diagnostic",
        "training_seed": config["training_seed"],
        "checkpoint": str(checkpoint_path(config)),
        "checkpoint_sha256": file_hash(
            checkpoint_path(config)
        ),
        "target_labels_used_for_training": False,
        "target_labels_used_for_this_diagnostic": True,
        "note": (
            "Development labels are used only for post-hoc "
            "diagnosis. Do not use final-test labels for model design."
        ),
        "latent_space": (
            "Frozen V2 target latent representation; "
            "post-ReLU shared latent before classifier"
        ),
        "source_centroids": (
            "Means of the same frozen source latent class pools "
            "used in V5f conditional alignment"
        ),
        "geometry_score_definition": (
            "d(target, SourceNormal centroid) "
            "- d(target, SourceAttack centroid); "
            "positive means relatively closer to Source Attack"
        ),
        "frozen_adaptation_thresholds": {
            "q02": q02,
            "q95": q95,
            "q98": q98,
        },
        "middle_definition": (
            "q02 < frozen V2 margin < q95"
        ),
        "development_rows": total_seen,
        "middle_rows": middle_n,
        "middle_fraction": middle_n / total_seen,
        "middle_true_attack_prevalence": float(
            labels.mean()
        ),
        "source_centroid_distance": centroid_distance,
        "nearest_source_centroid": {
            "closer_source_normal": normal_stats,
            "closer_source_attack": attack_stats,
            "ties": tie_stats,
        },
        "geometry_score_metrics": {
            "roc_auc": geometry_roc_auc,
            "average_precision": geometry_ap,
            "baseline_attack_prevalence": float(
                labels.mean()
            ),
        },
        "geometry_bins": bins,
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "x",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            allow_nan=False,
        )
        f.write("\n")

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
