"""Diagnose frozen V2 pseudo pools without reading target labels."""
import argparse
import json
import math

import torch

from evaluation.v5f_gradient_matrix import load_pools
from training.hda_v5f import checkpoint_path, load_context, load_student
from training.hda_v5b import file_hash
from training.thesis_protocol import resolve_path


def margin_distribution(values, bins, threshold=0.0):
    values = values.double()
    if not len(values) or not torch.isfinite(values).all():
        raise ValueError("Empty or nonfinite teacher margins")
    levels = torch.tensor([0., .01, .05, .25, .5, .75, .95, .99, 1.], dtype=torch.float64)
    counts, edges = torch.histogram(values, bins=bins)
    quantiles = torch.quantile(values, levels)
    return {"count": len(values), "min": values.min().item(),
            "threshold": threshold, "threshold_rule": "margin > threshold",
            "fraction_above_threshold": (values > threshold).double().mean().item(),
            "fraction_equal_threshold": (values == threshold).double().mean().item(),
            "median": quantiles[4].item(), "max": values.max().item(),
            "quantiles": dict(zip([str(x) for x in levels.tolist()], quantiles.tolist())),
            "histogram": {"edges": edges.tolist(), "counts": counts.long().tolist()}}


@torch.no_grad()
def inspect_pool(pool, teacher, models, batch_size, bins, threshold=0.0):
    margins = []
    sums = {}
    for x in pool.split(batch_size):
        _, logits = teacher(x)
        margins.append((logits[:, 1] - logits[:, 0]).double())
        for name, model in models.items():
            z, _ = model(x)
            if not torch.isfinite(z).all():
                raise ValueError("Nonfinite latent values")
            total = z.double().sum(0)
            sums[name] = sums.get(name, torch.zeros_like(total)) + total
    return margin_distribution(torch.cat(margins), bins, threshold), {k: v / len(pool) for k, v in sums.items()}


@torch.no_grad()
def pool_scatter(pool, model, centroid, source_normal, source_attack, label, batch_size):
    """Second pass: stable scatter and strict nearest-centroid counts, no target labels."""
    scatter = 0.0
    normal = attack = ties = 0
    for x in pool.split(batch_size):
        z, _ = model(x)
        z = z.double()
        if not torch.isfinite(z).all():
            raise ValueError("Nonfinite latent values")
        scatter += ((z - centroid) ** 2).sum().item()
        dn = ((z - source_normal) ** 2).sum(1)
        da = ((z - source_attack) ** 2).sum(1)
        normal += (dn < da).sum().item()
        attack += (da < dn).sum().item()
        ties += (dn == da).sum().item()
    count = len(pool)
    return {"count": count,
            "nearest_source_normal_fraction": normal / count,
            "nearest_source_attack_fraction": attack / count,
            "nearest_source_centroid_tie_fraction": ties / count,
            "nearest_source_same_class_ratio": (normal if label == 0 else attack) / count,
            "within_class_scatter": scatter / count,
            "rms_radius": math.sqrt(scatter / count)}


def fisher_separation(normal_centroid, attack_centroid, normal_stats, attack_stats):
    between = ((normal_centroid - attack_centroid) ** 2).sum().item()
    within = normal_stats["within_class_scatter"] + attack_stats["within_class_scatter"]
    return {"between_centroid_squared_distance": between,
            "sum_within_class_scatter": within,
            "fisher_ratio": between / within if within > 0 else None,
            "definition": "||mu_N-mu_A||^2 / (mean_N ||z-mu_N||^2 + mean_A ||z-mu_A||^2)",
            "undefined_reason": "zero within-class scatter" if within == 0 else None}


def geometry(tn, ta, sn, sa):
    centers = {"T_N": tn, "T_A": ta, "S_N": sn, "S_A": sa}
    pairs = (("T_N", "T_A"), ("T_N", "S_N"), ("T_N", "S_A"),
             ("T_A", "S_A"), ("T_A", "S_N"))
    distances = {f"d({a},{b})": torch.linalg.vector_norm(centers[a] - centers[b]).item() for a, b in pairs}
    return {"centroids": {k: v.tolist() for k, v in centers.items()},
            "euclidean_distances": distances,
            "normal_closer_to_source_normal": distances["d(T_N,S_N)"] < distances["d(T_N,S_A)"],
            "attack_closer_to_source_attack": distances["d(T_A,S_A)"] < distances["d(T_A,S_N)"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5f.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--margin-threshold", type=float, default=0.0,
                        help="Raw frozen V2 teacher margin threshold (default: 0; not V5f calibrated threshold)")
    parser.add_argument("--output", default="results/hda_v5f/diagnostics/pool_geometry_seed42.json")
    args = parser.parse_args()
    if args.bins < 1:
        parser.error("--bins must be positive")
    if not math.isfinite(args.margin_threshold):
        parser.error("--margin-threshold must be finite")
    output = resolve_path(args.output)
    if output.exists():
        parser.error("Output exists; choose a fresh --output")
    config, protocol, provenance, source, v2, teacher = load_context(args.config, args.seed)
    student, checkpoint = load_student(config, provenance, source, teacher)
    models = {"v2_pool_teacher": v2, "v5f": student}
    for model in (*models.values(), source):
        model.cpu().eval().requires_grad_(False)
    batch_size = protocol["training"]["batch_size"]
    sp, tp, _, _ = load_pools(config, protocol, provenance, source, v2, torch.device("cpu"), batch_size)
    distributions, centroids = {}, {}
    for label, name in ((0, "normal"), (1, "attack")):
        distributions[name], centroids[name] = inspect_pool(
            tp[label], v2, models, batch_size, args.bins, args.margin_threshold)
        print(f"Pseudo {name}: {distributions[name]['count']} samples; "
              f"margin min/median/max = {distributions[name]['min']:.6g} / "
              f"{distributions[name]['median']:.6g} / {distributions[name]['max']:.6g}", flush=True)
        print(f"  Fraction V2 margin > {args.margin_threshold}: "
              f"{distributions[name]['fraction_above_threshold']:.2%}", flush=True)
    sn, sa = (sp[k].double().mean(0) for k in (0, 1))
    result = {"training_seed": config["training_seed"], "checkpoint": str(checkpoint_path(config)),
              "checkpoint_sha256": file_hash(checkpoint_path(config)),
              "provenance": provenance, "pseudo_policy": protocol["pseudo_labels"],
              "target_labels_used": False, "teacher_margin": "frozen V2 logit_attack - logit_normal",
              "latent_space": "post-ReLU shared latent, before classifier; eval mode; CPU",
              "source_counts": {"normal": len(sp[0]), "attack": len(sp[1])},
              "margin_distributions": distributions, "geometry": {},
              "note": "Centroid proximity is descriptive, not proof of pseudo-label correctness."}
    for name in models:
        values = geometry(centroids["normal"][name], centroids["attack"][name], sn, sa)
        values["pool_statistics"] = {
            pool_name: pool_scatter(tp[label], models[name], centroids[pool_name][name],
                                    sn, sa, label, batch_size)
            for label, pool_name in ((0, "normal"), (1, "attack"))}
        values["separation"] = fisher_separation(
            centroids["normal"][name], centroids["attack"][name],
            values["pool_statistics"]["normal"], values["pool_statistics"]["attack"])
        result["geometry"][name] = values
        print(name, json.dumps({k: v for k, v in values.items() if k != "centroids"}, indent=2))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Saved:", output)


if __name__ == "__main__":
    main()
