"""V5f adapter gradient-conflict matrix across independent natural target batches.

Run from the NIDS-DA project root with PYTHONPATH=src. No model updates.
Conditional samples are independently drawn with replacement from the same
source latent / V2 pseudo-target pools used in V5f training.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import median

import torch

from evaluation.hda_v5b_calibration import build_frozen_target_pools, data_snapshot
from training.hda_v4 import build_source_pools, sample_pool
from training.hda_v5b import ROOT, file_hash, ranking_loss
from training.hda_v5f import checkpoint_path, load_context, load_student
from training.mkmmd import MKMMDLoss
from training.thesis_protocol import resolve_path
from training.v5e_performance import cached_pools
from training.v6_data import make_loader, make_teacher_loader

NAMES = ("hidden", "normal", "attack", "rank")
EPS = 1e-12


def load_pools(config, protocol, provenance, source, v2, device, batch_size):
    """Reuse *exactly* the dependency-keyed cache policy of V5f training."""
    source_path = ROOT / "data/features/unsw_train"
    target_path = resolve_path(protocol["target_data"]["adaptation_train"])
    source_snapshot = data_snapshot(source_path)
    target_snapshot = data_snapshot(target_path)
    producer_files = (
        "src/training/hda_v5f.py", "src/training/v5e_performance.py",
        "src/evaluation/hda_v5b_calibration.py", "src/training/hda_v4.py",
        "src/training/v6_data.py", "src/models/hda_v1.py",
        "src/models/baseline.py",
    )
    common = {
        "schema": 1,
        "torch": str(torch.__version__),
        "batch_size": batch_size,
        "producer_code": {name: file_hash(ROOT / name) for name in producer_files},
    }
    cache_dir = resolve_path(config["pool_cache_dir"])

    def build_target():
        normal, attack, metadata = build_frozen_target_pools(
            v2, protocol, provenance["target_dim"], batch_size
        )
        if data_snapshot(target_path) != target_snapshot:
            raise RuntimeError("Target data changed during pool creation")
        return {"normal": normal, "attack": attack, "metadata": metadata}

    target_cache = cached_pools(
        cache_dir,
        {
            **common,
            "kind": "target_v2_quantiles",
            "backend": "cpu",
            "data": target_snapshot,
            "target_dim": provenance["target_dim"],
            "v2_checkpoint": provenance["teacher_checkpoint_sha256"],
            "source_checkpoint": provenance["source_checkpoint_sha256"],
            "policy": protocol["pseudo_labels"],
        },
        build_target,
    )

    def build_source():
        source.to(device).eval()
        pools = build_source_pools(
            source, make_loader(source_path, provenance["source_dim"], batch_size), device
        )
        if data_snapshot(source_path) != source_snapshot:
            raise RuntimeError("Source data changed during pool creation")
        return pools

    source_pools = cached_pools(
        cache_dir,
        {
            **common,
            "kind": "source_latents",
            "backend": str(device),
            "data": source_snapshot,
            "source_dim": provenance["source_dim"],
            "source_checkpoint": provenance["source_checkpoint_sha256"],
        },
        build_source,
    )
    target_pools = {0: target_cache["normal"], 1: target_cache["attack"]}
    return source_pools, target_pools, source_path, target_path


def adapter_gradients(student, teacher, kernel, source_x, target_x,
                      source_pools, target_pools, n, weights,
                      fixed_bandwidth=None):
    """Four weighted gradients wrt the *same* adapter parameters; eval BN."""
    device = next(student.parameters()).device
    source_x = source_x.to(device)
    target_x = target_x.to(device)
    adapter_params = tuple(student.adapter.parameters())

    with torch.no_grad():
        source_h, _ = student.source_representations(source_x)
        _, teacher_logits = teacher(target_x)

    target_h = student.adapter(target_x)
    target_z = student.shared_latent(target_h)
    student_logits = student.classifier(target_z)
    balanced_x = torch.cat(
        [sample_pool(target_pools[k], n, device) for k in (0, 1)], dim=0
    )
    with student.balanced_adapter_batch():
        balanced_z = student.shared_latent(student.adapter(balanced_x))

    pairs = {
        "hidden": (source_h, target_h),
        "normal": (sample_pool(source_pools[0], n, device), balanced_z[:n]),
        "attack": (sample_pool(source_pools[1], n, device), balanced_z[n:]),
    }
    losses = {}
    for name, (s, t) in pairs.items():
        bandwidth = fixed_bandwidth[name] if fixed_bandwidth else None
        losses[name], _ = kernel(s, t, bandwidth_squared=bandwidth)
    losses["rank"] = ranking_loss(
        teacher_logits[:, 1] - teacher_logits[:, 0],
        student_logits[:, 1] - student_logits[:, 0],
    )

    gradients = {}
    for i, name in enumerate(NAMES):
        grads = torch.autograd.grad(
            weights[name] * losses[name], adapter_params,
            retain_graph=i != len(NAMES) - 1, allow_unused=True,
        )
        gradients[name] = torch.cat([
            (torch.zeros_like(p) if g is None else g).detach().flatten().cpu().double()
            for p, g in zip(adapter_params, grads)
        ])
        if not torch.isfinite(gradients[name]).all():
            raise ValueError(f"Nonfinite {name} gradient")
    return gradients


def summarize(gradient_batches):
    def med(values):
        return float(median(values)) if values else None

    pairwise = {}
    norms_by_name = {name: [] for name in NAMES}
    negative_attack_total = []
    for gradients in gradient_batches:
        for name in NAMES:
            norms_by_name[name].append(torch.linalg.vector_norm(gradients[name]).item())
        total = sum((gradients[name] for name in NAMES), torch.zeros_like(gradients[NAMES[0]]))
        negative_attack_total.append(torch.dot(gradients["attack"], total).item() < 0)

    for x in NAMES:
        pairwise[x] = {}
        for y in NAMES:
            cosines = []
            for gradients in gradient_batches:
                gx, gy = gradients[x], gradients[y]
                denominator = torch.linalg.vector_norm(gx).item() * torch.linalg.vector_norm(gy).item()
                if denominator > EPS:
                    cosines.append(torch.dot(gx, gy).item() / denominator)
            pairwise[x][y] = {
                "median_cosine": med(cosines),
                "negative_fraction": (sum(v < 0 for v in cosines) / len(cosines)) if cosines else None,
                "valid_batches": len(cosines),
            }

    return {
        "num_batches": len(gradient_batches),
        "pairwise": pairwise,
        "median_weighted_gradient_norms": {name: med(values) for name, values in norms_by_name.items()},
        "attack_dot_total_negative_fraction": sum(negative_attack_total) / len(negative_attack_total),
    }


def render(report):
    print("\nPAIRWISE COSINE: median (fraction < 0)")
    print(f"{'':>10}" + "".join(f"{n:>22}" for n in NAMES))
    for x in NAMES:
        cells = []
        for y in NAMES:
            cell = report["pairwise"][x][y]
            if cell["median_cosine"] is None:
                cells.append(f"{'undefined':>22}")
            else:
                label = f"{cell['median_cosine']:+.3f} ({cell['negative_fraction']:.0%})"
                cells.append(f"{label:>22}")
        print(f"{x:>10}" + "".join(cells))
    print("Median weighted gradient norms:", report["median_weighted_gradient_norms"])
    print("Attack dot total < 0 fraction:", f"{report['attack_dot_total_negative_fraction']:.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/hda_v5f.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--sample-seed", type=int, default=2026)
    ap.add_argument("--bandwidth", choices=("batch", "fixed"), default="batch")
    ap.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    ap.add_argument("--output", default="results/hda_v5f/diagnostics/gradient_matrix_seed42.json")
    args = ap.parse_args()
    if args.batches < 2:
        ap.error("Use at least two distinct natural batches")
    output = Path(args.output)
    if output.exists():
        ap.error(f"Output exists: {output}. Specify a fresh --output path.")
    device = torch.device(args.device)
    torch.manual_seed(args.sample_seed)
    config, protocol, provenance, source, v2, teacher = load_context(args.config, args.seed)
    student, checkpoint = load_student(config, provenance, source, teacher)
    student = student.to(device).eval().requires_grad_(False)
    student.adapter.requires_grad_(True)
    teacher = teacher.to(device).eval().requires_grad_(False)
    kernel = MKMMDLoss(checkpoint["kernel_scales"]).to(
        device=device, dtype=next(student.parameters()).dtype
    )
    settings = protocol["training"]
    n = settings["class_batch_size"]
    source_pools, target_pools, source_path, target_path = load_pools(
        config, protocol, provenance, source, v2, device, settings["batch_size"]
    )
    source_iter = iter(make_loader(source_path, provenance["source_dim"], settings["batch_size"]))
    target_iter = iter(make_teacher_loader(target_path, provenance["target_dim"], settings["batch_size"]))
    fixed = (checkpoint["fixed_diagnostic_batches"]["bandwidth_squared"]
             if args.bandwidth == "fixed" else None)
    rows = []
    for i in range(args.batches):
        try:
            source_x, _ = next(source_iter)
            target_x = next(target_iter)
        except StopIteration:
            raise ValueError(f"Only {i} paired batches available") from None
        grads = adapter_gradients(student, teacher, kernel, source_x, target_x,
                                  source_pools, target_pools, n, checkpoint["loss_weights"], fixed)
        rows.append(grads)
        if (i + 1) % 10 == 0:
            print(f"Diagnostic batch {i + 1}/{args.batches}", flush=True)
    result = summarize(rows)
    result.update({
        "checkpoint": str(checkpoint_path(config)),
        "checkpoint_sha256": file_hash(checkpoint_path(config)),
        "training_seed": config["training_seed"],
        "pseudo_label_source": "frozen V2 quantile pools on adaptation-train only",
        "source_samples": "UNSW training features and source latent pools",
        "target_samples": "unlabeled CICIDS adaptation-train features and pseudo pools",
        "batch_policy": "distinct natural source/target batches; random conditional pool samples (with replacement)",
        "sample_seed": args.sample_seed,
        "bandwidth_mode": args.bandwidth,
        "bn_mode": "eval",
        "gradient_scope": "target adapter, weighted losses, before Adam preconditioning",
        "device": args.device,
        "loss_weights": checkpoint["loss_weights"],
        "kernel_scales": checkpoint["kernel_scales"],
        "note": "No optimizer.step; descriptive diagnostics at one final checkpoint, not training trajectory",
    })
    render(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Saved:", output)


if __name__ == "__main__":
    main()
