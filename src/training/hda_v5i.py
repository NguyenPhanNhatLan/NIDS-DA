"""V5i: asymmetric selective MMD; weighted Normal middle, hard-only Attack."""
import argparse
import copy
import hashlib
import json
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from evaluation.hda_v5b_calibration import (
    build_frozen_target_pools, calibration_code_hashes, data_snapshot, load_frozen_models,
)
from models.hda_v5d import HDAV5DModel
from training.baseline import set_seed
from training.hda_v4 import build_source_pools, sample_pool
from training.hda_v5b import ROOT, file_hash, ranking_loss
from training.hda_v5f import (
    V5D_WEIGHTS, V5F_WEIGHTS, baseline_class_weights, diagnostic_metrics,
    load_v5d_reference, make_diagnostic,
)
from training.mkmmd import MKMMDLoss
from training.thesis_protocol import evaluation_target, resolve_path
from training.v5e_performance import cached_pools, StepProfiler
from training.v6_data import make_loader, make_teacher_loader, make_unlabeled_loader
from training.weighted_mkmmd import WeightedMKMMDLoss


V5I_WEIGHTS = {'hidden': 0.1, 'normal': 0.05, 'attack': 0.08, 'rank': 0.1, 'source': 0.1}


def code_hashes():
    hashes = calibration_code_hashes()
    for name in (
        "src/models/hda_v5d.py",
        "src/training/hda_v5f.py",
        "src/training/hda_v5i.py",
        "src/evaluation/hda_v5i.py",
        "src/training/mkmmd.py",
        "src/training/weighted_mkmmd.py",
        "src/training/v5e_performance.py",
        "src/training/adaptation.py",
        "src/training/baseline.py",
        "src/training/thesis_protocol.py",
        "src/evaluation/protocol_revision.py",
    ):
        hashes[name] = file_hash(ROOT / name)
    return hashes


def load_context(config_path, training_seed=None):
    path = resolve_path(config_path)
    config = json.loads(path.read_text())
    if (not config["kernel_scales"]
            or any(not math.isfinite(v) or v <= 0 for v in config["kernel_scales"])):
        raise ValueError("kernel_scales must contain positive finite multipliers of bandwidth squared")
    if (config["classifier_only"] or config["loss_weights"] != V5I_WEIGHTS
            or config["adapter_lr"] != 1e-4 or config["classifier_lr"] != 1e-5
            or config["weight_decay"] != 1e-4):
        raise ValueError("V5i requires gradient-derived loss weights and preserves V5h optimizer settings")
    if training_seed is not None:
        config["training_seed"] = training_seed
    if type(config["training_seed"]) is not int or config["training_seed"] not in (42, 43, 44):
        raise ValueError("training_seed must be 42, 43 or 44")
    if type(config["classifier_only"]) is not bool:
        raise ValueError("classifier_only must be a boolean")
    for name in ("adapter_lr", "classifier_lr"):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("weight_decay must be nonnegative and finite")
    if set(config["loss_weights"]) != {"hidden", "normal", "attack", "rank", "source"}:
        raise ValueError("Expected all five V5i/V5f loss weights")
    if any(not math.isfinite(v) or v < 0 for v in config["loss_weights"].values()):
        raise ValueError("Loss weights must be nonnegative and finite")
    if not 0 <= config["calibration_max_fpr"] <= 1:
        raise ValueError("calibration_max_fpr must be in [0, 1]")
    power = config["geometry_weight_power"]
    anchor_mass = config["conditional_anchor_mass"]
    if not math.isfinite(power) or power != 2.0:
        raise ValueError("V5i requires geometry_weight_power=2 for normal weights (1-r)^2")
    if not math.isfinite(anchor_mass) or not 0 < anchor_mass < 1:
        raise ValueError("conditional_anchor_mass must be strictly between 0 and 1")
    if config.get("exclude_q98_plus") is not True:
        raise ValueError("First V5i experiment must exclude q98+ from conditional alignment")

    reference_path = resolve_path(config["stage1_reference"])
    reference, protocol, provenance, source, v2, teacher = load_frozen_models(reference_path)
    if config["teacher_seed"] != 42 or reference["seed"] != config["teacher_seed"]:
        raise ValueError("V5i requires frozen V5b asymmetric seed 42")
    if reference["loss_weights"] != {"hidden": 1.0, "normal": 0.05, "attack": 0.02, "rank": 0.10}:
        raise ValueError("Expected asymmetric V5b teacher")
    if resolve_path(protocol["target_data"]["adaptation_train"]).resolve() == evaluation_target(protocol, "development").resolve():
        raise ValueError("Adaptation and development must be separate")
    if resolve_path(config["source_validation"]).resolve() != (ROOT / "data/features/unsw_val").resolve():
        raise ValueError("Use the fixed UNSW validation split")
    if resolve_path(config["source_metadata"]).resolve() != (ROOT / "data/features/unsw_metadata.json").resolve():
        raise ValueError("Use the same UNSW class-count metadata as baseline")

    provenance = {
        **provenance,
        "config_sha256": file_hash(path),
        "training_seed": config["training_seed"],
        "teacher_seed": config["teacher_seed"],
        "classifier_only": config["classifier_only"],
        "stage1_reference_sha256": file_hash(reference_path),
        "stage1_checkpoint_sha256": reference["checkpoint_sha256"],
        "source_metadata_sha256": file_hash(resolve_path(config["source_metadata"])),
    }
    return config, protocol, provenance, source, v2, teacher


def checkpoint_path(config):
    return resolve_path(config["checkpoint_dir"]) / f"v5i_seed{config['training_seed']}.pt"


def load_student(config, provenance, source, teacher):
    checkpoint = torch.load(checkpoint_path(config), map_location="cpu", weights_only=True)
    if (checkpoint.get("version") != "v5i" or checkpoint.get("architecture") != "hda_v5i"
            or checkpoint["provenance"] != provenance or checkpoint["code_sha256"] != code_hashes()
            or checkpoint["training_seed"] != config["training_seed"]
            or checkpoint["teacher_seed"] != config["teacher_seed"]
            or checkpoint["classifier_only"] != config["classifier_only"]):
        raise ValueError("V5i checkpoint provenance mismatch")
    model = HDAV5DModel(source, teacher.adapter, classifier_only=config["classifier_only"])
    model.adapter.load_state_dict(checkpoint["target_adapter_state_dict"])
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    model.eval().requires_grad_(False)
    return model, checkpoint


def tensor_sha256(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous().double().numpy()
        digest.update(value.tobytes())
    return digest.hexdigest()


def average_percentile_rank(values):
    """Tie-aware percentile ranks in [0,1], using average rank for ties."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Need at least two finite geometry scores")
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    starts = np.r_[0, np.flatnonzero(sorted_x[1:] != sorted_x[:-1]) + 1]
    ends = np.r_[starts[1:], len(x)]
    sorted_rank = np.empty(len(x), dtype=np.float64)
    for start, end in zip(starts, ends):
        # Average zero-based rank within [start, end).
        sorted_rank[start:end] = 0.5 * (start + end - 1)
    rank = np.empty(len(x), dtype=np.float64)
    rank[order] = sorted_rank
    return rank / (len(x) - 1)


@torch.no_grad()
def build_geometry_middle_pool(v2, target_path, target_dim, batch_size,
                               source_normal_centroid, source_attack_centroid,
                               q02, q95, power):
    """Freeze middle target features and geometry-based soft memberships.

    Geometry is computed in the frozen V2 post-ReLU shared latent space.
    No target labels are read. The V2 middle interval is q02 < margin < q95.
    """
    # V2 shares the frozen source tail with source/V5b. Moving it in place to
    # CPU would move the ranking teacher's tail off MPS/CUDA on a cache miss.
    v2 = copy.deepcopy(v2).cpu().eval().requires_grad_(False)
    sn = source_normal_centroid.cpu().double()
    sa = source_attack_centroid.cpu().double()
    feature_parts, geometry_parts = [], []
    seen = middle = 0
    for x in make_teacher_loader(target_path, target_dim, batch_size):
        z, logits = v2(x)
        margin = logits[:, 1] - logits[:, 0]
        if not torch.isfinite(margin).all() or not torch.isfinite(z).all():
            raise ValueError("Nonfinite V2 margins or latent representations")
        mask = (margin > q02) & (margin < q95)
        seen += len(x)
        if mask.any():
            zz = z[mask].double()
            dn = torch.linalg.vector_norm(zz - sn, dim=1)
            da = torch.linalg.vector_norm(zz - sa, dim=1)
            feature_parts.append(x[mask].cpu())
            geometry_parts.append((dn - da).cpu())
            middle += int(mask.sum())
    if seen == 0 or middle < 2:
        raise ValueError("Target adaptation set or V2 middle region is empty")
    features = torch.cat(feature_parts)
    geometry = torch.cat(geometry_parts).double().numpy()
    ranks = average_percentile_rank(geometry)
    normal_weight = np.power(1.0 - ranks, power)
    if not np.isfinite(normal_weight).all():
        raise ValueError("Nonfinite geometry membership weights")
    normal_weight = torch.from_numpy(normal_weight.astype(np.float32))
    rank_tensor = torch.from_numpy(ranks.astype(np.float32))
    if len(features) != len(normal_weight):
        raise ValueError("Middle feature/weight alignment failed")

    def ess(w):
        w = w.double()
        return ((w.sum() ** 2) / w.square().sum().clamp_min(1e-12)).item()

    metadata = {
        "rows_scored": seen,
        "middle_rows": middle,
        "middle_fraction": middle / seen,
        "q02": float(q02),
        "q95": float(q95),
        "geometry": "V2 latent: ||z-mu_SN||_2 - ||z-mu_SA||_2",
        "rank": "tie-aware percentile rank of geometry score within adaptation-train middle only",
        "weight_power": float(power),
        "normal_weight": "(1-r)^power",
        "target_labels_used": False,
        "geometry_min": float(np.min(geometry)),
        "geometry_median": float(np.median(geometry)),
        "geometry_max": float(np.max(geometry)),
        "rank_min": float(np.min(ranks)),
        "rank_median": float(np.median(ranks)),
        "rank_max": float(np.max(ranks)),
        "normal_weight_ess": ess(normal_weight),
    }
    return {
        "features": features,
        "normal_weight": normal_weight,
        "geometry_rank": rank_tensor,
        "metadata": metadata,
    }


def component_weights(raw_middle, n_anchor, anchor_mass, device, dtype):
    """Make a target mixture with exact anchor and middle probability mass."""
    raw = raw_middle.to(device=device, dtype=dtype)
    if raw.ndim != 1 or len(raw) == 0 or not torch.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Middle conditional weights must be a finite nonnegative vector")
    total = raw.sum()
    if total <= 0:
        raise ValueError("Sampled middle batch has zero conditional weight mass")
    anchor = torch.full((n_anchor,), anchor_mass / n_anchor, device=device, dtype=dtype)
    middle = (1.0 - anchor_mass) * raw / total
    return torch.cat((anchor, middle))


def sample_conditional_inputs(hard_target_pools, middle_pool, n, anchor_mass, device):
    if n < 2:
        raise ValueError("class_batch_size must be at least 2")
    n_anchor = int(round(n * anchor_mass))
    n_anchor = min(max(n_anchor, 1), n - 1)
    n_middle = n - n_anchor

    middle_idx = torch.randint(len(middle_pool["features"]), (n_middle,))
    middle_x = middle_pool["features"][middle_idx].to(device)
    wn = middle_pool["normal_weight"][middle_idx]

    normal_anchor = sample_pool(hard_target_pools[0], n_anchor, device)
    attack_anchor = sample_pool(hard_target_pools[1], n, device)
    normal_x = torch.cat((normal_anchor, middle_x), dim=0)
    attack_x = attack_anchor  # V5f policy: no middle samples in Attack MMD.
    dtype = normal_x.dtype
    normal_w = component_weights(wn, n_anchor, anchor_mass, device, dtype)
    return normal_x, normal_w, attack_x


def train_model(student, teacher, source_loader, target_loader, source_pools,
                hard_target_pools, middle_pool, class_weights, config, settings,
                diagnostic=None, profile_steps=0, profile_result=None):
    device = next(student.parameters()).device
    teacher.eval().requires_grad_(False)
    weights = config["loss_weights"]
    optimizer = torch.optim.Adam(
        student.optimizer_groups(config["adapter_lr"], config["classifier_lr"]),
        weight_decay=config["weight_decay"],
    )
    class_weights = class_weights.to(device)
    n = settings["class_batch_size"]
    hidden_kernel = MKMMDLoss(config["kernel_scales"]).to(
        device=device, dtype=next(student.parameters()).dtype)
    conditional_kernel = WeightedMKMMDLoss(config["kernel_scales"]).to(
        device=device, dtype=next(student.parameters()).dtype)
    profiler = StepProfiler(device, enabled=profile_steps > 0, warmup=min(10, profile_steps // 5))
    log_every = config.get("log_every_steps", 200)
    debug_every = config.get("kernel_debug_every_steps", 0)
    if log_every < 1 or debug_every < 0:
        raise ValueError("Invalid logging/debug interval")

    history = []
    for epoch in range(1, (1 if profile_steps else settings["epochs"]) + 1):
        student.train()
        source_batches = iter(source_loader)
        names = ("loss", "hidden", "normal", "attack", "rank", "source")
        totals = torch.zeros(len(names), device=device)
        steps = 0
        epoch_start = time.perf_counter()

        for target_x in profiler.iterate(target_loader, "read_target_batch"):
            try:
                source_x, source_y = next(source_batches)
            except StopIteration:
                source_batches = iter(source_loader)
                try:
                    source_x, source_y = next(source_batches)
                except StopIteration:
                    raise ValueError("Empty source loader") from None

            source_x, source_y, target_x = source_x.to(device), source_y.to(device), target_x.to(device)
            source_h, source_z = student.source_representations(source_x)
            terms = {k: torch.zeros((), device=device) for k in weights}

            if weights["source"] > 0:
                terms["source"] = F.cross_entropy(
                    student.classifier(source_z), source_y, weight=class_weights)

            target_h = student.adapter(target_x)
            if weights["hidden"] > 0:
                with profiler.measure("mk_forward_hidden"):
                    terms["hidden"], _ = hidden_kernel(
                        source_h, target_h,
                        validate=bool(debug_every and profiler.step % debug_every == 0))

            if weights["normal"] > 0 or weights["attack"] > 0:
                normal_x, normal_w, attack_x = sample_conditional_inputs(
                    hard_target_pools, middle_pool, n,
                    config["conditional_anchor_mass"], device)
                with student.balanced_adapter_batch():
                    both_z = student.shared_latent(student.adapter(torch.cat((normal_x, attack_x), dim=0)))
                normal_z, attack_z = both_z[:n], both_z[n:]
                validate = bool(debug_every and profiler.step % debug_every == 0)
                if weights["normal"] > 0:
                    with profiler.measure("weighted_mk_forward_normal"):
                        terms["normal"], _ = conditional_kernel(
                            sample_pool(source_pools[0], n, device), normal_z, normal_w,
                            validate=validate)
                if weights["attack"] > 0:
                    with profiler.measure("mk_forward_attack"):
                        terms["attack"], _ = hidden_kernel(
                            sample_pool(source_pools[1], n, device), attack_z,
                            validate=validate)

            if weights["rank"] > 0:
                target_logits = student.classifier(student.shared_latent(target_h))
                with torch.no_grad():
                    _, teacher_logits = teacher(target_x)
                terms["rank"] = ranking_loss(
                    teacher_logits[:, 1] - teacher_logits[:, 0],
                    target_logits[:, 1] - target_logits[:, 0])

            loss = sum(weights[k] * value for k, value in terms.items())
            if not torch.isfinite(loss):
                raise ValueError("V5i loss contains NaN/Inf")
            optimizer.zero_grad()
            with profiler.measure("backward"):
                loss.backward()
            with profiler.measure("optimizer"):
                optimizer.step()

            values = {**terms, "loss": loss}
            totals += torch.stack([values[name].detach() for name in names])
            steps += 1
            profiler.step += 1
            if steps % log_every == 0:
                print(f"Epoch {epoch} step {steps} | " + str(
                    dict(zip(names, (totals / steps).cpu().tolist()))), flush=True)
            if profile_steps and steps >= profile_steps:
                break

        if not steps:
            raise ValueError("Empty target loader")
        means = (totals / steps).cpu().tolist()
        if any(not math.isfinite(v) for v in means):
            raise ValueError("Nonfinite epoch statistics")
        row = {
            "epoch": epoch,
            **dict(zip(names, means)),
            "steps": steps,
            "seconds": time.perf_counter() - epoch_start,
        }
        if diagnostic is not None:
            row["fixed_hard_anchor_diagnostic"] = diagnostic_metrics(
                student, diagnostic, config["kernel_scales"])
        history.append(row)
        print(f"Epoch {epoch}/{settings['epochs']} | {row}", flush=True)

    if profile_result is not None:
        profile_result.update(
            timings=profiler.summary(), steps=profiler.step,
            warmup_steps=profiler.warmup,
            steps_per_second=history[-1]["steps"] / history[-1]["seconds"])
    student.eval()
    return history


def preflight(config_path, training_seed=None):
    from evaluation.hda_v5f import load_v5b_calibration
    from evaluation.hda_v5i import load_v5h_reference
    context = load_context(config_path, training_seed)
    config, protocol, provenance, *_ = context
    load_v5b_calibration(config, protocol, provenance)
    load_v5d_reference(config, protocol, provenance)
    load_v5h_reference(config, protocol)
    print("Stage 1 checkpoint: VERIFIED", flush=True)
    print("V5b calibration: VERIFIED", flush=True)
    print("V5d reference: VERIFIED", flush=True)
    print("V5h reference: VERIFIED", flush=True)
    for name in (
        "teacher_seed", "training_seed", "classifier_only", "calibration_max_fpr",
        "geometry_weight_power", "conditional_anchor_mass", "exclude_q98_plus",
    ):
        print(f"{name}: {config[name]}", flush=True)
    print(f"loss_weights: {config['loss_weights']}", flush=True)
    return context


def run(config_path, device_name="auto", training_seed=None, profile_steps=0, profile_output=None):
    if profile_steps < 0:
        raise ValueError("profile_steps must be nonnegative")
    startup = {}
    started = time.perf_counter()
    print("Startup: verifying frozen checkpoints, code and data hashes...", flush=True)
    config, protocol, provenance, source, v2, teacher = preflight(config_path, training_seed)
    startup["preflight_seconds"] = time.perf_counter() - started
    calibration_hash = file_hash(resolve_path(config["v5b_calibration"]))
    output = checkpoint_path(config)
    if output.exists() and not profile_steps:
        raise FileExistsError(f"Checkpoint already exists: {output}")
    profile_path = resolve_path(profile_output or f"results/hda_v5i/profiles/seed{config['training_seed']}.json")
    if profile_steps and profile_path.exists():
        raise FileExistsError(f"Profile already exists: {profile_path}; choose --profile-output")

    set_seed(config["training_seed"])
    code = code_hashes()
    settings = protocol["training"]
    batch_size = settings["batch_size"]
    source_path = ROOT / "data/features/unsw_train"
    target_path = resolve_path(protocol["target_data"]["adaptation_train"])
    snapshots = {
        "source": data_snapshot(source_path),
        "target": data_snapshot(target_path),
    }

    metadata = json.loads(resolve_path(config["source_metadata"]).read_text())
    if int(metadata["input_dim"]) != provenance["source_dim"]:
        raise ValueError("UNSW metadata dimension mismatch")
    counts = [int(metadata["class_counts"][str(k)]) for k in (0, 1)]
    class_weights = baseline_class_weights(counts)

    # Reuse the exact V5f hard-pool cache dependencies so existing caches hit.
    producer_files = (
        "src/training/hda_v5f.py", "src/training/v5e_performance.py",
        "src/evaluation/hda_v5b_calibration.py", "src/training/hda_v4.py",
        "src/training/v6_data.py", "src/models/hda_v1.py", "src/models/baseline.py",
    )
    common = {
        "schema": 1,
        "torch": str(torch.__version__),
        "batch_size": batch_size,
        "producer_code": {name: file_hash(ROOT / name) for name in producer_files},
    }
    base_cache_dir = resolve_path(config["pool_cache_dir"])

    started = time.perf_counter()
    print("Startup: frozen V2 hard anchors...", flush=True)
    def build_hard_target():
        normal, attack, meta = build_frozen_target_pools(
            v2, protocol, provenance["target_dim"], batch_size)
        if data_snapshot(target_path) != snapshots["target"]:
            raise ValueError("Target data changed while building hard-anchor cache")
        return {"normal": normal, "attack": attack, "metadata": meta}
    hard_cache = cached_pools(base_cache_dir, {
        **common, "kind": "target_v2_quantiles", "backend": "cpu",
        "data": snapshots["target"], "target_dim": provenance["target_dim"],
        "v2_checkpoint": provenance["teacher_checkpoint_sha256"],
        "source_checkpoint": provenance["source_checkpoint_sha256"],
        "policy": protocol["pseudo_labels"],
    }, build_hard_target)
    hard_target_pools = {0: hard_cache["normal"], 1: hard_cache["attack"]}
    pseudo = hard_cache["metadata"]
    startup["hard_target_seconds"] = time.perf_counter() - started

    device = torch.device(("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
                          if device_name == "auto" else device_name)
    student = HDAV5DModel(source, teacher.adapter, classifier_only=False).to(device)
    source.to(device)
    teacher.to(device)

    started = time.perf_counter()
    print("Startup: source latent pools...", flush=True)
    def build_source():
        pools = build_source_pools(
            source, make_loader(source_path, provenance["source_dim"], batch_size), device)
        if data_snapshot(source_path) != snapshots["source"]:
            raise ValueError("Source data changed while building cache")
        return pools
    source_pools = cached_pools(base_cache_dir, {
        **common, "kind": "source_latents", "backend": str(device),
        "data": snapshots["source"], "source_dim": provenance["source_dim"],
        "source_checkpoint": provenance["source_checkpoint_sha256"],
    }, build_source)
    startup["source_pools_seconds"] = time.perf_counter() - started

    sn = source_pools[0].double().mean(0)
    sa = source_pools[1].double().mean(0)
    source_centroid_distance = torch.linalg.vector_norm(sn - sa).item()

    started = time.perf_counter()
    print("Startup: frozen V2 middle geometry weights (CPU; cached after first build)...", flush=True)
    geometry_cache_dir = resolve_path(config["geometry_cache_dir"])
    geometry_dependencies = {
        "schema": 1,
        "kind": "v5i_v2_middle_geometry_weights",
        "torch": str(torch.__version__),
        "batch_size": batch_size,
        "target_data": snapshots["target"],
        "target_dim": provenance["target_dim"],
        "source_data": snapshots["source"],
        "source_checkpoint": provenance["source_checkpoint_sha256"],
        "source_pool_backend": str(device),
        "source_centroids_sha256": tensor_sha256(sn, sa),
        "v2_checkpoint": provenance["teacher_checkpoint_sha256"],
        "q02": float(pseudo["q02"]),
        "q95": float(pseudo["q95"]),
        "q98": float(pseudo["q98"]),
        "geometry_weight_power": float(config["geometry_weight_power"]),
        "exclude_q98_plus": True,
        "producer_code": {
            "src/training/hda_v5i.py": file_hash(ROOT / "src/training/hda_v5i.py"),
            "src/training/v6_data.py": file_hash(ROOT / "src/training/v6_data.py"),
            "src/models/hda_v1.py": file_hash(ROOT / "src/models/hda_v1.py"),
        },
    }
    def build_middle():
        result = build_geometry_middle_pool(
            v2, target_path, provenance["target_dim"], batch_size,
            sn, sa, float(pseudo["q02"]), float(pseudo["q95"]),
            float(config["geometry_weight_power"]))
        if data_snapshot(target_path) != snapshots["target"]:
            raise ValueError("Target data changed while building geometry cache")
        return result
    middle_pool = cached_pools(geometry_cache_dir, geometry_dependencies, build_middle)
    if middle_pool["metadata"]["rows_scored"] != pseudo["train_rows"]:
        raise ValueError("Geometry pool did not score every adaptation-train row")
    startup["middle_geometry_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    print("Startup: fixed hard-anchor diagnostics...", flush=True)
    diagnostic = make_diagnostic(
        student, source_path, target_path, provenance, source_pools,
        hard_target_pools, batch_size, settings["class_batch_size"])
    diagnostic_initial = diagnostic_metrics(student, diagnostic, config["kernel_scales"])
    v5d_checkpoint = torch.load(
        resolve_path(config["v5d_reference_checkpoint"]), map_location="cpu", weights_only=True)
    v5d = HDAV5DModel(source, teacher.adapter).to(device)
    v5d.adapter.load_state_dict(v5d_checkpoint["target_adapter_state_dict"])
    v5d.classifier.load_state_dict(v5d_checkpoint["classifier_state_dict"])
    diagnostic_v5d = diagnostic_metrics(v5d, diagnostic, config["kernel_scales"])
    del v5d, v5d_checkpoint
    startup["diagnostic_seconds"] = time.perf_counter() - started

    print(f"Startup timings: {startup}", flush=True)
    print(f"V5i | device={device} | source weights={class_weights.tolist()}", flush=True)
    print(f"V5i conditional policy | anchor_mass={config['conditional_anchor_mass']} | "
          f"normal_middle_power={config['geometry_weight_power']} | Attack hard-only | q98+ excluded", flush=True)
    print(f"Middle geometry metadata: {middle_pool['metadata']}", flush=True)

    profile = {}
    history = train_model(
        student, teacher,
        make_loader(source_path, provenance["source_dim"], batch_size, training=True),
        make_unlabeled_loader(target_path, provenance["target_dim"], batch_size),
        source_pools, hard_target_pools, middle_pool, class_weights,
        config, settings, diagnostic=diagnostic,
        profile_steps=profile_steps, profile_result=profile)

    if profile_steps:
        profile.update(
            startup=startup, device=str(device), torch=str(torch.__version__),
            code_sha256=code, training_seed=config["training_seed"],
            conditional_policy={
                "anchor_mass": config["conditional_anchor_mass"],
                "geometry_weight_power": config["geometry_weight_power"],
                "q98_plus_excluded": True,
            },
            note="Disposable V5i profile run; no checkpoint saved.",
            history=history,
        )
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with profile_path.open("x") as stream:
            json.dump(profile, stream, indent=2, allow_nan=False)
        print(f"Profile saved (no checkpoint): {profile_path}")
        return

    if (snapshots != {
            "source": data_snapshot(source_path),
            "target": data_snapshot(target_path)} or code != code_hashes()):
        raise ValueError("Training data or code changed during training")
    _, _, current_provenance, *_ = load_context(config_path, training_seed)
    if (current_provenance != provenance
            or file_hash(resolve_path(config["v5b_calibration"])) != calibration_hash
            or file_hash(resolve_path(config["v5h_reference_report"])) != config["v5h_reference_sha256"]):
        raise ValueError("Frozen inputs changed during training")
    load_v5d_reference(config, protocol, provenance)

    checkpoint = {
        "version": "v5i",
        "architecture": "hda_v5i",
        "training_seed": config["training_seed"],
        "teacher_seed": config["teacher_seed"],
        "classifier_only": False,
        "provenance": provenance,
        "code_sha256": code,
        "training_data": snapshots,
        "preflight_v5b_calibration_sha256": calibration_hash,
        "loss_weights": config["loss_weights"],
        "training": settings,
        "optimizer": {k: config[k] for k in ("adapter_lr", "classifier_lr", "weight_decay")},
        "source_class_counts": counts,
        "source_class_weights": class_weights.tolist(),
        "pseudo_metadata": pseudo,
        "target_labels_used": False,
        "checkpoint_selection": "last epoch",
        "teacher": "frozen_v5b_asymmetric",
        "history": history,
        "startup_timings": startup,
        "kernel_scales": config["kernel_scales"],
        "kernel_aggregation": "mean",
        "conditional_alignment": {
            "method": "asymmetric_selective_mk_mmd",
            "anchor_mass": config["conditional_anchor_mass"],
            "middle_mass": 1.0 - config["conditional_anchor_mass"],
            "mass_scope": "Normal branch only",
            "normal_branch": "weighted MK-MMD: hard q02 + middle weighted by (1-r)^2",
            "attack_branch": "ordinary MK-MMD: hard q95-q98 only; no middle",
            "geometry_weight_power": config["geometry_weight_power"],
            "middle_definition": "q02 < frozen V2 margin < q95",
            "normal_anchor": "V2 margin <= q02",
            "attack_anchor": "q95 <= V2 margin < q98",
            "q98_plus": "excluded",
            "geometry": "V2 latent distance-to-source-centroid difference, percentile-ranked in middle",
            "source_centroid_distance": source_centroid_distance,
            "middle_metadata": middle_pool["metadata"],
        },
        "fixed_diagnostic_batches": diagnostic,
        "fixed_diagnostic_v5b_initial": diagnostic_initial,
        "fixed_diagnostic_v5d_reference": diagnostic_v5d,
        "target_adapter_state_dict": {
            k: v.detach().cpu() for k, v in student.adapter.state_dict().items()},
        "classifier_state_dict": {
            k: v.detach().cpu() for k, v in student.classifier.state_dict().items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(checkpoint, stream)
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5i.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--training-seed", type=int, choices=(42, 43, 44))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=0,
                        help="Disposable short training profile; no checkpoint saved")
    parser.add_argument("--profile-output")
    args = parser.parse_args()
    if args.preflight_only:
        preflight(args.config, args.training_seed)
    else:
        run(args.config, args.device, args.training_seed, args.profile_steps, args.profile_output)


if __name__ == "__main__":
    main()
