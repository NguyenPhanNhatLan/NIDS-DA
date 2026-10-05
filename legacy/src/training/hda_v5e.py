"""V5e: V5b alignment + ranking + weighted source CE on a private classifier."""
import argparse
import json
import math
import time

import torch
from torch.nn import functional as F

from evaluation.hda_v5b_calibration import (
    build_frozen_target_pools, calibration_code_hashes, data_snapshot, load_frozen_models,
)
from models.hda_v5d import HDAV5DModel
from training.adaptation import estimate_bandwidth_squared
from training.mkmmd import mk_mmd_loss, MKMMDLoss
from training.v5e_performance import cached_pools, StepProfiler, legacy_mk_mmd_loss
from training.baseline import set_seed
from training.hda_v4 import build_source_pools, sample_pool
from training.hda_v5b import ROOT, file_hash, ranking_loss
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader, make_unlabeled_loader, make_teacher_loader


def code_hashes():
    hashes = calibration_code_hashes()
    for name in ("src/models/hda_v5d.py", "src/training/hda_v5e.py",
                 "src/evaluation/hda_v5e.py", "src/training/mkmmd.py", "src/training/v5e_performance.py", "src/training/adaptation.py",
                 "src/training/baseline.py", "src/training/thesis_protocol.py",
                 "src/evaluation/protocol_revision.py"):
        hashes[name] = file_hash(ROOT / name)
    return hashes


def load_context(config_path, training_seed=None):
    path = resolve_path(config_path)
    config = json.loads(path.read_text())
    if (not config["kernel_scales"] or any(not math.isfinite(v) or v <= 0 for v in config["kernel_scales"])):
        raise ValueError("kernel_scales must contain positive finite multipliers of bandwidth squared")
    if (config["classifier_only"] or config["loss_weights"] !=
            {"hidden": 1., "normal": .05, "attack": .02, "rank": .10, "source": .10}
            or config["adapter_lr"] != 1e-4 or config["classifier_lr"] != 1e-5
            or config["weight_decay"] != 1e-4):
        raise ValueError("V5e-1 must preserve V5d joint loss weights and optimizer settings")
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
        raise ValueError("Expected all five V5e loss weights")
    if any(not math.isfinite(v) or v < 0 for v in config["loss_weights"].values()):
        raise ValueError("Loss weights must be nonnegative and finite")
    if not 0 <= config["calibration_max_fpr"] <= 1:
        raise ValueError("calibration_max_fpr must be in [0, 1]")
    reference_path = resolve_path(config["stage1_reference"])
    reference, protocol, provenance, source, v2, teacher = load_frozen_models(reference_path)
    if config["teacher_seed"] != 42 or reference["seed"] != config["teacher_seed"]:
        raise ValueError("V5e requires frozen V5b asymmetric seed 42")
    if reference["loss_weights"] != {"hidden": 1.0, "normal": 0.05, "attack": 0.02, "rank": 0.10}:
        raise ValueError("Expected asymmetric V5b teacher")
    if resolve_path(protocol["target_data"]["adaptation_train"]).resolve() == evaluation_target(protocol, "development").resolve():
        raise ValueError("Adaptation and development must be separate")
    if resolve_path(config["source_validation"]).resolve() != (ROOT / "data/features/unsw_val").resolve():
        raise ValueError("Use the fixed UNSW validation split")
    if resolve_path(config["source_metadata"]).resolve() != (ROOT / "data/features/unsw_metadata.json").resolve():
        raise ValueError("Use the same UNSW class-count metadata as baseline")
    provenance = {**provenance, "config_sha256": file_hash(path),
                  "training_seed": config["training_seed"], "teacher_seed": config["teacher_seed"],
                  "classifier_only": config["classifier_only"],
                  "stage1_reference_sha256": file_hash(reference_path),
                  "stage1_checkpoint_sha256": reference["checkpoint_sha256"],
                  "source_metadata_sha256": file_hash(resolve_path(config["source_metadata"]))}
    return config, protocol, provenance, source, v2, teacher


def baseline_class_weights(counts):
    counts = torch.as_tensor(counts, dtype=torch.float32)
    if counts.shape != (2,) or not torch.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("Source counts must contain two positive finite counts")
    return counts.sum() / (2 * counts)


def checkpoint_path(config):
    return resolve_path(config["checkpoint_dir"]) / f"v5e_seed{config['training_seed']}.pt"


def load_student(config, provenance, source, teacher):
    checkpoint = torch.load(checkpoint_path(config), map_location="cpu", weights_only=True)
    if (checkpoint.get("version") != "v5e" or checkpoint.get("architecture") != "hda_v5e"
            or checkpoint["provenance"] != provenance or checkpoint["code_sha256"] != code_hashes()
            or checkpoint["training_seed"] != config["training_seed"]
            or checkpoint["teacher_seed"] != config["teacher_seed"]
            or checkpoint["classifier_only"] != config["classifier_only"]):
        raise ValueError("V5e checkpoint provenance mismatch")
    model = HDAV5DModel(source, teacher.adapter, classifier_only=config["classifier_only"])
    model.adapter.load_state_dict(checkpoint["target_adapter_state_dict"])
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    model.eval().requires_grad_(False)
    return model, checkpoint


def train_model(student, teacher, source_loader, target_loader, source_pools,
                target_pools, class_weights, config, settings, diagnostic=None,
                profile_steps=0, profile_kernel="optimized", profile_result=None):
    device = next(student.parameters()).device
    teacher.eval().requires_grad_(False)
    weights = config["loss_weights"]
    optimizer = torch.optim.Adam(
        student.optimizer_groups(config["adapter_lr"], config["classifier_lr"]),
        weight_decay=config["weight_decay"])
    class_weights = class_weights.to(device)
    n = settings["class_batch_size"]
    kernel = MKMMDLoss(config["kernel_scales"]).to(device=device, dtype=next(student.parameters()).dtype)
    if not (torch.isfinite(kernel.scale_tensor).all() & (kernel.scale_tensor > 0).all()):
        raise ValueError("Kernel scales are not representable in the training dtype")
    profiler = StepProfiler(device, enabled=profile_steps > 0, warmup=min(10, profile_steps // 5))
    log_every = config.get("log_every_steps", 200)
    debug_every = config.get("kernel_debug_every_steps", 0)
    if log_every < 1 or debug_every < 0:
        raise ValueError("Invalid logging/debug interval")
    def compute_mmd(s, t, name):
        with profiler.measure("mk_forward_" + name):
            if profile_kernel == "legacy":
                return legacy_mk_mmd_loss(s, t, config["kernel_scales"])
            return kernel(s, t, validate=bool(debug_every and profiler.step % debug_every == 0))
    history = []
    for epoch in range(1, (1 if profile_steps else settings["epochs"]) + 1):
        student.train()
        source_batches = iter(source_loader)
        names = ("loss", "hidden", "normal", "attack", "rank", "source")
        totals = torch.zeros(len(names), device=device)
        steps = 0
        epoch_start = time.perf_counter()
        for target_x in profiler.iterate(target_loader, "read_target_batch"):
            read_start = time.perf_counter()
            try:
                source_x, source_y = next(source_batches)
            except StopIteration:
                source_batches = iter(source_loader)
                try:
                    source_x, source_y = next(source_batches)
                except StopIteration:
                    raise ValueError("Empty source loader") from None
            if profiler.enabled and profiler.step >= profiler.warmup:
                profiler.records.setdefault("read_source_batch", []).append(time.perf_counter() - read_start)
            source_x, source_y, target_x = source_x.to(device), source_y.to(device), target_x.to(device)
            source_h, source_z = student.source_representations(source_x)
            terms = {k: torch.zeros((), device=device) for k in weights}
            if weights["source"] > 0:
                terms["source"] = F.cross_entropy(student.classifier(source_z), source_y, weight=class_weights)
            # Joint training retains one natural-batch BN update. A frozen adapter
            # needs no forward at all when both hidden MMD and ranking are disabled.
            if not student.classifier_only or weights["hidden"] > 0 or weights["rank"] > 0:
                target_h = student.adapter(target_x)
            if weights["hidden"] > 0:
                terms["hidden"], _ = compute_mmd(source_h, target_h, "hidden")
            if weights["normal"] > 0 or weights["attack"] > 0:
                balanced_x = torch.cat([sample_pool(target_pools[k], n, device) for k in (0, 1)])
                with student.balanced_adapter_batch():
                    balanced_z = student.shared_latent(student.adapter(balanced_x))
                if weights["normal"] > 0:
                    terms["normal"], _ = compute_mmd(sample_pool(source_pools[0], n, device), balanced_z[:n], "normal")
                if weights["attack"] > 0:
                    terms["attack"], _ = compute_mmd(sample_pool(source_pools[1], n, device), balanced_z[n:], "attack")
            if weights["rank"] > 0:
                target_logits = student.classifier(student.shared_latent(target_h))
                with torch.no_grad():
                    _, teacher_logits = teacher(target_x)
                terms["rank"] = ranking_loss(teacher_logits[:, 1] - teacher_logits[:, 0],
                                             target_logits[:, 1] - target_logits[:, 0])
            loss = sum(weights[k] * value for k, value in terms.items())
            if not torch.isfinite(loss):
                raise ValueError("V5e loss contains NaN/Inf")
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
                print(f"Epoch {epoch} step {steps} | " + str(dict(zip(names, (totals / steps).cpu().tolist()))), flush=True)
            if profile_steps and steps >= profile_steps:
                break
        if not steps:
            raise ValueError("Empty target loader")
        means = (totals / steps).cpu().tolist()
        if any(not math.isfinite(v) for v in means):
            raise ValueError("Nonfinite epoch statistics")
        row = {"epoch": epoch, **dict(zip(names, means)), "steps": steps,
               "seconds": time.perf_counter() - epoch_start}
        if diagnostic is not None:
            row["fixed_batch_diagnostic"] = diagnostic_metrics(student, diagnostic, config["kernel_scales"])
        history.append(row)
        print(f"Epoch {epoch}/{settings['epochs']} | {row}", flush=True)
    if profile_result is not None:
        profile_result.update(timings=profiler.summary(), steps=profiler.step,
                              warmup_steps=profiler.warmup, kernel=profile_kernel,
                              steps_per_second=history[-1]["steps"] / history[-1]["seconds"])
    student.eval()
    return history


def load_v5d_reference(config, protocol, provenance):
    path = resolve_path(config["v5d_reference_report"])
    if file_hash(path) != config["v5d_reference_sha256"]:
        raise ValueError("Pinned V5d reference report changed")
    report = json.loads(path.read_text())
    ref = report["dependencies"]["provenance"]
    for key in ("source_checkpoint_sha256", "teacher_checkpoint_sha256", "stage1_checkpoint_sha256"):
        if ref[key] != provenance[key]:
            raise ValueError(f"V5d reference dependency differs: {key}")
    if (report["classifier_only"] or report["training_seed"] != config["training_seed"]
            or report["phase"] != "development" or report["max_source_fpr"] != config["calibration_max_fpr"]
            or report["v5b_calibration_payload_sha256"] != config["v5b_calibration_payload_sha256"]
            or resolve_path(report["target_data"]).resolve() != evaluation_target(protocol, "development").resolve()):
        raise ValueError("V5d reference split/seed/threshold policy mismatch")
    if file_hash(resolve_path(config["v5d_reference_checkpoint"])) != report["dependencies"]["v5d_checkpoint_sha256"]:
        raise ValueError("Pinned V5d checkpoint changed")
    checkpoint = torch.load(resolve_path(config["v5d_reference_checkpoint"]), map_location="cpu", weights_only=True)
    if (checkpoint["loss_weights"] != config["loss_weights"]
            or checkpoint["optimizer"] != {key: config[key] for key in ("adapter_lr", "classifier_lr", "weight_decay")}
            or checkpoint["training"] != protocol["training"]
            or checkpoint["classifier_only"] or checkpoint["checkpoint_selection"] != "last epoch"):
        raise ValueError("V5e-1 must match V5d supervision, optimizer, training schedule and checkpoint selection")
    if checkpoint["training_data"] != {
            "source": data_snapshot(ROOT / "data/features/unsw_train"),
            "target": data_snapshot(resolve_path(protocol["target_data"]["adaptation_train"]))}:
        raise ValueError("V5d/V5e training datasets differ")
    return report


def diagnostic_pairs(model, batch):
    device = next(model.parameters()).device
    target_h = model.adapter(batch["target"].to(device))
    return {
        "hidden": (batch["source_hidden"].to(device), target_h),
        "normal": (batch["source_normal"].to(device), model.shared_latent(model.adapter(batch["target_normal"].to(device)))),
        "attack": (batch["source_attack"].to(device), model.shared_latent(model.adapter(batch["target_attack"].to(device)))),
    }


def diagnostic_metrics(model, batch, scales):
    # Restore every mode, preserve buffers, and consume no random samples.
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            result = {}
            for name, (s, t) in diagnostic_pairs(model, batch).items():
                bandwidth = batch["bandwidth_squared"][name]
                single, _ = mk_mmd_loss(s, t, scales=(1.,), bandwidth_squared=bandwidth)
                multi, _ = mk_mmd_loss(s, t, scales=scales, bandwidth_squared=bandwidth)
                result[name] = {"single_rbf": single.item(), "mk_rbf": multi.item(),
                                "fixed_bandwidth_squared": bandwidth}
            return result
    finally:
        for module, mode in modes:
            module.training = mode


def make_diagnostic(student, source_path, target_path, provenance, source_pools, target_pools, batch_size, n):
    # Dedicated first batches; restore CPU RNG consumed by deterministic loaders.
    with torch.random.fork_rng(devices=[]):
        source_x, _ = next(iter(make_loader(source_path, provenance["source_dim"], batch_size)))
        target_x = next(iter(make_teacher_loader(target_path, provenance["target_dim"], batch_size)))
    device = next(student.parameters()).device
    with torch.no_grad():
        source_h, _ = student.source_representations(source_x.to(device))
    batch = {"source_hidden": source_h.cpu(), "target": target_x,
             "source_normal": source_pools[0][:n].cpu(), "source_attack": source_pools[1][:n].cpu(),
             "target_normal": target_pools[0][:n].cpu(), "target_attack": target_pools[1][:n].cpu()}
    modes = [(module, module.training) for module in student.modules()]
    student.eval()
    try:
        with torch.no_grad():
            batch["bandwidth_squared"] = {name: estimate_bandwidth_squared(s, t).item()
                                          for name, (s, t) in diagnostic_pairs(student, batch).items()}
    finally:
        for module, mode in modes:
            module.training = mode
    return batch


def preflight(config_path, training_seed=None):
    # Local import avoids a module-level cycle with the evaluation entry point.
    from evaluation.hda_v5e import load_v5b_calibration

    context = load_context(config_path, training_seed)
    config, protocol, provenance, *_ = context
    load_v5b_calibration(config, protocol, provenance)
    load_v5d_reference(config, protocol, provenance)
    print("Stage 1 checkpoint: VERIFIED", flush=True)
    print("V5b calibration: VERIFIED", flush=True)
    for name in ("teacher_seed", "training_seed", "classifier_only", "calibration_max_fpr"):
        print(f"{name}: {config[name]}", flush=True)
    return context


def run(config_path, device_name="auto", training_seed=None, profile_steps=0,
        profile_kernel="optimized", profile_output=None):
    if profile_steps < 0 or (profile_kernel == "legacy" and not profile_steps):
        raise ValueError("Legacy kernel is available only with positive --profile-steps")
    startup = {}
    started = time.perf_counter()
    print("Startup: verifying frozen checkpoints, code and data hashes...", flush=True)
    config, protocol, provenance, source, v2, teacher = preflight(config_path, training_seed)
    startup["preflight_seconds"] = time.perf_counter() - started
    calibration_hash = file_hash(resolve_path(config["v5b_calibration"]))
    output = checkpoint_path(config)
    if output.exists() and not profile_steps:
        raise FileExistsError(f"Checkpoint already exists: {output}")
    profile_path = resolve_path(profile_output or f"results/hda_v5e/profiles/{profile_kernel}_seed{config['training_seed']}.json")
    if profile_steps and profile_path.exists():
        raise FileExistsError(f"Profile already exists: {profile_path}; choose --profile-output")
    set_seed(config["training_seed"])
    code = code_hashes()
    settings = protocol["training"]
    batch_size = settings["batch_size"]
    source_path = ROOT / "data/features/unsw_train"
    target_path = resolve_path(protocol["target_data"]["adaptation_train"])
    started = time.perf_counter()
    print("Startup: snapshot training Parquet files...", flush=True)
    snapshots = {"source": data_snapshot(source_path), "target": data_snapshot(target_path)}
    startup["snapshot_seconds"] = time.perf_counter() - started
    metadata = json.loads(resolve_path(config["source_metadata"]).read_text())
    if int(metadata["input_dim"]) != provenance["source_dim"]:
        raise ValueError("UNSW metadata dimension mismatch")
    counts = [int(metadata["class_counts"][str(k)]) for k in (0, 1)]
    class_weights = baseline_class_weights(counts)
    needs_conditional = any(config["loss_weights"][k] > 0 for k in ("normal", "attack"))
    target_pools = None
    pseudo = {"used_for_training": False}
    producer_files = ("src/training/hda_v5e.py", "src/training/v5e_performance.py", "src/evaluation/hda_v5b_calibration.py",
                      "src/training/hda_v4.py", "src/training/v6_data.py",
                      "src/models/hda_v1.py", "src/models/baseline.py")
    common_dependencies = {"schema": 1, "torch": str(torch.__version__),
                           "batch_size": batch_size,
                           "producer_code": {name: file_hash(ROOT / name) for name in producer_files}}
    cache_dir = resolve_path(config["pool_cache_dir"])
    started = time.perf_counter()
    print("Startup: target pseudo pools (CPU teacher; cached after first build)...", flush=True)
    if needs_conditional:
        def build_target():
            normal, attack, metadata = build_frozen_target_pools(v2, protocol, provenance["target_dim"], batch_size)
            if data_snapshot(target_path) != snapshots["target"]:
                raise ValueError("Target data changed while building cache")
            return {"normal": normal, "attack": attack, "metadata": metadata}
        target_cache = cached_pools(cache_dir, {
            **common_dependencies, "kind": "target_v2_quantiles", "backend": "cpu",
            "data": snapshots["target"], "target_dim": provenance["target_dim"],
            "v2_checkpoint": provenance["teacher_checkpoint_sha256"],
            "source_checkpoint": provenance["source_checkpoint_sha256"],
            "policy": protocol["pseudo_labels"],
        }, build_target)
        target_pools = {0: target_cache["normal"], 1: target_cache["attack"]}
        pseudo = target_cache["metadata"]
    startup["target_pools_seconds"] = time.perf_counter() - started
    device = torch.device(("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
                          if device_name == "auto" else device_name)
    student = HDAV5DModel(source, teacher.adapter, classifier_only=config["classifier_only"]).to(device)
    source.to(device)
    teacher.to(device)
    source_pools = None
    started = time.perf_counter()
    print("Startup: source latent pools...", flush=True)
    if needs_conditional:
        def build_source():
            pools = build_source_pools(source, make_loader(source_path, provenance["source_dim"], batch_size), device)
            if data_snapshot(source_path) != snapshots["source"]:
                raise ValueError("Source data changed while building cache")
            return pools
        source_pools = cached_pools(cache_dir, {
            **common_dependencies, "kind": "source_latents", "backend": str(device),
            "data": snapshots["source"], "source_dim": provenance["source_dim"],
            "source_checkpoint": provenance["source_checkpoint_sha256"],
        }, build_source)
    startup["source_pools_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    print("Startup: fixed diagnostics...", flush=True)
    diagnostic = make_diagnostic(student, source_path, target_path, provenance, source_pools,
                                 target_pools, batch_size, settings["class_batch_size"])
    diagnostic_initial = diagnostic_metrics(student, diagnostic, config["kernel_scales"])
    # The V5d model is used ONLY for fixed-batch diagnostics, never initialization or loss.
    v5d_checkpoint = torch.load(resolve_path(config["v5d_reference_checkpoint"]), map_location="cpu", weights_only=True)
    v5d = HDAV5DModel(source, teacher.adapter).to(device)
    v5d.adapter.load_state_dict(v5d_checkpoint["target_adapter_state_dict"])
    v5d.classifier.load_state_dict(v5d_checkpoint["classifier_state_dict"])
    diagnostic_v5d = diagnostic_metrics(v5d, diagnostic, config["kernel_scales"])
    print(f"Fixed diagnostic V5b initialization: {diagnostic_initial}", flush=True)
    print(f"Fixed diagnostic V5d reference: {diagnostic_v5d}", flush=True)
    del v5d, v5d_checkpoint
    startup["diagnostic_seconds"] = time.perf_counter() - started
    print(f"Startup timings: {startup}", flush=True)
    print(f"V5e | device={device} | source weights={class_weights.tolist()}", flush=True)
    profile = {}
    history = train_model(
        student, teacher, make_loader(source_path, provenance["source_dim"], batch_size, training=True),
        make_unlabeled_loader(target_path, provenance["target_dim"], batch_size),
        source_pools, target_pools, class_weights, config, settings, diagnostic=diagnostic,
        profile_steps=profile_steps, profile_kernel=profile_kernel, profile_result=profile)
    if profile_steps:
        profile.update(startup=startup, device=str(device), torch=str(torch.__version__),
                       code_sha256=code, training_seed=config["training_seed"],
                       note="Disposable synchronized profile run; no trained checkpoint saved. Throughput includes warmup.",
                       history=history)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with profile_path.open("x") as stream:
            json.dump(profile, stream, indent=2, allow_nan=False)
        print(f"Profile saved (no checkpoint): {profile_path}")
        return
    if (snapshots != {"source": data_snapshot(source_path), "target": data_snapshot(target_path)}
            or code != code_hashes()):
        raise ValueError("Training data or code changed during training")
    # Also revalidate frozen checkpoints and configuration before publishing output.
    _, _, current_provenance, *_ = load_context(config_path, training_seed)
    if (current_provenance != provenance
            or file_hash(resolve_path(config["v5b_calibration"])) != calibration_hash):
        raise ValueError("Frozen inputs changed during training")
    load_v5d_reference(config, protocol, provenance)
    checkpoint = {
        "version": "v5e", "architecture": "hda_v5e",
        "training_seed": config["training_seed"], "teacher_seed": config["teacher_seed"],
        "classifier_only": config["classifier_only"],
        "provenance": provenance, "code_sha256": code, "training_data": snapshots,
        "preflight_v5b_calibration_sha256": calibration_hash,
        "loss_weights": config["loss_weights"], "training": settings,
        "optimizer": {k: config[k] for k in ("adapter_lr", "classifier_lr", "weight_decay")},
        "source_class_counts": counts, "source_class_weights": class_weights.tolist(),
        "pseudo_metadata": pseudo, "target_labels_used": False,
        "checkpoint_selection": "last epoch", "teacher": "frozen_v5b_asymmetric",
        "history": history,
        "startup_timings": startup,
        "kernel_scales": config["kernel_scales"], "kernel_aggregation": "mean",
        "fixed_diagnostic_batches": diagnostic,
        "fixed_diagnostic_v5b_initial": diagnostic_initial,
        "fixed_diagnostic_v5d_reference": diagnostic_v5d,
        "target_adapter_state_dict": {k: v.detach().cpu() for k, v in student.adapter.state_dict().items()},
        "classifier_state_dict": {k: v.detach().cpu() for k, v in student.classifier.state_dict().items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(checkpoint, stream)
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5e.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--training-seed", type=int, choices=(42, 43, 44))
    parser.add_argument("--preflight-only", action="store_true",
                        help="Verify frozen Stage 1 and calibration, then exit without training")
    parser.add_argument("--profile-steps", type=int, default=0,
                        help="Run a disposable short training profile; never save a checkpoint")
    parser.add_argument("--profile-kernel", choices=("optimized", "legacy"), default="optimized")
    parser.add_argument("--profile-output")
    args = parser.parse_args()
    if args.preflight_only:
        preflight(args.config, args.training_seed)
    else:
        run(args.config, args.device, args.training_seed, args.profile_steps, args.profile_kernel, args.profile_output)


if __name__ == "__main__":
    main()
