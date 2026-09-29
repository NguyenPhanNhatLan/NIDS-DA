"""V5d: V5b alignment + ranking + weighted source CE on a private classifier."""
import argparse
import json
import math

import torch
from torch.nn import functional as F

from evaluation.hda_v5b_calibration import (
    build_frozen_target_pools, calibration_code_hashes, data_snapshot, load_frozen_models,
)
from models.hda_v5d import HDAV5DModel
from training.adaptation import mmd_loss
from training.baseline import set_seed
from training.hda_v4 import build_source_pools, sample_pool
from training.hda_v5b import ROOT, file_hash, ranking_loss
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader, make_unlabeled_loader


def code_hashes():
    hashes = calibration_code_hashes()
    for name in ("src/models/hda_v5d.py", "src/training/hda_v5d.py",
                 "src/evaluation/hda_v5d.py", "src/training/adaptation.py",
                 "src/training/baseline.py", "src/training/thesis_protocol.py",
                 "src/evaluation/protocol_revision.py"):
        hashes[name] = file_hash(ROOT / name)
    return hashes


def load_context(config_path, training_seed=None):
    path = resolve_path(config_path)
    config = json.loads(path.read_text())
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
        raise ValueError("Expected all five V5d loss weights")
    if any(not math.isfinite(v) or v < 0 for v in config["loss_weights"].values()):
        raise ValueError("Loss weights must be nonnegative and finite")
    if not 0 <= config["calibration_max_fpr"] <= 1:
        raise ValueError("calibration_max_fpr must be in [0, 1]")
    reference_path = resolve_path(config["stage1_reference"])
    reference, protocol, provenance, source, v2, teacher = load_frozen_models(reference_path)
    if config["teacher_seed"] != 42 or reference["seed"] != config["teacher_seed"]:
        raise ValueError("V5d requires frozen V5b asymmetric seed 42")
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
    return resolve_path(config["checkpoint_dir"]) / f"v5d_seed{config['training_seed']}.pt"


def load_student(config, provenance, source, teacher):
    checkpoint = torch.load(checkpoint_path(config), map_location="cpu", weights_only=True)
    if (checkpoint.get("version") != "v5d" or checkpoint.get("architecture") != "hda_v5d"
            or checkpoint["provenance"] != provenance or checkpoint["code_sha256"] != code_hashes()
            or checkpoint["training_seed"] != config["training_seed"]
            or checkpoint["teacher_seed"] != config["teacher_seed"]
            or checkpoint["classifier_only"] != config["classifier_only"]):
        raise ValueError("V5d checkpoint provenance mismatch")
    model = HDAV5DModel(source, teacher.adapter, classifier_only=config["classifier_only"])
    model.adapter.load_state_dict(checkpoint["target_adapter_state_dict"])
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    model.eval().requires_grad_(False)
    return model, checkpoint


def train_model(student, teacher, source_loader, target_loader, source_pools,
                target_pools, class_weights, config, settings):
    device = next(student.parameters()).device
    teacher.eval().requires_grad_(False)
    weights = config["loss_weights"]
    optimizer = torch.optim.Adam(
        student.optimizer_groups(config["adapter_lr"], config["classifier_lr"]),
        weight_decay=config["weight_decay"])
    class_weights = class_weights.to(device)
    n = settings["class_batch_size"]
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        student.train()
        source_batches = iter(source_loader)
        totals = dict.fromkeys(("loss", "hidden", "normal", "attack", "rank", "source"), 0.0)
        steps = 0
        for target_x in target_loader:
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
            target_h = student.adapter(target_x)
            target_z = student.shared_latent(target_h)
            source_logits = student.classifier(source_z)
            target_logits = student.classifier(target_z)
            source_ce = F.cross_entropy(source_logits, source_y, weight=class_weights)
            hidden, _ = mmd_loss(source_h, target_h)
            balanced_x = torch.cat([sample_pool(target_pools[k], n, device) for k in (0, 1)])
            with student.balanced_adapter_batch():
                balanced_z = student.shared_latent(student.adapter(balanced_x))
            normal, _ = mmd_loss(sample_pool(source_pools[0], n, device), balanced_z[:n])
            attack, _ = mmd_loss(sample_pool(source_pools[1], n, device), balanced_z[n:])
            with torch.no_grad():
                _, teacher_logits = teacher(target_x)
            rank = ranking_loss(teacher_logits[:, 1] - teacher_logits[:, 0],
                                target_logits[:, 1] - target_logits[:, 0])
            terms = {"hidden": hidden, "normal": normal, "attack": attack,
                     "rank": rank, "source": source_ce}
            loss = sum(weights[k] * value for k, value in terms.items())
            if not torch.isfinite(loss):
                raise ValueError("V5d loss contains NaN/Inf")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for name, value in {**terms, "loss": loss}.items():
                totals[name] += value.item()
            steps += 1
        if not steps:
            raise ValueError("Empty target loader")
        row = {"epoch": epoch, **{k: v / steps for k, v in totals.items()}}
        history.append(row)
        print(f"Epoch {epoch}/{settings['epochs']} | {row}", flush=True)
    student.eval()
    return history


def run(config_path, device_name="auto", training_seed=None):
    config, protocol, provenance, source, v2, teacher = load_context(config_path, training_seed)
    output = checkpoint_path(config)
    if output.exists():
        raise FileExistsError(f"Checkpoint already exists: {output}")
    set_seed(config["training_seed"])
    code = code_hashes()
    settings = protocol["training"]
    batch_size = settings["batch_size"]
    source_path = ROOT / "data/features/unsw_train"
    target_path = resolve_path(protocol["target_data"]["adaptation_train"])
    snapshots = {"source": data_snapshot(source_path), "target": data_snapshot(target_path)}
    metadata = json.loads(resolve_path(config["source_metadata"]).read_text())
    if int(metadata["input_dim"]) != provenance["source_dim"]:
        raise ValueError("UNSW metadata dimension mismatch")
    counts = [int(metadata["class_counts"][str(k)]) for k in (0, 1)]
    class_weights = baseline_class_weights(counts)
    normal, attack, pseudo = build_frozen_target_pools(v2, protocol, provenance["target_dim"], batch_size)
    device = torch.device(("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
                          if device_name == "auto" else device_name)
    student = HDAV5DModel(source, teacher.adapter, classifier_only=config["classifier_only"]).to(device)
    source.to(device)
    teacher.to(device)
    source_pools = build_source_pools(source, make_loader(source_path, provenance["source_dim"], batch_size), device)
    print(f"V5d | device={device} | source weights={class_weights.tolist()}", flush=True)
    history = train_model(
        student, teacher, make_loader(source_path, provenance["source_dim"], batch_size, training=True),
        make_unlabeled_loader(target_path, provenance["target_dim"], batch_size),
        source_pools, {0: normal, 1: attack}, class_weights, config, settings)
    if (snapshots != {"source": data_snapshot(source_path), "target": data_snapshot(target_path)}
            or code != code_hashes()):
        raise ValueError("Training data or code changed during training")
    # Also revalidate frozen checkpoints and configuration before publishing output.
    _, _, current_provenance, *_ = load_context(config_path, training_seed)
    if current_provenance != provenance:
        raise ValueError("Frozen inputs changed during training")
    checkpoint = {
        "version": "v5d", "architecture": "hda_v5d",
        "training_seed": config["training_seed"], "teacher_seed": config["teacher_seed"],
        "classifier_only": config["classifier_only"],
        "provenance": provenance, "code_sha256": code, "training_data": snapshots,
        "loss_weights": config["loss_weights"], "training": settings,
        "optimizer": {k: config[k] for k in ("adapter_lr", "classifier_lr", "weight_decay")},
        "source_class_counts": counts, "source_class_weights": class_weights.tolist(),
        "pseudo_metadata": pseudo, "target_labels_used": False,
        "checkpoint_selection": "last epoch", "teacher": "frozen_v5b_asymmetric",
        "history": history,
        "target_adapter_state_dict": {k: v.detach().cpu() for k, v in student.adapter.state_dict().items()},
        "classifier_state_dict": {k: v.detach().cpu() for k, v in student.classifier.state_dict().items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(checkpoint, stream)
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5d.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--training-seed", type=int, choices=(42, 43, 44))
    args = parser.parse_args()
    run(args.config, args.device, args.training_seed)


if __name__ == "__main__":
    main()
