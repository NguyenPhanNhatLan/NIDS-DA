"""V5l: continued low-LR adapter refinement from frozen asymmetric V5b."""
import argparse
import json

import torch

from evaluation.hda_v5b_calibration import (
    build_frozen_target_pools, calibration_code_hashes, data_snapshot, load_frozen_models,
)
from models.hda_v1 import HDAV1Model
from training.baseline import set_seed
from training.hda_v4 import build_source_pools
from training.hda_v5b import ROOT, file_hash, train_hda_v5b
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader, make_unlabeled_loader

V5B_WEIGHTS = {"hidden": 1.0, "normal": 0.05, "attack": 0.02, "rank": 0.10}


def code_hashes():
    hashes = calibration_code_hashes()
    for name in ("src/training/hda_v5l.py", "src/evaluation/hda_v5l.py",
                 "src/training/adaptation.py", "src/training/baseline.py",
                 "src/training/thesis_protocol.py", "src/evaluation/protocol_revision.py"):
        hashes[name] = file_hash(ROOT / name)
    return hashes


def load_context(config_path):
    path = resolve_path(config_path)
    config = json.loads(path.read_text())
    if (config["loss_weights"] != V5B_WEIGHTS or config["adapter_lr"] != 1e-4
            or config["weight_decay"] != 1e-4 or type(config["epochs"]) is not int
            or config["epochs"] != 10):
        raise ValueError("V5l requires V5b weights, adapter_lr=1e-4, weight_decay=1e-4, epochs=10")
    if (type(config["teacher_seed"]) is not int or config["teacher_seed"] != 42
            or type(config["training_seed"]) is not int or config["training_seed"] != 42):
        raise ValueError("This V5l experiment fixes teacher/training seed 42")
    ref_path = resolve_path(config["stage1_reference"])
    reference, protocol, provenance, source, v2, v5b = load_frozen_models(ref_path)
    if reference["seed"] != 42 or reference["loss_weights"] != V5B_WEIGHTS:
        raise ValueError("V5l requires frozen asymmetric V5b seed42")
    if resolve_path(protocol["target_data"]["adaptation_train"]).resolve() == evaluation_target(protocol, "development").resolve():
        raise ValueError("Adaptation and development must be separate")
    provenance = {**provenance, "config_sha256": file_hash(path),
                  "stage1_reference_sha256": file_hash(ref_path),
                  "v5b_checkpoint_sha256": reference["checkpoint_sha256"],
                  "teacher_seed": config["teacher_seed"], "training_seed": config["training_seed"]}
    return config, protocol, provenance, source, v2, v5b


def checkpoint_path(config):
    return resolve_path(config["checkpoint_dir"]) / f"v5l_seed{config['training_seed']}.pt"


def training_snapshot(protocol):
    return {"source": data_snapshot(ROOT / "data/features/unsw_train"),
            "target": data_snapshot(resolve_path(protocol["target_data"]["adaptation_train"]))}


def load_student(config, protocol, provenance, source, v5b):
    checkpoint = torch.load(checkpoint_path(config), map_location="cpu", weights_only=True)
    if (checkpoint.get("version") != "v5l" or checkpoint.get("architecture") != "hda_v1"
            or checkpoint["provenance"] != provenance or checkpoint["code_sha256"] != code_hashes()
            or checkpoint["loss_weights"] != V5B_WEIGHTS
            or checkpoint["training_data"] != training_snapshot(protocol)):
        raise ValueError("V5l checkpoint provenance or training data mismatch")
    student = HDAV1Model(provenance["target_dim"], source)
    student.adapter.load_state_dict(checkpoint["target_adapter_state_dict"])
    return student.eval().requires_grad_(False), checkpoint


def train_v5l(source, v5b_teacher, source_loader, target_loader, source_pools,
              target_pools, weights, epochs=10, lr=1e-4, weight_decay=1e-4,
              class_batch_size=64):
    if weights != V5B_WEIGHTS or lr != 1e-4 or weight_decay != 1e-4:
        raise ValueError("V5l refinement policy mismatch")
    before_source = {k: v.detach().clone() for k, v in source.state_dict().items()}
    before_teacher = {k: v.detach().clone() for k, v in v5b_teacher.state_dict().items()}
    # The existing V5b loop creates a new HDAV1Model, copies the supplied
    # teacher.adapter, and optimizes ONLY that adapter (Adam, weight_decay=1e-4).
    # Passing V5b here supplies BOTH V5b initialization and the V5b rank teacher.
    # V2 appears only in the separate pseudo-pool builder in run().
    student, history = train_hda_v5b(
        source, v5b_teacher, source_loader, target_loader, source_pools,
        target_pools, weights, epochs=epochs, lr=lr, class_batch_size=class_batch_size)
    for model, snapshot in ((source, before_source), (v5b_teacher, before_teacher)):
        for key, value in model.state_dict().items():
            if not torch.equal(value, snapshot[key]):
                raise ValueError(f"Frozen source/teacher changed: {key}")
    return student, history


def preflight(config_path):
    context = load_context(config_path)
    config = context[0]
    print("Frozen V5b initialization/ranking teacher and V2 pseudo teacher: VERIFIED", flush=True)
    for key in ("loss_weights", "adapter_lr", "weight_decay", "epochs"):
        print(f"{key}: {config[key]}", flush=True)
    return context


def run(config_path, device_name="auto"):
    config, protocol, provenance, source, v2, v5b = preflight(config_path)
    output = checkpoint_path(config)
    if output.exists():
        raise FileExistsError(f"Checkpoint already exists: {output}")
    set_seed(config["training_seed"])
    code, snapshots = code_hashes(), training_snapshot(protocol)
    bs = protocol["training"]["batch_size"]
    source_path = ROOT / "data/features/unsw_train"
    target_path = resolve_path(protocol["target_data"]["adaptation_train"])
    print("Building frozen V2 hard anchors on CPU...", flush=True)
    normal, attack, pseudo = build_frozen_target_pools(v2, protocol, provenance["target_dim"], bs)
    device = torch.device(("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
                          if device_name == "auto" else device_name)
    source.to(device)
    v5b.to(device)
    source_pools = build_source_pools(source, make_loader(source_path, provenance["source_dim"], bs), device)
    print(f"V5l refinement | device={device} | source/classifier frozen", flush=True)
    student, history = train_v5l(
        source, v5b, make_loader(source_path, provenance["source_dim"], bs, training=True),
        make_unlabeled_loader(target_path, provenance["target_dim"], bs),
        source_pools, {0: normal, 1: attack}, config["loss_weights"],
        epochs=config["epochs"], lr=config["adapter_lr"], weight_decay=config["weight_decay"],
        class_batch_size=protocol["training"]["class_batch_size"])
    if (snapshots != training_snapshot(protocol) or code != code_hashes()
            or load_context(config_path)[2] != provenance):
        raise ValueError("Frozen inputs, training data or code changed during refinement")
    checkpoint = {
        "version": "v5l", "architecture": "hda_v1", "provenance": provenance,
        "code_sha256": code, "training_data": snapshots, "loss_weights": config["loss_weights"],
        "training": {"epochs": config["epochs"], "batch_size": bs,
                     "class_batch_size": protocol["training"]["class_batch_size"]},
        "optimizer": {"adapter_lr": config["adapter_lr"], "weight_decay": config["weight_decay"]},
        "initialization": "frozen_v5b_adapter", "ranking_teacher": "frozen_v5b",
        "pseudo_teacher": "frozen_v2", "pseudo_metadata": pseudo,
        "classifier_frozen": True, "source_frozen": True, "target_labels_used": False,
        "checkpoint_selection": "last epoch", "history": history,
        "target_adapter_state_dict": {k: v.detach().cpu() for k, v in student.adapter.state_dict().items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(checkpoint, stream)
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5l.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight(args.config)
    else:
        run(args.config, args.device)


if __name__ == "__main__":
    main()
