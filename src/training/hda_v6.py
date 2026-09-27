import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from models.hda_v6 import HDAV6Model
from training.adaptation import make_unlabeled_loader, mmd_loss
from training.baseline import make_loader, set_seed, train_baseline
from training.hda_v4 import build_pseudo_pools, make_teacher_loader, sample_pool


ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    path = ROOT / path
    config = json.loads(path.read_text())
    return config, file_hash(path)


def code_hashes():
    files = [
        "src/models/hda_v6.py", "src/training/hda_v6.py",
        "src/training/baseline.py", "src/training/adaptation.py",
        "src/training/hda_v4.py", "src/models/baseline.py",
        "src/models/hda_v1.py", "src/evaluation/baseline.py",
    ]
    return {name: file_hash(ROOT / name) for name in files}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_path(config, version, stage, seed):
    architecture = config["variants"][version]["architecture"]
    directory = ROOT / config["checkpoint_dir"]
    if stage == "source":
        return directory / f"source_{architecture}_seed{config['source_seed']}.pt"
    if stage == "warmup":
        return directory / f"warmup_{architecture}_seed{seed}.pt"
    return directory / f"{version}_seed{seed}.pt"


def load_checkpoint(path, config_hash):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["config_sha256"] != config_hash:
        raise ValueError(f"Checkpoint config mismatch: {path}")
    if checkpoint["code_sha256"] != code_hashes():
        raise ValueError(f"Checkpoint code mismatch: {path}")
    return checkpoint


def make_source_pools(loader):
    parts = {0: [], 1: []}
    for features, labels in loader:
        for label in (0, 1):
            parts[label].append(features[labels == label])
    if not parts[0]:
        raise ValueError("Source loader is empty")
    pools = {label: torch.cat(parts[label]) for label in (0, 1)}
    if any(len(pool) < 2 for pool in pools.values()):
        raise ValueError("Source pools require both classes")
    return pools


def load_teacher(config, seed, source_dim, target_dim, device):
    source_path = ROOT / f"models/baselines/unsw_seed{config['source_seed']}.pt"
    teacher_path = ROOT / config["teacher_dir"] / f"unsw_to_cicids_mmd_v2_seed{seed}.pt"
    source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=True)
    expected_protocol = file_hash(ROOT / config["teacher_protocol"])
    if (checkpoint.get("protocol_sha256") != expected_protocol
            or checkpoint.get("method") != "hda_shared_semantic_hidden_mmd"
            or checkpoint["seed"] != seed
            or checkpoint.get("source_seed", seed) != config["source_seed"]
            or checkpoint["source_dim"] != source_dim
            or checkpoint["target_dim"] != target_dim
            or source_checkpoint["input_dim"] != source_dim):
        raise ValueError("V2 teacher does not match source, target, seed or protocol")
    source = BaselineMLP(source_dim).to(device)
    source.load_state_dict(source_checkpoint["model_state_dict"])
    teacher = HDAV1Model(target_dim, source).to(device)
    teacher.adapter.load_state_dict(checkpoint["target_adapter_state_dict"])
    teacher.eval()
    return teacher, {
        "teacher_checkpoint": str(teacher_path),
        "teacher_sha256": file_hash(teacher_path),
        "teacher_source_sha256": file_hash(source_path),
    }


def configure_training(model, train_shared):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = train_shared
    for parameter in model.target_stem.parameters():
        parameter.requires_grad = True
    if train_shared:
        model.train()
    else:
        model.target_stem.train()


def train_alignment(model, source_loader, target_loader, settings, epochs, lr,
                    weight_decay, class_batch_size=64, source_pools=None,
                    target_pools=None):
    device = next(model.parameters()).device
    configure_training(model, settings["train_shared"])
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    history = []
    conditional = settings["normal"] > 0 or settings["attack"] > 0
    if conditional and (source_pools is None or target_pools is None):
        raise ValueError("Conditional alignment requires source and target pools")

    for epoch in range(1, epochs + 1):
        source_batches = iter(source_loader)
        totals = {name: 0.0 for name in ("loss", "source_ce", "marginal", "normal", "attack")}
        steps = 0
        for target_x in target_loader:
            try:
                source_x, source_y = next(source_batches)
            except StopIteration:
                source_batches = iter(source_loader)
                try:
                    source_x, source_y = next(source_batches)
                except StopIteration:
                    raise ValueError("Source loader is empty") from None
            source_x = source_x.to(device)
            source_y = source_y.to(device)
            target_x = target_x.to(device)
            source_h = model.encode_hidden(source_x, "source")
            target_h = model.encode_hidden(target_x, "target")
            marginal, _ = mmd_loss(source_h, target_h)
            source_ce = torch.zeros((), device=device)
            if settings["source_ce"] > 0:
                source_logits = model.classifier(model.encoder(source_h))
                source_ce = criterion(source_logits, source_y)
            normal = torch.zeros((), device=device)
            attack = torch.zeros((), device=device)
            if conditional:
                source_normal = sample_pool(source_pools[0], class_batch_size, device)
                source_attack = sample_pool(source_pools[1], class_batch_size, device)
                target_normal = sample_pool(target_pools[0], class_batch_size, device)
                target_attack = sample_pool(target_pools[1], class_batch_size, device)
                source_z, _ = model(torch.cat((source_normal, source_attack)), "source")
                target_z, _ = model(torch.cat((target_normal, target_attack)), "target")
                normal, _ = mmd_loss(source_z[:class_batch_size], target_z[:class_batch_size])
                attack, _ = mmd_loss(source_z[class_batch_size:], target_z[class_batch_size:])
            loss = (settings["source_ce"] * source_ce
                    + settings["marginal"] * marginal
                    + settings["normal"] * normal
                    + settings["attack"] * attack)
            if not torch.isfinite(loss):
                raise ValueError("V6 loss contains NaN or Inf")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            values = {"loss": loss, "source_ce": source_ce, "marginal": marginal,
                      "normal": normal, "attack": attack}
            for name, value in values.items():
                totals[name] += value.item()
            steps += 1
        if steps == 0:
            raise ValueError("Target loader is empty")
        row = {"epoch": epoch, **{name: value / steps for name, value in totals.items()}}
        history.append(row)
        print(f"Epoch {epoch:02d}/{epochs} | {row}", flush=True)
    model.eval()
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hda_v6.json")
    parser.add_argument("--version", choices=["v6a", "v6b", "v6c"], required=True)
    parser.add_argument("--stage", choices=["source", "warmup", "adapt"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config, config_hash = load_config(args.config)
    if args.seed not in config["development_seeds"]:
        raise ValueError("Seed is not declared for development")
    settings = config["variants"][args.version]
    architecture = settings["architecture"]
    seed = config["source_seed"] if args.stage == "source" else args.seed
    output = checkpoint_path(config, args.version, args.stage, seed)
    if output.exists():
        raise FileExistsError(f"Checkpoint already exists: {output}")
    set_seed(seed)
    source_metadata = json.loads((ROOT / "data/features/unsw_metadata.json").read_text())
    target_metadata = json.loads((ROOT / "data/features/cicids_metadata.json").read_text())
    source_dim = source_metadata["input_dim"]
    target_dim = target_metadata["input_dim"]
    device = get_device()
    model = HDAV6Model(source_dim, target_dim, architecture).to(device)
    batch_size = config["batch_size"]
    source_loader = make_loader(ROOT / config["source_train"], source_dim, batch_size, training=True)
    metadata = {
        "experiment_id": config["experiment_id"], "version": args.version,
        "stage": args.stage, "architecture": architecture,
        "seed": seed, "source_seed": config["source_seed"],
        "source_dim": source_dim, "target_dim": target_dim,
        "config_sha256": config_hash, "code_sha256": code_hashes(),
        "target_labels_used": False,
    }
    print(f"{args.version} | stage={args.stage} | seed={seed} | device={device}", flush=True)
    if args.stage == "source":
        counts = [source_metadata["class_counts"][str(label)] for label in (0, 1)]
        validation_loader = make_loader(ROOT / config["source_validation"], source_dim, batch_size)
        model, best_epoch, best_ap = train_baseline(
            model, counts, source_loader, validation_loader,
            epochs=config["source_epochs"], lr=config["learning_rate"],
            patience=config["source_patience"],
        )
        metadata.update(best_epoch=best_epoch, best_val_ap=best_ap)
    else:
        previous_stage = "source" if args.stage == "warmup" else "warmup"
        previous_path = checkpoint_path(config, args.version, previous_stage, seed)
        previous = load_checkpoint(previous_path, config_hash)
        if previous["architecture"] != architecture or previous["stage"] != previous_stage:
            raise ValueError("Previous checkpoint architecture or stage mismatch")
        expected_seed = config["source_seed"] if previous_stage == "source" else seed
        if (previous["seed"] != expected_seed or previous["source_seed"] != config["source_seed"]
                or previous["source_dim"] != source_dim or previous["target_dim"] != target_dim):
            raise ValueError("Previous checkpoint seed or feature dimensions mismatch")
        if args.stage == "warmup":
            state = previous["model_state_dict"]
            for name in ("source_stem", "encoder", "classifier"):
                prefix = name + "."
                weights = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
                getattr(model, name).load_state_dict(weights)
        else:
            model.load_state_dict(previous["model_state_dict"])
        metadata.update(initial_checkpoint=str(previous_path), initial_sha256=file_hash(previous_path))
        target_loader = make_unlabeled_loader(ROOT / config["target_train"], target_dim, batch_size)
        source_pools = None
        target_pools = None
        if args.stage == "warmup":
            settings = {"train_shared": False, "source_ce": 0.0,
                        "marginal": 1.0, "normal": 0.0, "attack": 0.0}
            epochs = config["warmup_epochs"]
        else:
            teacher, teacher_metadata = load_teacher(config, seed, source_dim, target_dim, device)
            normal_pool, attack_pool, pseudo_metadata = build_pseudo_pools(
                teacher, make_teacher_loader(ROOT / config["target_train"], target_dim, batch_size), device,
            )
            target_pools = {0: normal_pool, 1: attack_pool}
            source_pools = make_source_pools(make_loader(ROOT / config["source_train"], source_dim, batch_size))
            metadata.update(teacher_metadata, pseudo_label_metadata=pseudo_metadata)
            epochs = config["adaptation_epochs"]
            del teacher
        history = train_alignment(
            model, source_loader, target_loader, settings, epochs,
            config["learning_rate"], config["weight_decay"], config["class_batch_size"],
            source_pools, target_pools,
        )
        metadata.update(history=history, loss_weights=settings)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**metadata, "model_state_dict": model.cpu().state_dict()}, output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
