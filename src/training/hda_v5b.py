import argparse
import hashlib
import json
from pathlib import Path

import torch

from evaluation.protocol_revision import load_evaluation_revision
from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from training.adaptation import mmd_loss
from training.baseline import set_seed
from training.hda_v4 import build_pseudo_pools, build_source_pools, sample_pool
from training.thesis_protocol import resolve_path
from training.v6_data import make_loader, make_teacher_loader, make_unlabeled_loader


ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    files = ["src/models/hda_v1.py", "src/training/hda_v5b.py",
             "src/training/v6_data.py", "src/training/baseline.py"]
    return {name: file_hash(ROOT / name) for name in files}


def load_setup(config_path):
    path = resolve_path(config_path)
    config = json.loads(path.read_text())
    if config.get("architecture") != "hda_v1":
        raise ValueError("V5b requires the HDAV1Model configuration")
    protocol, protocol_hash, _ = load_evaluation_revision(
        config["parent_protocol"], config["evaluation_revision"],
    )
    return config, file_hash(path), protocol, protocol_hash


def load_models(protocol, protocol_hash, seed, device):
    if seed not in protocol["development_seeds"]:
        raise ValueError("Seed is not declared for development")
    source_seed = protocol["source_pretraining_seed"]
    source_path = ROOT / f"models/baselines/unsw_seed{source_seed}.pt"
    teacher_path = resolve_path(protocol["checkpoint_dir"]) / f"unsw_to_cicids_mmd_v2_seed{seed}.pt"
    source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=True)
    source_dim = source_checkpoint["input_dim"]
    target_dim = checkpoint["target_dim"]
    if (checkpoint.get("protocol_sha256") != protocol_hash
            or checkpoint.get("method") != "hda_shared_semantic_hidden_mmd"
            or checkpoint["seed"] != seed or checkpoint["source_dim"] != source_dim
            or checkpoint.get("source_seed", seed) != source_seed):
        raise ValueError("V2 teacher does not match the frozen source/protocol/seed")
    source = BaselineMLP(source_dim).to(device)
    source.load_state_dict(source_checkpoint["model_state_dict"])
    teacher = HDAV1Model(target_dim, source).to(device)
    teacher.adapter.load_state_dict(checkpoint["target_adapter_state_dict"])
    source.eval()
    teacher.eval()
    for model in (source, teacher):
        for parameter in model.parameters():
            parameter.requires_grad = False
    provenance = {
        "source_seed": source_seed, "source_dim": source_dim, "target_dim": target_dim,
        "source_checkpoint_sha256": file_hash(source_path),
        "teacher_checkpoint_sha256": file_hash(teacher_path),
    }
    return source, teacher, provenance


def ranking_loss(teacher_margin, student_margin):
    teacher = teacher_margin.detach() - teacher_margin.detach().mean()
    student = student_margin - student_margin.mean()
    teacher_norm = torch.linalg.vector_norm(teacher)
    if teacher_norm <= 1e-8:
        return student.sum() * 0.0
    student_norm = torch.linalg.vector_norm(student).clamp_min(1e-8)
    correlation = (teacher * student).sum() / (teacher_norm * student_norm)
    return 1.0 - correlation.clamp(-1.0, 1.0)


def train_hda_v5b(source, teacher, source_loader, target_loader, source_pools,
                  target_pools, weights, epochs=10, lr=0.001, class_batch_size=64):
    device = next(source.parameters()).device
    source.eval()
    teacher.eval()
    for model in (source, teacher):
        for parameter in model.parameters():
            parameter.requires_grad = False
    student = HDAV1Model(teacher.adapter.input_dim, source).to(device)
    student.adapter.load_state_dict(teacher.adapter.state_dict())
    optimizer = torch.optim.Adam(student.adapter.parameters(), lr=lr, weight_decay=1e-4)
    adapter_bns = [layer for layer in student.adapter.modules()
                   if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)]
    history = []

    for epoch in range(1, epochs + 1):
        student.eval()
        student.adapter.train()
        source_batches = iter(source_loader)
        totals = {name: 0.0 for name in ("loss", "hidden", "normal", "attack", "rank")}
        steps = 0
        for target_x in target_loader:
            try:
                source_x, _ = next(source_batches)
            except StopIteration:
                source_batches = iter(source_loader)
                try:
                    source_x, _ = next(source_batches)
                except StopIteration:
                    raise ValueError("Source loader is empty") from None
            target_x = target_x.to(device)
            with torch.no_grad():
                source_h = source.encode_hidden(source_x.to(device))
            target_h = student.adapter(target_x)
            hidden, _ = mmd_loss(source_h, target_h)
            rank = torch.zeros((), device=device)
            if weights["rank"] > 0:
                with torch.no_grad():
                    _, teacher_logits = teacher(target_x)
                student_z = torch.relu(source.bn2(source.fc2(target_h)))
                student_logits = source.classifier(student_z)
                rank = ranking_loss(teacher_logits[:, 1] - teacher_logits[:, 0],
                                    student_logits[:, 1] - student_logits[:, 0])
            target_normal = sample_pool(target_pools[0], class_batch_size, device)
            target_attack = sample_pool(target_pools[1], class_batch_size, device)
            for bn in adapter_bns:
                bn.eval()
            try:
                target_z = student.encoder(torch.cat((target_normal, target_attack)))
            finally:
                for bn in adapter_bns:
                    bn.train()
            source_normal = sample_pool(source_pools[0], class_batch_size, device)
            source_attack = sample_pool(source_pools[1], class_batch_size, device)
            normal, _ = mmd_loss(source_normal, target_z[:class_batch_size])
            attack, _ = mmd_loss(source_attack, target_z[class_batch_size:])
            loss = (weights["hidden"] * hidden + weights["normal"] * normal
                    + weights["attack"] * attack + weights["rank"] * rank)
            if not torch.isfinite(loss):
                raise ValueError("V5b loss contains NaN or Inf")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for name, value in (("loss", loss), ("hidden", hidden), ("normal", normal),
                                ("attack", attack), ("rank", rank)):
                totals[name] += value.item()
            steps += 1
        if steps == 0:
            raise ValueError("Target loader is empty")
        row = {"epoch": epoch, **{name: value / steps for name, value in totals.items()}}
        history.append(row)
        print(f"Epoch {epoch:02d}/{epochs} | {row}", flush=True)
    student.eval()
    return student, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hda_v5b_rank_0p10_symmetric.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config, config_hash, protocol, protocol_hash = load_setup(args.config)
    output = resolve_path(config["checkpoint_dir"]) / f"v5b_seed{args.seed}.pt"
    if output.exists():
        raise FileExistsError(f"Checkpoint already exists: {output}")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    source, teacher, provenance = load_models(protocol, protocol_hash, args.seed, device)
    threshold_path = ROOT / f"results/baseline/unsw_seed{provenance['source_seed']}.json"
    threshold = json.loads(threshold_path.read_text())["threshold"]
    settings = protocol["training"]
    batch_size = settings["batch_size"]
    target_train = resolve_path(protocol["target_data"]["adaptation_train"])
    source_train = ROOT / "data/features/unsw_train"
    print(f"V5b | {config['experiment_id']} | seed={args.seed} | device={device}", flush=True)
    normal, attack, pseudo = build_pseudo_pools(
        teacher, make_teacher_loader(target_train, provenance["target_dim"], batch_size), device,
    )
    source_pools = build_source_pools(
        source, make_loader(source_train, provenance["source_dim"], batch_size), device,
    )
    student, history = train_hda_v5b(
        source, teacher,
        make_loader(source_train, provenance["source_dim"], batch_size, training=True),
        make_unlabeled_loader(target_train, provenance["target_dim"], batch_size),
        source_pools, {0: normal, 1: attack}, config["loss_weights"],
        epochs=settings["epochs"], lr=settings["learning_rate"],
        class_batch_size=settings["class_batch_size"],
    )
    checkpoint = {
        **provenance, "experiment_id": config["experiment_id"], "version": "v5b",
        "architecture": "hda_v1", "seed": args.seed, "config_sha256": config_hash,
        "protocol_sha256": protocol_hash, "code_sha256": code_hashes(),
        "target_labels_used": False, "loss_weights": config["loss_weights"],
        "training": {key: value for key, value in settings.items() if key != "lambda_conditional"},
        "threshold_source": threshold, "threshold_file_sha256": file_hash(threshold_path),
        "pseudo_label_metadata": pseudo, "history": history,
        "conditional_adapter_bn_mode": "eval",
        "target_adapter_state_dict": {key: value.detach().cpu() for key, value in student.adapter.state_dict().items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

