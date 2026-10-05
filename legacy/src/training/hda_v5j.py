"""V5j: frozen V5b asymmetric HDA + latent virtual adversarial training."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from evaluation.protocol_revision import load_evaluation_revision
from evaluation.hda_v5b_calibration import data_snapshot
from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from training.adaptation import mmd_loss
from training.baseline import set_seed
from training.hda_v4 import build_pseudo_pools, build_source_pools, sample_pool
from training.hda_v5b import ranking_loss
from training.latent_vat import latent_vat_loss
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader, make_teacher_loader, make_unlabeled_loader


ROOT = Path(__file__).resolve().parents[2]
V5B_WEIGHTS = {"hidden": 1.0, "normal": 0.05, "attack": 0.02, "rank": 0.10}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    files = (
        "src/models/hda_v1.py",
        "src/training/hda_v5j.py",
        "src/training/latent_vat.py",
        "src/training/v6_data.py",
        "src/training/baseline.py",
        "src/training/adaptation.py",
        "src/training/hda_v4.py",
        "src/training/hda_v5b.py",
        "src/evaluation/hda_v5j.py",
        "src/evaluation/calibration.py",
        "src/evaluation/hda_v5b_calibration.py",
        "src/evaluation/baseline.py",
        "src/evaluation/protocol_revision.py",
        "src/training/thesis_protocol.py",
    )
    return {name: file_hash(ROOT / name) for name in files}


def load_setup(config_path):
    path = resolve_path(config_path)
    config = json.loads(path.read_text())
    if config.get("architecture") != "hda_v1":
        raise ValueError("V5j requires HDAV1Model")
    if config.get("loss_weights") != V5B_WEIGHTS:
        raise ValueError("V5j must preserve the exact V5b asymmetric loss weights")

    vat = config.get("vat", {})
    expected = {"weight", "epsilon_ratio", "xi_ratio", "power_iterations"}
    if set(vat) != expected:
        raise ValueError(f"vat must contain exactly: {sorted(expected)}")
    if not math.isfinite(vat["weight"]) or vat["weight"] <= 0:
        raise ValueError("vat.weight must be positive and finite")
    if not math.isfinite(vat["epsilon_ratio"]) or not 0 < vat["epsilon_ratio"] <= 1:
        raise ValueError("vat.epsilon_ratio must be in (0,1]")
    if not math.isfinite(vat["xi_ratio"]) or not 0 < vat["xi_ratio"] < vat["epsilon_ratio"]:
        raise ValueError("vat.xi_ratio must be positive and smaller than epsilon_ratio")
    if type(vat["power_iterations"]) is not int or vat["power_iterations"] != 1:
        raise ValueError("First V5j experiment fixes vat.power_iterations=1")

    protocol, protocol_hash, _ = load_evaluation_revision(
        config["parent_protocol"], config["evaluation_revision"],
    )
    if not math.isfinite(config["calibration_max_fpr"]) or not 0 <= config["calibration_max_fpr"] <= 1:
        raise ValueError("calibration_max_fpr must be in [0,1]")
    if resolve_path(config["source_validation"]).resolve() != (ROOT / "data/features/unsw_val").resolve():
        raise ValueError("Use the fixed UNSW validation split")
    if resolve_path(protocol["target_data"]["adaptation_train"]).resolve() == evaluation_target(protocol, "development").resolve():
        raise ValueError("Adaptation and development must be separate")
    return config, file_hash(path), protocol, protocol_hash


def training_snapshot(protocol):
    return {"source": data_snapshot(ROOT / "data/features/unsw_train"),
            "target": data_snapshot(resolve_path(protocol["target_data"]["adaptation_train"]))}


def load_v5b_reference(config, protocol, seed):
    path = resolve_path(config["v5b_reference_report"])
    if file_hash(path) != config["v5b_reference_sha256"]:
        raise ValueError("Pinned V5b reference report changed")
    report = json.loads(path.read_text())
    if (seed != config["v5b_reference_seed"] or report.get("phase") != "development"
            or resolve_path(report["target_data"]).resolve() != evaluation_target(protocol, "development").resolve()
            or report["threshold_policy"]["max_fpr"] != config["calibration_max_fpr"]):
        raise ValueError("V5b reference seed, development split or FPR policy differs")
    return report


def preflight(config_path, seed=42):
    config, config_hash, protocol, protocol_hash = load_setup(config_path)
    source, teacher, provenance = load_models(protocol, protocol_hash, seed, torch.device("cpu"))
    load_v5b_reference(config, protocol, seed)
    print("V2 teacher/source checkpoints and V5b reference: VERIFIED", flush=True)
    print(f"loss_weights: {config['loss_weights']}", flush=True)
    print(f"VAT: {config['vat']}", flush=True)
    return config, config_hash, protocol, protocol_hash, source, teacher, provenance


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
            or checkpoint["seed"] != seed
            or checkpoint["source_dim"] != source_dim
            or checkpoint.get("source_seed", seed) != source_seed):
        raise ValueError("V2 teacher does not match the frozen source/protocol/seed")

    source = BaselineMLP(source_dim).to(device)
    source.load_state_dict(source_checkpoint["model_state_dict"])
    teacher = HDAV1Model(target_dim, source).to(device)
    teacher.adapter.load_state_dict(checkpoint["target_adapter_state_dict"])

    for model in (source, teacher):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

    provenance = {
        "source_seed": source_seed,
        "source_dim": source_dim,
        "target_dim": target_dim,
        "source_checkpoint_sha256": file_hash(source_path),
        "teacher_checkpoint_sha256": file_hash(teacher_path),
    }
    return source, teacher, provenance


def checkpoint_path(config, seed):
    return resolve_path(config["checkpoint_dir"]) / f"v5j_seed{seed}.pt"


def load_student(config_path, seed=42, device=torch.device("cpu")):
    config, config_hash, protocol, protocol_hash = load_setup(config_path)
    source, teacher, provenance = load_models(protocol, protocol_hash, seed, device)
    checkpoint = torch.load(checkpoint_path(config, seed), map_location="cpu", weights_only=True)
    if (checkpoint.get("version") != "v5j"
            or checkpoint.get("architecture") != "hda_v1"
            or checkpoint.get("seed") != seed
            or checkpoint.get("config_sha256") != config_hash
            or checkpoint.get("protocol_sha256") != protocol_hash
            or checkpoint.get("code_sha256") != code_hashes()
            or checkpoint.get("loss_weights") != V5B_WEIGHTS
            or checkpoint.get("vat") != config["vat"]):
        raise ValueError("V5j checkpoint provenance mismatch")
    for key, value in provenance.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"V5j frozen dependency mismatch: {key}")
    if checkpoint.get("training_data") != training_snapshot(protocol):
        raise ValueError("V5j training data changed")

    student = HDAV1Model(provenance["target_dim"], source).to(device)
    student.adapter.load_state_dict(checkpoint["target_adapter_state_dict"])
    student.eval().requires_grad_(False)
    return config, protocol, provenance, source, teacher, student, checkpoint


def train_hda_v5j(
    source,
    teacher,
    source_loader,
    target_loader,
    source_pools,
    target_pools,
    weights,
    vat_config,
    epochs=10,
    lr=0.001,
    class_batch_size=64,
):
    device = next(source.parameters()).device
    source.eval()
    teacher.eval()
    for model in (source, teacher):
        for parameter in model.parameters():
            parameter.requires_grad = False

    student = HDAV1Model(teacher.adapter.input_dim, source).to(device)
    student.adapter.load_state_dict(teacher.adapter.state_dict())
    optimizer = torch.optim.Adam(student.adapter.parameters(), lr=lr, weight_decay=1e-4)

    adapter_bns = [
        layer for layer in student.adapter.modules()
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
    ]

    history = []
    for epoch in range(1, epochs + 1):
        student.eval()
        student.adapter.train()
        source_batches = iter(source_loader)
        totals = {name: 0.0 for name in (
            "loss", "hidden", "normal", "attack", "rank", "vat"
        )}
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

            # Natural target forward: same BN behavior as V5b.
            target_h = student.adapter(target_x)
            hidden, _ = mmd_loss(source_h, target_h)

            # Shared frozen source tail. This latent is used by rank + VAT so the
            # only new mechanism is local target consistency at z.
            target_z = torch.relu(source.bn2(source.fc2(target_h)))
            target_logits = source.classifier(target_z)

            rank = torch.zeros((), device=device)
            if weights["rank"] > 0:
                with torch.no_grad():
                    _, teacher_logits = teacher(target_x)
                rank = ranking_loss(
                    teacher_logits[:, 1] - teacher_logits[:, 0],
                    target_logits[:, 1] - target_logits[:, 0],
                )

            vat = latent_vat_loss(
                target_z,
                source.classifier,
                epsilon_ratio=vat_config["epsilon_ratio"],
                xi_ratio=vat_config["xi_ratio"],
                power_iterations=vat_config["power_iterations"],
            )

            # Preserve V5b hard conditional pseudo policy exactly.
            target_normal = sample_pool(target_pools[0], class_batch_size, device)
            target_attack = sample_pool(target_pools[1], class_batch_size, device)

            # Preserve V5b conditional adapter BN=eval behavior.
            for bn in adapter_bns:
                bn.eval()
            try:
                conditional_z = student.encoder(
                    torch.cat((target_normal, target_attack))
                )
            finally:
                for bn in adapter_bns:
                    bn.train()

            source_normal = sample_pool(source_pools[0], class_batch_size, device)
            source_attack = sample_pool(source_pools[1], class_batch_size, device)
            normal, _ = mmd_loss(
                source_normal, conditional_z[:class_batch_size]
            )
            attack, _ = mmd_loss(
                source_attack, conditional_z[class_batch_size:]
            )

            loss = (
                weights["hidden"] * hidden
                + weights["normal"] * normal
                + weights["attack"] * attack
                + weights["rank"] * rank
                + vat_config["weight"] * vat
            )

            if not torch.isfinite(loss):
                raise ValueError("V5j loss contains NaN/Inf")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for name, value in (
                ("loss", loss),
                ("hidden", hidden),
                ("normal", normal),
                ("attack", attack),
                ("rank", rank),
                ("vat", vat),
            ):
                totals[name] += value.item()
            steps += 1

        if steps == 0:
            raise ValueError("Target loader is empty")

        row = {
            "epoch": epoch,
            **{name: value / steps for name, value in totals.items()},
        }
        history.append(row)
        print(f"Epoch {epoch:02d}/{epochs} | {row}", flush=True)

    student.eval()
    return student, history


def run(config_path, seed=42, device_name="auto"):
    config, config_hash, protocol, protocol_hash = load_setup(config_path)
    output = checkpoint_path(config, seed)
    if output.exists():
        raise FileExistsError(f"Checkpoint already exists: {output}")

    set_seed(seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available()
         else "mps" if torch.backends.mps.is_available()
         else "cpu")
        if device_name == "auto" else device_name
    )

    source, teacher, provenance = load_models(protocol, protocol_hash, seed, device)
    load_v5b_reference(config, protocol, seed)
    initial_code = code_hashes()
    snapshots = training_snapshot(protocol)
    settings = protocol["training"]
    batch_size = settings["batch_size"]

    target_train = resolve_path(protocol["target_data"]["adaptation_train"])
    source_train = ROOT / "data/features/unsw_train"

    print(
        f"V5j | {config['experiment_id']} | seed={seed} | device={device}",
        flush=True,
    )
    print(f"V5b weights: {config['loss_weights']}", flush=True)
    print(f"VAT: {config['vat']}", flush=True)

    # Same V5b teacher and fixed pseudo policy.
    normal, attack, pseudo = build_pseudo_pools(
        teacher,
        make_teacher_loader(
            target_train, provenance["target_dim"], batch_size
        ),
        device,
    )
    source_pools = build_source_pools(
        source,
        make_loader(
            source_train, provenance["source_dim"], batch_size
        ),
        device,
    )

    student, history = train_hda_v5j(
        source,
        teacher,
        make_loader(
            source_train,
            provenance["source_dim"],
            batch_size,
            training=True,
        ),
        make_unlabeled_loader(
            target_train,
            provenance["target_dim"],
            batch_size,
        ),
        source_pools,
        {0: normal, 1: attack},
        config["loss_weights"],
        config["vat"],
        epochs=settings["epochs"],
        lr=settings["learning_rate"],
        class_batch_size=settings["class_batch_size"],
    )

    _, current_config_hash, _, current_protocol_hash = load_setup(config_path)
    _, _, current_provenance = load_models(protocol, protocol_hash, seed, torch.device("cpu"))
    if (current_config_hash != config_hash or current_protocol_hash != protocol_hash
            or current_provenance != provenance or initial_code != code_hashes()
            or snapshots != training_snapshot(protocol)):
        raise ValueError("Frozen inputs, training data or code changed during training")
    load_v5b_reference(config, protocol, seed)
    checkpoint = {
        **provenance,
        "experiment_id": config["experiment_id"],
        "version": "v5j",
        "architecture": "hda_v1",
        "seed": seed,
        "config_sha256": config_hash,
        "protocol_sha256": protocol_hash,
        "code_sha256": initial_code,
        "training_data": snapshots,
        "target_labels_used": False,
        "loss_weights": config["loss_weights"],
        "vat": config["vat"],
        "training": {
            key: value
            for key, value in settings.items()
            if key != "lambda_conditional"
        },
        "pseudo_label_metadata": pseudo,
        "history": history,
        "conditional_adapter_bn_mode": "eval",
        "checkpoint_selection": "last epoch",
        "target_adapter_state_dict": {
            key: value.detach().cpu()
            for key, value in student.adapter.state_dict().items()
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(checkpoint, stream)
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5j.json")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight(args.config, args.seed)
    else:
        run(args.config, args.seed, args.device)


if __name__ == "__main__":
    main()
