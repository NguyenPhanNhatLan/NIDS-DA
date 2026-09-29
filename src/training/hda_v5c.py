"""Frozen V5b teacher + adapter-only filtered pseudo-label training."""
import argparse

import torch
from torch.nn import functional as F

from evaluation.hda_v5b_calibration import build_frozen_target_pools, data_snapshot
from models.hda_v1 import HDAV1Model
from training.adaptation import mmd_loss
from training.baseline import set_seed
from training.hda_v4 import build_source_pools, sample_pool
from training.hda_v5b import ROOT, file_hash, ranking_loss
from training.thesis_protocol import resolve_path
from training.v5c_common import audit_paths, checkpoint_path, load_audit, load_context, pipeline_hashes
from training.v6_data import make_loader, make_unlabeled_loader


LOSS_WEIGHTS = {"hidden": 1.0, "normal": 0.05, "attack": 0.02, "rank": 0.10}


def train_adapter(source, teacher, source_loader, target_loader, source_pools,
                  conditional_pools, filtered_pools, settings, lambda_pl):
    device = next(source.parameters()).device
    for model in (source, teacher):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
    student = HDAV1Model(teacher.adapter.input_dim, source).to(device)
    student.adapter.load_state_dict(teacher.adapter.state_dict())
    optimizer = torch.optim.Adam(student.adapter.parameters(),
                                 lr=settings["learning_rate"], weight_decay=1e-4)
    n = settings["class_batch_size"]
    labels = torch.cat((torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long))).to(device)
    bns = [m for m in student.adapter.modules()
           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        student.eval()
        student.adapter.train()
        source_batches = iter(source_loader)
        totals = dict.fromkeys(("loss", "hidden", "normal", "attack", "rank", "pl"), 0.0)
        steps = 0
        for target_x in target_loader:
            try:
                source_x, _ = next(source_batches)
            except StopIteration:
                source_batches = iter(source_loader)
                try:
                    source_x, _ = next(source_batches)
                except StopIteration:
                    raise ValueError("Empty source training loader") from None
            target_x = target_x.to(device)
            with torch.no_grad():
                source_h = source.encode_hidden(source_x.to(device))
                _, teacher_logits = teacher(target_x)
            target_h = student.adapter(target_x)
            hidden, _ = mmd_loss(source_h, target_h)
            logits = source.classifier(torch.relu(source.bn2(source.fc2(target_h))))
            rank = ranking_loss(teacher_logits[:, 1] - teacher_logits[:, 0],
                                logits[:, 1] - logits[:, 0])
            conditional_x = torch.cat([sample_pool(conditional_pools[k], n, device) for k in (0, 1)])
            pseudo_x = torch.cat([sample_pool(filtered_pools[k], n, device) for k in ("normal", "attack")])
            # Natural target batch updates adapter BN once. Balanced batches do not.
            for bn in bns:
                bn.eval()
            try:
                conditional_z = student.encoder(conditional_x)
                _, pseudo_logits = student(pseudo_x)
            finally:
                for bn in bns:
                    bn.train()
            normal, _ = mmd_loss(sample_pool(source_pools[0], n, device), conditional_z[:n])
            attack, _ = mmd_loss(sample_pool(source_pools[1], n, device), conditional_z[n:])
            pl = F.cross_entropy(pseudo_logits, labels)
            terms = {"hidden": hidden, "normal": normal, "attack": attack, "rank": rank}
            loss = sum(LOSS_WEIGHTS[k] * v for k, v in terms.items()) + lambda_pl * pl
            if not torch.isfinite(loss):
                raise ValueError("V5c loss contains NaN/Inf")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for name, value in {**terms, "pl": pl, "loss": loss}.items():
                totals[name] += value.item()
            steps += 1
        if not steps:
            raise ValueError("Empty target training loader")
        row = {"epoch": epoch, **{k: v / steps for k, v in totals.items()}}
        history.append(row)
        print(f"Epoch {epoch}/{settings['epochs']} | {row}", flush=True)
    student.eval()
    return student, history


def run(config_path, device_name="auto"):
    config, protocol, provenance, source, v2, teacher = load_context(config_path)
    output = checkpoint_path(config)
    if output.exists():
        raise FileExistsError(f"Checkpoint already exists: {output}")
    audit, filtered = load_audit(config, protocol, provenance)
    audit_path, _ = audit_paths(config)
    audit_hash = file_hash(audit_path)
    code = pipeline_hashes()
    set_seed(config["seed"])
    settings = protocol["training"]
    batch_size = settings["batch_size"]
    target_path = resolve_path(protocol["target_data"]["adaptation_train"])
    source_path = ROOT / "data/features/unsw_train"
    source_snapshot = data_snapshot(source_path)
    # Preserve V5b conditional MMD pools; filtered pools are used for the new CE term.
    normal, attack, pseudo = build_frozen_target_pools(v2, protocol, provenance["target_dim"], batch_size)
    device = torch.device(("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
                          if device_name == "auto" else device_name)
    source.to(device)
    teacher.to(device)
    source_pools = build_source_pools(
        source, make_loader(source_path, provenance["source_dim"], batch_size), device)
    print(f"V5c | seed={config['seed']} | lambda_PL={config['lambda_pl']} | device={device}", flush=True)
    student, history = train_adapter(
        source, teacher,
        make_loader(source_path, provenance["source_dim"], batch_size, training=True),
        make_unlabeled_loader(target_path, provenance["target_dim"], batch_size),
        source_pools, {0: normal, 1: attack}, filtered, settings, config["lambda_pl"])
    if (data_snapshot(target_path) != audit["target_train_files"]
            or data_snapshot(source_path) != source_snapshot
            or file_hash(audit_path) != audit_hash or pipeline_hashes() != code):
        raise ValueError("Data, audit or code changed during training")
    checkpoint = {
        "version": "v5c", "architecture": "hda_v1", "seed": config["seed"],
        "experiment_id": config["experiment_id"], "provenance": provenance,
        "code_sha256": code, "audit_sha256": audit_hash,
        "loss_weights": {**LOSS_WEIGHTS, "pl": config["lambda_pl"]},
        "training": settings, "history": history, "target_labels_used": False,
        "teacher": "frozen_v5b_asymmetric", "initialization": "frozen_v5b_asymmetric",
        "conditional_pseudo_metadata": pseudo, "conditional_adapter_bn_mode": "eval",
        "source_train_files": source_snapshot, "target_train_files": audit["target_train_files"],
        "target_adapter_state_dict": {k: v.detach().cpu() for k, v in student.adapter.state_dict().items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(checkpoint, stream)
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5c.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    run(args.config, args.device)


if __name__ == "__main__":
    main()
