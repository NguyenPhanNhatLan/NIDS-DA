"""Development-only lambda ablation; reuse the frozen V4 training implementation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from evaluation.baseline import collect_scores, compute_metrics, select_f1_threshold
from evaluation.protocol_revision import load_evaluation_revision
from models.baseline import BaselineMLP
from models.hda_v1 import HDAV1Model
from training.adaptation import make_unlabeled_loader
from training.baseline import make_loader, set_seed
from training.hda_v4 import (
    build_pseudo_pools, build_source_pools, make_teacher_loader, train_hda_v4,
)
from training.thesis_protocol import evaluation_target, resolve_path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5a_sweep.json")
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    config = json.loads(config_path.read_text())
    for name, expected in config["code_sha256"].items():
        if sha256(resolve_path(name)) != expected:
            raise ValueError(f"Sweep code changed: {name}")
    protocol, protocol_hash, evaluation_metadata = load_evaluation_revision(
        config["parent_protocol"], config["evaluation_revision"]
    )
    seed = config["adaptation_seed"]
    source_seed = protocol["source_pretraining_seed"]
    if seed not in protocol["development_seeds"]:
        raise ValueError("Sweep seed must be a development seed.")
    settings = protocol["training"]
    feature_dir = ROOT / "data/features"
    teacher_path = resolve_path(protocol["checkpoint_dir"]) / f"unsw_to_cicids_mmd_v2_seed{seed}.pt"
    source_path = ROOT / f"models/baselines/unsw_seed{source_seed}.pt"
    source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    teacher_checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=True)
    if (teacher_checkpoint.get("protocol_sha256") != protocol_hash
            or teacher_checkpoint.get("method") != "hda_shared_semantic_hidden_mmd"
            or teacher_checkpoint["seed"] != seed
            or teacher_checkpoint.get("source_seed", seed) != source_seed):
        raise ValueError("Teacher does not match the frozen parent protocol/seed.")
    source_dim, target_dim = source_checkpoint["input_dim"], teacher_checkpoint["target_dim"]
    if teacher_checkpoint["source_dim"] != source_dim:
        raise ValueError("Source dimension mismatch.")
    threshold_path = ROOT / f"results/baseline/unsw_seed{source_seed}.json"
    threshold = json.loads(threshold_path.read_text())["threshold"]
    reference_path = resolve_path(config["reference"])
    reference = json.loads(reference_path.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    output = resolve_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "sweep_id": config["sweep_id"], "sweep_sha256": sha256(config_path),
        "parent_protocol_sha256": protocol_hash,
        "source_checkpoint_sha256": sha256(source_path),
        "teacher_checkpoint_sha256": sha256(teacher_path),
        "threshold_file_sha256": sha256(threshold_path),
        "reference_sha256": sha256(reference_path),
        "phase": "development", "seed": seed, "source_seed": source_seed,
        "target_data": str(evaluation_target(protocol, "development")),
        **evaluation_metadata,
    }
    # Resume only completed runs from the identical sweep and input artifacts.
    manifest = output / "provenance.json"
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise ValueError("Output belongs to a different sweep/input snapshot.")
    manifest.write_text(json.dumps(provenance, indent=2) + "\n")
    rows = []
    for weight in config["lambda_conditional"]:
        tag = f"lambda_{weight:.2f}_seed{seed}"
        result_path = output / f"{tag}.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            rows.append(result)
            print(f"Resume: {tag} already complete", flush=True)
            continue
        # Reproduce V4 initialization and RNG order independently for each weight.
        set_seed(seed)
        source = BaselineMLP(source_dim).to(device)
        source.load_state_dict(source_checkpoint["model_state_dict"])
        teacher = HDAV1Model(target_dim, source).to(device)
        teacher.adapter.load_state_dict(teacher_checkpoint["target_adapter_state_dict"])
        teacher.eval()
        print(f"V5a {tag} | device={device}", flush=True)
        normal, attack, pseudo = build_pseudo_pools(
            teacher, make_teacher_loader(resolve_path(protocol["target_data"]["adaptation_train"]),
                                         target_dim, settings["batch_size"]), device
        )
        source_pools = build_source_pools(
            source, make_loader(feature_dir / "unsw_train", source_dim, settings["batch_size"]), device
        )
        student, history = train_hda_v4(
            source, teacher,
            make_loader(feature_dir / "unsw_train", source_dim, settings["batch_size"], training=True),
            make_unlabeled_loader(resolve_path(protocol["target_data"]["adaptation_train"]),
                                  target_dim, settings["batch_size"]),
            source_pools, {0: normal, 1: attack}, epochs=settings["epochs"],
            lr=settings["learning_rate"], class_batch_size=settings["class_batch_size"],
            lambda_conditional=weight,
        )
        torch.save({
            **provenance, "version": "v5a", "lambda_conditional": weight,
            "source_dim": source_dim, "target_dim": target_dim,
            "target_labels_used": False, "pseudo_label_metadata": pseudo,
            "training": {**settings, "lambda_conditional": weight}, "history": history,
            "target_adapter_state_dict": {k: v.detach().cpu() for k, v in student.adapter.state_dict().items()},
        }, output / f"{tag}.pt")
        # Development labels are first read after the final training epoch.
        student.cpu().eval()
        labels, scores = collect_scores(student, make_loader(
            evaluation_target(protocol, "development"), target_dim,
            settings["batch_size"], training=False,
        ))
        metrics = compute_metrics(labels, scores, threshold)
        oracle_threshold = select_f1_threshold(labels, scores)
        oracle = compute_metrics(labels, scores, oracle_threshold)
        print(f"DEV ORACLE ONLY | threshold={oracle_threshold:.6f} | F1={oracle['f1']:.6f}", flush=True)
        result = {**provenance, "lambda_conditional": weight, **metrics,
                  "delta_vs_frozen_v4": {key: metrics[key] - value for key, value in reference["metrics"].items()}}
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        rows.append(result)
        print(f"Saved {result_path} | {metrics}", flush=True)
    fields = ["lambda_conditional", "pr_auc", "roc_auc", "f1", "recall", "fpr"]
    with (output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", *fields])
        writer.writeheader()
        writer.writerow({"run": "frozen_v4_reference", "lambda_conditional": 1.0, **reference["metrics"]})
        for row in rows:
            writer.writerow({"run": "v5a", **{key: row[key] for key in fields}})
    print(f"Comparison: {output / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
