import argparse
import json

import torch

from evaluation.baseline import collect_scores, compute_metrics, select_f1_threshold
from models.hda_v1 import HDAV1Model
from training.hda_v5b import ROOT, code_hashes, file_hash, load_models, load_setup
from training.thesis_protocol import evaluation_target, resolve_path
from training.v6_data import make_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hda_v5b_rank_0p10_symmetric.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config, config_hash, protocol, protocol_hash = load_setup(args.config)
    path = resolve_path(config["checkpoint_dir"]) / f"v5b_seed{args.seed}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint["config_sha256"] != config_hash
            or checkpoint["protocol_sha256"] != protocol_hash
            or checkpoint["code_sha256"] != code_hashes()
            or checkpoint["seed"] != args.seed or checkpoint.get("version") != "v5b" or checkpoint.get("architecture") != "hda_v1"):
        raise ValueError("V5b checkpoint config, code, architecture or seed mismatch")
    source, teacher, provenance = load_models(protocol, protocol_hash, args.seed, torch.device("cpu"))
    for key, value in provenance.items():
        if checkpoint[key] != value:
            raise ValueError(f"V5b checkpoint dependency changed: {key}")
    model = HDAV1Model(provenance["target_dim"], source)
    model.adapter.load_state_dict(checkpoint["target_adapter_state_dict"])
    target_path = evaluation_target(protocol, "development")
    loader = make_loader(target_path, provenance["target_dim"], protocol["training"]["batch_size"])
    labels, scores = collect_scores(model, loader)
    threshold = checkpoint["threshold_source"]
    metrics = compute_metrics(labels, scores, threshold)
    teacher_labels, teacher_scores = collect_scores(teacher, loader)
    teacher_metrics = compute_metrics(teacher_labels, teacher_scores, threshold)
    oracle_threshold = select_f1_threshold(labels, scores)
    oracle = compute_metrics(labels, scores, oracle_threshold)
    print(f"DEV ORACLE ONLY | threshold={oracle_threshold:.6f} | "
          f"F1={oracle['f1']:.6f} | Recall={oracle['recall']:.6f} | FPR={oracle['fpr']:.6f}")
    reference_path = ROOT / "configs/hda_v4_development_reference.json"
    reference = json.loads(reference_path.read_text())
    metric_names = ["pr_auc", "roc_auc", "f1", "recall", "fpr"]
    result = {
        "experiment_id": config["experiment_id"], "version": "v5b", "seed": args.seed,
        "source_seed": provenance["source_seed"], "phase": "development",
        "target_data": str(target_path), "target_labels_train": 0,
        "loss_weights": checkpoint["loss_weights"],
        "config_sha256": config_hash, "protocol_sha256": protocol_hash,
        "checkpoint_sha256": file_hash(path), "training_code_sha256": checkpoint["code_sha256"],
        "evaluation_sha256": file_hash(ROOT / "src/evaluation/hda_v5b.py"),
        "threshold_source": threshold, "teacher_metrics": teacher_metrics,
        "delta_vs_v2": {key: metrics[key] - teacher_metrics[key] for key in metric_names},
        "delta_vs_frozen_v4": {key: metrics[key] - reference["metrics"][key] for key in metric_names},
        **metrics,
    }
    if args.seed == 42:
        v5_path = ROOT / "results/thesis/hda_v5a_conditional_weight_seed42/lambda_0.10_seed42.json"
    else:
        v5_path = ROOT / f"results/thesis/hda_v5a_lambda_0p10_seed{args.seed}/lambda_0.10_seed{args.seed}.json"
    if v5_path.exists():
        v5 = json.loads(v5_path.read_text())
        if (v5.get("parent_protocol_sha256") == protocol_hash
                and v5.get("teacher_checkpoint_sha256") == provenance["teacher_checkpoint_sha256"]
                and v5.get("source_checkpoint_sha256") == provenance["source_checkpoint_sha256"]
                and v5.get("lambda_conditional") == 0.10
                and v5.get("seed") == args.seed
                and v5.get("threshold") == threshold
                and v5.get("phase") == "development"
                and v5.get("target_data") == str(target_path)):
            result["v5a_metrics"] = {key: v5[key] for key in metric_names}
            result["v5a_reference_sha256"] = file_hash(v5_path)
            result["delta_vs_v5a_0p10"] = {key: metrics[key] - v5[key] for key in metric_names}
    output = resolve_path(config["result_dir"]) / f"v5b_seed{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    for name, values in (("V2 teacher", teacher_metrics), ("V5b", metrics)):
        print(name + " | " + " | ".join(f"{key}={values[key]:.6f}" for key in metric_names))
    if "v5a_metrics" in result:
        print("V5a lambda=0.10 | " + " | ".join(f"{key}={result['v5a_metrics'][key]:.6f}" for key in metric_names))
    else:
        print("V5a comparison unavailable: no matching development result for this seed and protocol.")
    if "delta_vs_v5a_0p10" in result:
        print(f"Delta vs V5a lambda=0.10: {result['delta_vs_v5a_0p10']}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
