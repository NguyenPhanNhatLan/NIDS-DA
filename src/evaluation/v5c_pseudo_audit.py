"""Audit frozen V2/V5b agreement using adaptation features only."""
import argparse
import json

import numpy as np
import torch

from evaluation.hda_v5b_calibration import data_snapshot
from training.hda_v5b import file_hash
from training.thesis_protocol import evaluation_target, resolve_path
from training.v5c_common import audit_paths, load_context, pipeline_hashes
from training.v6_data import make_teacher_loader


def analyze_margins(v2, v5b, thresholds, minimum=64, attack_confirmation=0.5):
    v2, v5b = np.asarray(v2, dtype=float), np.asarray(v5b, dtype=float)
    if v2.ndim != 1 or v2.shape != v5b.shape or not v2.size:
        raise ValueError("Expected nonempty aligned margin vectors")
    if not np.isfinite(v2).all() or not np.isfinite(v5b).all():
        raise ValueError("Nonfinite model margins")
    q02, q95, q98 = np.quantile(v2, [0.02, 0.95, 0.98])
    p2 = v2 >= thresholds["v2_margin_threshold"]
    p5 = v5b >= thresholds["v5b_margin_threshold"]
    candidates = {"normal": v2 <= q02, "attack": (v2 >= q95) & (v2 < q98)}
    accepted = {"normal": candidates["normal"] & ~p2 & ~p5,
                "attack": candidates["attack"] & p2 & p5}
    counts = np.bincount(p2.astype(int) * 2 + p5.astype(int), minlength=4).reshape(2, 2)
    observed = float((p2 == p5).mean())
    expected = float(p2.mean() * p5.mean() + (1-p2.mean()) * (1-p5.mean()))
    reasons, pools = [], {}
    for name in candidates:
        n, kept = int(candidates[name].sum()), int(accepted[name].sum())
        confirmed = int((candidates[name] & (p5 if name == "attack" else ~p5)).sum())
        pools[name] = {"candidate_rows": n, "accepted_rows": kept,
                       "v5b_confirmation_rate": confirmed / n if n else None,
                       "retention_rate": kept / n if n else None}
        if kept < minimum:
            reasons.append(f"{name}: {kept} accepted rows < {minimum}")
    rate = pools["attack"]["v5b_confirmation_rate"]
    if rate is None or rate < attack_confirmation:
        reasons.append(f"attack: V5b confirmation rate below {attack_confirmation}")
    return {"train_rows": len(v2), "agreement_rate": observed,
            "cohen_kappa": (observed-expected)/(1-expected) if expected < 1 else None,
            "confusion_rows_v2_columns_v5b_normal_attack": counts.tolist(),
            "v2_attack_rate": float(p2.mean()), "v5b_attack_rate": float(p5.mean()),
            "quantiles": {"q02": float(q02), "q95": float(q95), "q98": float(q98)},
            "pools": pools, "training_allowed": not reasons, "blocked_reasons": reasons}, accepted


def run(config_path):
    config, protocol, provenance, _, v2, v5b = load_context(config_path)
    report_path, pool_path = audit_paths(config)
    if report_path.exists() or pool_path.exists():
        raise FileExistsError(f"Audit output already exists in {report_path.parent}; use a new audit_dir")
    train_path = resolve_path(protocol["target_data"]["adaptation_train"])
    if train_path.resolve() == evaluation_target(protocol, "development").resolve():
        raise ValueError("Audit requires adaptation train, not development")
    policy = protocol["pseudo_labels"]
    if (policy["normal_quantile"], policy["attack_quantile_low"],
            policy["attack_quantile_high_exclusive"], policy["dynamic_updates"]) != (0.02, 0.95, 0.98, False):
        raise ValueError("Expected frozen q02 / q95 / q98 pseudo policy")
    snapshot = data_snapshot(train_path)
    loader = make_teacher_loader(train_path, provenance["target_dim"], protocol["training"]["batch_size"])
    margins = [[], []]
    print("Scoring V2/V5b on adaptation train (no labels)...", flush=True)
    with torch.no_grad():
        for features in loader:
            for model, parts in zip((v2, v5b), margins):
                _, logits = model(features)
                parts.append((logits[:, 1]-logits[:, 0]).cpu().numpy())
    if not margins[0]:
        raise ValueError("Empty adaptation train")
    result, masks = analyze_margins(
        *[np.concatenate(parts) for parts in margins], provenance["threshold_policy"],
        config["min_samples_per_class"], config["min_attack_confirmation_rate"],
    )
    parts = {name: [] for name in masks}
    offset = 0
    for features in loader:
        end = offset + len(features)
        for name, mask in masks.items():
            parts[name].append(features[torch.from_numpy(mask[offset:end])])
        offset = end
    if offset != result["train_rows"] or snapshot != data_snapshot(train_path):
        raise ValueError("Adaptation data changed during audit")
    pools = {name: torch.cat(values) for name, values in parts.items()}
    pools.update({name + "_row_indices": torch.from_numpy(np.flatnonzero(mask)) for name, mask in masks.items()})
    result.update(provenance=provenance, code_sha256=pipeline_hashes(),
                  target_train_files=snapshot, target_data=str(train_path), labels_used=False,
                  acceptance_rule="V2 quantile candidate AND both models predict the candidate class",
                  gate={"min_samples_per_class": config["min_samples_per_class"],
                        "min_attack_confirmation_rate": config["min_attack_confirmation_rate"]},
                  limitation="Agreement is consistency, not pseudo-label accuracy; gate is a heuristic.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with pool_path.open("xb") as stream:
        torch.save(pools, stream)
    result["pool_sha256"] = file_hash(pool_path)
    with report_path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("agreement_rate", "cohen_kappa", "pools", "training_allowed", "blocked_reasons")}, indent=2))
    print(f"Saved: {report_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hda_v5c.json")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
