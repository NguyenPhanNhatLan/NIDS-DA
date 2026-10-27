import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from project_utils.result_files import result_file, publish_results

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from evaluation.proposal_development_lock import require_development_lock
from evaluation.proposal_negative_transfer import load_models, sample_labeled, bn_shift
from experiments.proposal_protocol import ROOT, DIRECTIONS, SEEDS, METHOD_CONFIGS
from project_utils import domains, sha256
from training.common import balanced_cross_entropy, freeze_bn_stats
from features.parquet_vectors import vector_matrix
from training.data_revision import revision_manifest, revision_path, verify_revision
from training.proposal_class_aware import confidence_setting, prepare_class_batch
from training.proposal_data import ParquetBatchStream
from training.proposal_mmd import compute_alignment_loss, load_config

from evaluation.plot_style import METHOD_LABELS, DIRECTION_LABELS

FEATURE_ROOT = revision_path("feature_root", None)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    values = np.array([value for value in values if value is not None and math.isfinite(value)])
    return {
        "mean": float(values.mean()) if len(values) else None,
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        "n": len(values),
    }


def cell(values, decimals=4, scale=1):
    result = stats(values)
    if not result["n"]:
        return "NA (0/5)"
    mean = f"{result['mean'] * scale:.{decimals}f}"
    sd = f"{result['sd'] * scale:.{decimals}f}" if result["sd"] is not None else "NA"
    suffix = f" (n={result['n']}/5)" if result["n"] != 5 else ""
    return f"{mean} ± {sd}{suffix}"


def markdown_table(title, rows, note):
    columns = list(rows[0])
    lines = [f"## {title}", "", "| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows]
    return "\n".join(lines) + "\n\n" + note + "\n"


def canonical_knn(folder, locked_hashes, k):
    manifest = json.loads(result_file(folder / "manifest.json").read_text())
    if manifest["training_seeds"] != list(SEEDS) or k not in manifest["k"]:
        raise ValueError("kNN audit does not cover the locked seeds/requested k")
    if manifest["metric"] != "euclidean" or not manifest["same_fixed_flows_across_models_and_seeds"]:
        raise ValueError("Expected fixed, class-balanced Euclidean kNN samples")
    for record in manifest["provenance"]:
        if "method" in record:
            for role in ("source_checkpoint", "adapted_checkpoint"):
                path = Path(record[role]).resolve()
                if locked_hashes.get(path) != record[role + "_sha256"]:
                    raise ValueError(f"kNN model does not match development lock: {path}")
        else:
            for role in ("source_files", "target_files"):
                for artifact in record[role]:
                    if sha256(artifact["path"]) != artifact["sha256"]:
                        raise ValueError("kNN sample came from different prepared data")
    with result_file(folder / "knn_runs.csv").open(encoding="utf-8-sig") as stream:
        raw = [row for row in csv.DictReader(stream) if int(row["k"]) == k and row["class_group"] != "Macro"]
    lookup = {}
    for row in raw:
        task = "target_to_source" if row["comparison"] == "cross_domain" else (
            "target_within" if row["query_role"] == "Target" else "source_within")
        key = (row["direction"], row["method"], int(row["training_seed"]), task, row["class_group"])
        if key in lookup:
            raise ValueError(f"Duplicate kNN observation: {key}")
        if int(row["n_query"]) != manifest["per_domain_class"]:
            raise ValueError("kNN class sample count differs from manifest")
        lookup[key] = float(row["mixing"])
    rows = []
    for direction in DIRECTIONS:
        for method in METHOD_CONFIGS:
            for seed in SEEDS:
                for task in ("target_within", "target_to_source", "source_within"):
                    for label in ("Benign", "Attack"):
                        key = (direction, method, seed, task, label)
                        before = lookup[direction, "source_only", seed, task, label]
                        after = lookup[key]
                        if not all(0 <= value <= 1 for value in (before, after)):
                            raise ValueError("Invalid kNN mixing")
                        rows.append(dict(direction=direction, method=method, seed=seed,
                                         task=task, class_label="Normal" if label == "Benign" else "Attack",
                                         k=k, per_class=manifest["per_domain_class"],
                                         before=before, after=after, delta=after-before))
    return rows, manifest


@torch.inference_mode()
def predict_validation(model, directory):
    labels, probabilities = [], []
    model.eval()
    for features, batch_labels in ParquetBatchStream(directory, 1024, False, 0, True):
        _, logits = model(features)
        probabilities.append(logits.softmax(1).numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def pseudo_rows(probabilities, labels, direction, method, seed, phase, thresholds, training_threshold):
    predicted = probabilities.argmax(1)
    confidence = probabilities.max(1)
    rows = []
    true_attacks = int(np.sum(labels == 1))
    for threshold in thresholds:
        accepted = confidence >= threshold
        pseudo_attack = accepted & (predicted == 1)
        accepted_attack = int(pseudo_attack.sum())
        correct = int(np.sum(pseudo_attack & (labels == 1)))
        true_normals = len(labels) - true_attacks
        pseudo_normal = accepted & (predicted == 0)
        accepted_normal = int(pseudo_normal.sum())
        correct_normal = int(np.sum(pseudo_normal & (labels == 0)))
        rows.append(dict(direction=direction, method=method, seed=seed, phase=phase,
                         threshold=threshold, is_training_threshold=training_threshold == threshold,
                         sample_rows=len(labels), true_attacks=true_attacks, accepted=int(accepted.sum()),
                         coverage=float(accepted.mean()), accepted_attack=accepted_attack,
                         correct_attack=correct,
                         attack_precision=correct/accepted_attack if accepted_attack else None,
                         attack_recall=correct/true_attacks,
                         true_normals=true_normals, accepted_normal=accepted_normal,
                         correct_normal=correct_normal,
                         normal_precision=correct_normal/accepted_normal if accepted_normal else None,
                         normal_recall=correct_normal/true_normals))
    return rows


@torch.inference_mode()
def activation_means(model, features):
    sums = {name: None for name in ("bn1", "bn2")}
    hooks = []
    for name in sums:
        def capture(module, args, name=name):
            part = args[0].double().sum(0).cpu()
            sums[name] = part if sums[name] is None else sums[name] + part
        hooks.append(getattr(model, name).register_forward_pre_hook(capture))
    model.eval()
    try:
        for start in range(0, len(features), 1024):
            model(torch.from_numpy(features[start:start+1024]))
    finally:
        for hook in hooks:
            hook.remove()
    return {name: (value/len(features)).numpy() for name, value in sums.items()}


def measure_bn(model, source_x, target_x, direction, method, seed):
    shifts = bn_shift(model, activation_means(model, source_x), activation_means(model, target_x))
    rows = []
    for layer, values in shifts.items():
        source, target = values["source_shift"], values["target_shift"]
        ratio = target/source if source > 0 else None
        rows.append(dict(direction=direction, method=method, seed=seed, layer=layer,
                         source_shift=source, target_shift=target, target_source_ratio=ratio,
                         flag=target > 2*source and target-source > 0.5))
    return rows


def sample_training_rows(directory, rows, seed, include_labels):
    files = sorted(directory.glob("*.parquet"))
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    if total < rows:
        raise ValueError("Not enough distinct training rows for gradient audit")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(total, rows, replace=False))
    features, labels = [], []
    offset = 0
    columns = ["features", "label"] if include_labels else ["features"]
    for path in files:
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=65536, columns=columns):
                left, right = np.searchsorted(selected, [offset, offset+len(batch)])
                if right > left:
                    indices = selected[left:right]-offset
                    features.append(vector_matrix(batch.column("features"), path)[indices])
                    if include_labels:
                        labels.append(batch.column("label").to_numpy(zero_copy_only=False)[indices])
                offset += len(batch)
    order = rng.permutation(rows)
    x = torch.from_numpy(np.concatenate(features)[order].copy())
    y = torch.from_numpy(np.concatenate(labels)[order].astype(np.int64)) if include_labels else None
    return x, y


def gradient_batches(base, source, target, batch_size, batches, sampling_seed):
    rows = batch_size*batches
    source_x, source_y = sample_training_rows(base/f"{source}_train", rows, sampling_seed, True)
    target_x, _ = sample_training_rows(base/f"{target}_train", rows, sampling_seed+1, False)
    return [(source_x[start:start+batch_size], source_y[start:start+batch_size], target_x[start:start+batch_size])
            for start in range(0, rows, batch_size)]


def measure_gradients(model, config, pairs, counts, sampling_seed):
    parameters = [*model.fc1.parameters(), *model.bn1.parameters(), *model.fc2.parameters(), *model.bn2.parameters()]
    criterion = balanced_cross_entropy(counts, torch.device("cpu"))
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    flags = {module: module.training for module in model.modules()}
    model.train()
    freeze_bn_stats(model)
    rows = []
    try:
        for index, (source_x, source_y, target_x) in enumerate(pairs):
            torch.manual_seed(sampling_seed+index)
            n = len(source_x)
            latent, logits = model(torch.cat([source_x, target_x]))
            ce = float(config["alpha_ce"])*criterion(logits[:n], source_y)
            class_batch = None
            if config["method"] == "class_aware_mmd":
                class_batch = prepare_class_batch(source_y, logits[n:], confidence_setting(config))
            alignment = compute_alignment_loss(latent[:n], source_y, latent[n:], logits[n:], config, batch=class_batch)[0]
            alignment = float(config["lambda_mmd"])*alignment
            g_ce = torch.autograd.grad(ce, parameters, retain_graph=True, allow_unused=True)
            g_alignment = torch.autograd.grad(alignment, parameters, allow_unused=True)
            def flatten(gradients):
                return torch.cat([torch.zeros_like(p).flatten() if gradient is None else gradient.flatten()
                                  for p, gradient in zip(parameters, gradients)])
            left, right = flatten(g_ce), flatten(g_alignment)
            ce_norm, align_norm = float(left.norm()), float(right.norm())
            eligible = class_batch is None or class_batch.can_align
            reason = "class_ineligible" if not eligible else (
                "zero_alignment_gradient" if align_norm <= 1e-12 else (
                    "zero_classification_gradient" if ce_norm <= 1e-12 else "valid"))
            cosine = float(F.cosine_similarity(left, right, dim=0, eps=1e-12)) if reason == "valid" else None
            if cosine is not None and not math.isfinite(cosine):
                raise ValueError("Nonfinite gradient cosine")
            rows.append(dict(batch=index, status=reason, cosine=cosine, ce_norm=ce_norm,
                             alignment_norm=align_norm, alignment_loss=float(alignment.detach()),
                             source_normal=int((source_y==0).sum()), source_attack=int((source_y==1).sum()),
                             accepted_pseudo_normal=class_batch.accepted_counts[0] if class_batch is not None else None,
                             accepted_pseudo_attack=class_batch.accepted_counts[1] if class_batch is not None else None))
    finally:
        for module, flag in flags.items():
            module.training = flag
    if not all(torch.equal(value, original_state[name]) for name, value in model.state_dict().items()):
        raise ValueError("Gradient audit changed model parameters or BatchNorm buffers")
    valid = [row["cosine"] for row in rows if row["status"] == "valid"]
    return dict(attempted_batches=len(rows), valid_batches=len(valid),
                class_ineligible_batches=sum(row["status"] == "class_ineligible" for row in rows),
                zero_alignment_batches=sum(row["status"] == "zero_alignment_gradient" for row in rows),
                negative_fraction=float(np.mean(np.array(valid)<0)) if valid else None,
                mean_cosine=float(np.mean(valid)) if valid else None), rows


def build_pseudo_tables(output, pseudo):
    canonical = [r for r in pseudo if r["method"] == "class_aware_mmd"]
    attack, normal, populations = [], [], []
    for direction in DIRECTIONS:
        population = {(r["sample_rows"], r["true_attacks"]) for r in canonical if r["direction"] == direction}
        if len(population) != 1:
            raise ValueError("Pseudo-label audit must use the same full validation population")
        n, attacks = population.pop()
        populations.append(f"{DIRECTION_LABELS[direction]}: N={n:,}, true Attack={attacks:,} ({100*attacks/n:.3f}%)")
        for phase in ("before", "after"):
            group = [r for r in canonical if r["direction"] == direction and r["phase"] == phase and r["is_training_threshold"]]
            if sorted(r["seed"] for r in group) != list(SEEDS) or len({r["threshold"] for r in group}) != 1:
                raise ValueError("Require five seeds at the configured training confidence")
            base = {"Direction": DIRECTION_LABELS[direction], "Model stage": phase.title(),
                    "Threshold": f"{group[0]['threshold']:.2f}",
                    "Coverage (%)": cell([r["coverage"] for r in group], 2, 100)}
            attack.append({**base,
                "Accepted pseudo-Attack (mean/seed)": cell([r["accepted_attack"] for r in group], 1),
                "Attack precision (%)": cell([r["attack_precision"] for r in group], 2, 100),
                "Attack recall (%)": cell([r["attack_recall"] for r in group], 2, 100)})
            normal.append({**base,
                "Accepted pseudo-Normal (mean/seed)": cell([r["accepted_normal"] for r in group], 1),
                "Normal precision (%)": cell([r["normal_precision"] for r in group], 2, 100),
                "Normal recall (%)": cell([r["normal_recall"] for r in group], 2, 100)})
    write_csv(output/"table2_pseudo_labels_training_threshold.csv", attack)
    write_csv(output/"appendix_pseudo_normal_training_threshold.csv", normal)
    normal_fields = ("direction", "method", "seed", "phase", "threshold", "is_training_threshold",
                     "sample_rows", "true_normals", "accepted_normal", "correct_normal", "normal_precision", "normal_recall")
    write_csv(output/"appendix_pseudo_normal_runs.csv", [{key:r[key] for key in normal_fields} for r in canonical])
    note = ("Canonical Class-Aware MMD; configured confidence τ=0.80. Post-training audit on FULL target validation, "
            "fixed saved models in eval mode; not online training label errors. Before: source-only. After: selected model. "
            "Unweighted mean ± sample SD (ddof=1), seeds 42–46; accepted counts are mean flows PER SEED, not totals. "
            "Precision and recall are calculated per seed before averaging; undefined precision is excluded with valid n shown. "
            "Coverage = all accepted / all target rows. Class precision = correct accepted pseudo-class / accepted pseudo-class. "
            "Class recall = correct accepted pseudo-class / all true members of that class, including rejected rows. "
            + "; ".join(populations) + ".")
    return [markdown_table("Table 2 — Target pseudo-label quality at the configured confidence threshold", attack, note),
            markdown_table("Appendix — Pseudo-Normal quality at the configured confidence threshold", normal, note)]


def build_tables(output, knn, pseudo, gradient, bn, composition, threshold):
    sections = []
    table1 = []
    for direction in DIRECTIONS:
        for method in METHOD_CONFIGS:
            row = {"Direction": DIRECTION_LABELS[direction], "Method": METHOD_LABELS[method]}
            for task, title in (("target_within", "Target"),):
                for label in ("Normal", "Attack"):
                    group = [r for r in knn if r["direction"] == direction and r["method"] == method and r["task"] == task and r["class_label"] == label]
                    row[f"{title} {label} mixing before → after"] = cell([r["before"] for r in group])+" → "+cell([r["after"] for r in group])
            table1.append(row)
    write_csv(output/"table1_class_structure.csv", table1)
    sections.append(markdown_table("Table 1 — Within-target class structure", table1,
        "Mean ± sample SD (ddof=1), five paired training seeds, mixing in [0,1]. Fixed class-balanced validation samples; mixing is the fraction of opposite-class neighbors, not an MMD separation ratio."))

    sections.extend(build_pseudo_tables(output, pseudo))

    table3a = [{"Dataset": r["dataset"], "Split/sample": r["population"], "Direction/role": r["role"],
                "Normal": r["normal"], "Attack": r["attack"], "Attack (%)": f"{100*r['attack']/(r['normal']+r['attack']):.3f}"} for r in composition]
    write_csv(output/"table3a_class_composition.csv", table3a)
    sections.append(markdown_table("Table 3A — Class composition", table3a,
        "Full validation is used for pseudo-label/threshold audits. Training rows supply gradient batches. BN uses fixed natural-prevalence validation subsamples (sampling seed 42 for source, 43 for target); these counts are reported separately. kNN uses separate balanced samples stated in Figure 1."))
    table3b = []
    for direction in DIRECTIONS:
        for method in ("source_only", *METHOD_CONFIGS):
            for layer in ("bn1", "bn2"):
                group = [r for r in bn if r["direction"] == direction and r["method"] == method and r["layer"] == layer]
                table3b.append({"Direction": DIRECTION_LABELS[direction], "Method": METHOD_LABELS[method], "BN layer": layer,
                                "Source shift": cell([r["source_shift"] for r in group]),
                                "Target shift": cell([r["target_shift"] for r in group]),
                                "Target/source ratio": cell([r["target_source_ratio"] for r in group]),
                                "Heuristic flags / 5": sum(r["flag"] for r in group)})
    write_csv(output/"table3b_normalization.csv", table3b)
    sections.append(markdown_table("Table 3B — BatchNorm mismatch", table3b,
        "Shift(D,l) = mean_j |mean_D(pre-BN activation_j) − running_mean_lj| / sqrt(running_var_lj + eps_l). Evaluate saved models with dropout off and unchanged BN running statistics. Report per-seed mean ± sample SD; the displayed ratio summarizes the per-seed target/source ratios. A flag requires target > 2×source AND target−source > 0.5. These are study heuristics, not significance thresholds. No BN-statistics ablation was run; this table does not estimate an effect on AP."))

    table4 = []
    for direction in DIRECTIONS:
        for method in ("source_only", *METHOD_CONFIGS):
            group = [r for r in threshold if r["direction"] == direction and r["method"] == method]
            table4.append({"Direction": DIRECTION_LABELS[direction], "Method": METHOD_LABELS[method],
                           "Attack F1 at source threshold": cell([r["source_f1"] for r in group]),
                           "Target-validation optimal Attack F1": cell([r["oracle_f1"] for r in group]),
                           "F1 gain": cell([r["f1_gain"] for r in group]), "Target AP": cell([r["target_ap"] for r in group])})
    write_csv(output/"table4_threshold_transfer.csv", table4)
    sections.append(markdown_table("Table 4 — Threshold transfer and remaining detection loss", table4,
        "Mean ± sample SD across the SAME five seeds. F1 is binary Attack-class F1, never Macro-F1. Source thresholds maximize Attack F1 on source validation. The target-optimal threshold uses target-validation labels for diagnosis only and is not a deployed/tuned threshold. AP is calculated from the same freshly evaluated full-validation scores as both F1 values and is unchanged by replacing the decision threshold. The maximizing threshold and the two F1 values for every run are in appendix_threshold_runs.csv. This is an in-sample target-validation optimum, not held-out performance."))
    (output/"tables.md").write_text("# Canonical mechanism diagnostics\n\n"+"\n".join(sections))


def threshold_row(probabilities, labels, source_threshold, direction, method, seed):
    from evaluation.baseline import select_f1_threshold
    from sklearn.metrics import average_precision_score

    scores = probabilities[:, 1]
    oracle_threshold = select_f1_threshold(labels, scores)
    def attack_f1(threshold):
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum(~predicted & (labels == 1)))
        return 2*tp/(2*tp+fp+fn)
    source_f1, oracle_f1 = attack_f1(source_threshold), attack_f1(oracle_threshold)
    if oracle_f1 + 1e-12 < source_f1:
        raise ValueError("Target-validation optimum is below source-threshold F1")
    return dict(direction=direction, method=method, seed=seed,
                source_threshold=source_threshold, target_diagnostic_threshold=oracle_threshold,
                source_f1=source_f1, oracle_f1=oracle_f1, f1_gain=oracle_f1-source_f1,
                target_ap=float(average_precision_score(labels, scores)))


def run(args):
    if result_file(args.output / "audit_manifest.json").exists() or args.output.exists():
        raise FileExistsError(f"Choose a new audit directory: {args.output}")
    revision = verify_revision()
    lock = require_development_lock(revision)
    locked_hashes = {Path(row["path"]).resolve(): row["sha256"] for row in lock["artifacts"]}
    manifest = revision_manifest()
    knn, knn_manifest = canonical_knn(args.knn_root, locked_hashes, args.k)
    count_lookup = defaultdict(lambda: [0, 0])
    for row in manifest["counts"]["class_counts"]:
        count_lookup[row["domain"], row["split"]][int(row["label"])] += int(row["count"])
    composition = []
    for domain in ("unsw", "cicids"):
        for split in ("train", "val"):
            normal, attack = count_lookup[domain, split]
            composition.append(dict(dataset=domain, population=f"full {split}", role="both directions",
                                    normal=normal, attack=attack))
    pseudo, gradients, gradient_batch_rows, bn, thresholds, provenance = [], [], [], [], [], []
    args.output.mkdir(parents=True, exist_ok=False)
    gradient_sampling_seed = args.sampling_seed+1000
    sampling = []
    for direction in DIRECTIONS:
        source, target = domains(direction)
        base = FEATURE_ROOT/direction
        source_x, source_y = sample_labeled(base/f"{source}_val", args.bn_sample_rows, args.sampling_seed)
        target_x, target_y = sample_labeled(base/f"{target}_val", args.bn_sample_rows, args.sampling_seed+1)
        for role, domain, labels in (("source", source, source_y), ("target", target, target_y)):
            composition.append(dict(dataset=domain, population="BN fixed validation sample", role=f"{direction}/{role}",
                                    normal=int(np.sum(labels==0)), attack=int(np.sum(labels==1))))
        pairs = gradient_batches(base, source, target, args.batch_size, args.gradient_batches, gradient_sampling_seed)
        import hashlib
        batch_hash = hashlib.sha256()
        for pair in pairs:
            for tensor in pair:
                batch_hash.update(tensor.numpy().tobytes())
        sampling.append(dict(direction=direction, source_bn_rows=len(source_y), target_bn_rows=len(target_y),
                             gradient_pairs_sha256=batch_hash.hexdigest()))
        for seed in SEEDS:
            baseline_probabilities = baseline_labels = baseline_metrics = None
            for method, config_path in METHOD_CONFIGS.items():
                config = load_config(config_path)
                if config["training"]["batch_size"] != args.batch_size:
                    raise ValueError("Use the actual training batch size for gradient measurements")
                baseline, adapted, source_result, adapted_result, paths = load_models(direction, seed, config_path)
                for artifact in (source_result, adapted_result):
                    if any(artifact.get(key) != value for key, value in revision.items()):
                        raise ValueError("Model result belongs to another data revision")
                if source_result["source_train_counts"] != count_lookup[source, "train"]:
                    raise ValueError("Source loss class weights differ from canonical training counts")
                provenance.append(dict(direction=direction, method=method, seed=seed, **paths,
                                       config=str(config_path), config_sha256=sha256(config_path),
                                       selected_stage=adapted_result["selected_stage"], best_epoch=adapted_result["best_epoch"]))
                if baseline_probabilities is None:
                    baseline_probabilities, baseline_labels = predict_validation(baseline, base/f"{target}_val")
                    expected = count_lookup[target, "val"]
                    if np.bincount(baseline_labels, minlength=2).tolist() != expected:
                        raise ValueError("Target validation class counts differ from manifest")
                    baseline_metrics = threshold_row(baseline_probabilities, baseline_labels,
                                                     source_result["threshold_from_source_val"], direction, "source_only", seed)
                    thresholds.append(baseline_metrics)
                    bn.extend(measure_bn(baseline, source_x, target_x, direction, "source_only", seed))
                    if abs(baseline_metrics["target_ap"]-source_result["target_development"]["pr_auc"]) > 1e-3:
                        raise ValueError("Re-evaluated baseline AP differs from saved result")
                if adapted_result["selected_stage"] == "source_pretrained":
                    if not all(torch.equal(value, baseline.state_dict()[name]) for name, value in adapted.state_dict().items()):
                        raise ValueError("Retained source-only model differs from source weights")
                    probabilities, labels = baseline_probabilities, baseline_labels
                else:
                    probabilities, labels = predict_validation(adapted, base/f"{target}_val")
                    if not np.array_equal(labels, baseline_labels):
                        raise ValueError("Methods use different target-validation rows")
                adapted_metrics = threshold_row(probabilities, labels, adapted_result["threshold"], direction, method, seed)
                if abs(adapted_metrics["target_ap"]-adapted_result["cross_domain"]["pr_auc"]) > 1e-3:
                    raise ValueError("Re-evaluated adapted AP differs from saved result")
                thresholds.append(adapted_metrics)
                training_threshold = confidence_setting(config) if method == "class_aware_mmd" else None
                if isinstance(training_threshold, dict):
                    raise ValueError("This canonical figure expects a scalar training confidence")
                levels = sorted(set([0.80, 0.90, 0.95]+([training_threshold] if training_threshold is not None else [])))
                for phase, values in (("before", baseline_probabilities), ("after", probabilities)):
                    pseudo.extend(pseudo_rows(values, labels, direction, method, seed, phase, levels, training_threshold))
                bn.extend(measure_bn(adapted, source_x, target_x, direction, method, seed))
                summary, batches = measure_gradients(adapted, config, pairs, source_result["source_train_counts"], gradient_sampling_seed)
                gradient_run = dict(direction=direction, method=method, seed=seed,
                                    selected_stage=adapted_result["selected_stage"], best_epoch=adapted_result["best_epoch"],
                                    alignment_kernel=config["mmd"]["kernel"],
                                    confidence=training_threshold,
                                    delta_ap=adapted_metrics["target_ap"]-baseline_metrics["target_ap"],
                                    source_only_ap=baseline_metrics["target_ap"], adapted_ap=adapted_metrics["target_ap"],
                                    **summary)
                gradients.append(gradient_run)
                gradient_batch_rows.extend(dict(direction=direction, method=method, seed=seed, **row) for row in batches)
                print(f"{direction} {method} seed{seed}: usable gradient batches {summary['valid_batches']}/{summary['attempted_batches']}", flush=True)

                write_csv(args.output/"appendix_gradient_runs.csv", gradients)
                write_csv(args.output/"appendix_pseudo_label_runs.csv", pseudo)
    if len(gradients) != 30 or len(thresholds) != 40 or len(bn) != 80 or len(pseudo) != 180:
        raise ValueError("Incomplete canonical diagnostic matrix")
    for filename, rows in (
        ("knn_runs.csv", [row for row in knn if row["task"] != "source_within"]),
        ("appendix_knn_source_within.csv", [row for row in knn if row["task"] == "source_within"]),
        ("appendix_pseudo_label_runs.csv", pseudo), ("appendix_gradient_runs.csv", gradients),
        ("appendix_gradient_batches.csv", gradient_batch_rows), ("appendix_bn_runs.csv", bn),
        ("appendix_threshold_runs.csv", thresholds),
    ):
        write_csv(args.output/filename, rows)
    build_tables(args.output, knn, pseudo, gradients, bn, composition, thresholds)
    audit_manifest = dict(
        **revision, status="complete", scope="canonical_only", training_seeds=list(SEEDS),
        methods=list(METHOD_CONFIGS), target_test_read=False, model_parameters_updated=False,
        training_configs={method:dict(path=str(path), sha256=sha256(path)) for method,path in METHOD_CONFIGS.items()},
        knn=dict(input=str(args.knn_root.resolve()), input_sha256=sha256(args.knn_root/"knn_runs.csv"),
                 manifest_sha256=sha256(args.knn_root/"manifest.json"), k=args.k,
                 per_class=knn_manifest["per_domain_class"], sampling_seed=knn_manifest["sampling_seed"],
                 metric="euclidean in original latent space", self_row_excluded=True, other_duplicate_flows_retained=True),
        pseudo_labels=dict(population="full target validation, natural prevalence", phases=["before","after"],
                           thresholds=[.8,.9,.95], training_methods=["class_aware_mmd"],
                           mode="fixed saved model, eval mode; not online training label errors",
                           aggregate="unweighted per-seed mean and sample SD; undefined precision excluded with valid n"),
        gradient=dict(objective="actual configured alignment loss, same function called by training",
                      classification="source class-balanced cross-entropy, multiplied by alpha_ce",
                      alignment="configured loss multiplied by lambda_mmd", parameters="shared encoder fc1,bn1,fc2,bn2; excludes classifier",
                      checkpoint="selected saved model, best source-validation AP including epoch 0",
                      batches=args.gradient_batches, batch_size=args.batch_size, sampling_seed=gradient_sampling_seed,
                      batch_selection="uniform without-replacement rows from each ENTIRE training split, then randomly permuted into batches",
                      same_pairs_across_methods_and_seeds=True, target_train_labels_read=False,
                      model_mode="train mode with dropout active and BatchNorm statistics frozen",
                      dropout_seed="gradient_sampling_seed + batch_index, identical across models",
                      cosine_epsilon=1e-12,
                      denominator="only batches with both gradient norms > 1e-12; ineligible/zero gradients excluded and counted",
                      interpretation="fixed-model gradient audit; not a reconstructed optimization trajectory or causal AP effect"),
        bn=dict(sample_rows_requested=args.bn_sample_rows, source_sampling_seed=args.sampling_seed, target_sampling_seed=args.sampling_seed+1,
                formula="mean_j(abs(mean_D(preBN_j)-running_mean_j)/sqrt(running_var_j+eps))",
                ratio="target/source calculated within each seed before aggregation",
                heuristic="target > 2*source AND target-source > 0.5; study heuristic only", ablation_performed=False),
        threshold=dict(metric="binary Attack-class F1", population="full target validation",
                       source="saved source-validation Attack-F1 threshold",
                       oracle="target-validation F1 optimum for diagnosis only; not deployed", ap="same full-validation scores; threshold-independent"),
        statistics="five fixed training seeds; mean ± sample SD (ddof=1); describes model-run variation conditional on fixed data/samples",
        sampling=sampling, provenance=provenance,
    )
    require_development_lock(verify_revision())
    audit_manifest["outputs_sha256"] = {
        path.name: sha256(path) for path in args.output.iterdir() if path.suffix in (".csv", ".md")
    }
    audit_manifest["implementation_sha256"] = {
        str(path): sha256(path) for path in map(Path, (
            "src/evaluation/proposal_mechanism_audit.py", "src/training/proposal_mmd.py",
            "src/training/proposal_class_aware.py", "src/training/adaptation.py",
            "src/training/proposal_mkmmd.py",
        ))
    }
    (args.output/"audit_manifest.json").write_text(json.dumps(audit_manifest, indent=2, allow_nan=False)+"\n")
    publish_results(args.output)
    print("Canonical mechanism measurements saved; see results/tables and results/json.", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Canonical-only measurements and tables for thesis sections 4.5.1–4.5.5.\n\nReuses verified canonical kNN samples and measures pseudo-label quality, BN shift,\nand the actual method-specific alignment gradients without updating any model.\n')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--knn-root", type=Path, default=ROOT/"results/analysis/latent_knn_balanced_v1")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--bn-sample-rows", type=int, default=50000)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--gradient-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.gradient_batches < 1 or args.bn_sample_rows < 2:
        parser.error("Require positive batch count and BN sample size >= 2")
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
