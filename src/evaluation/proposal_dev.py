import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from evaluation.proposal_development_lock import require_development_lock
from evaluation.proposal_domain_shift import sample_features
from evaluation.proposal_negative_transfer import sample_labeled
from experiments.proposal_protocol import DIRECTIONS, METHOD_CONFIGS, ROOT, SEEDS
from features.common_features import COMMON_FEATURES
from models.baseline import BaselineMLP
from project_utils import domains, sha256
from training.data_revision import revision_path, verify_revision
from training.proposal_mmd import output_paths

DECAYS = (.1, .03, .01, .003, .001, .0003, .0001, .00003, .00001)


def dev_risk(weights, errors):
    weights, errors = np.asarray(weights, dtype=float), np.asarray(errors, dtype=float)
    if weights.ndim != 1 or weights.shape != errors.shape or len(weights) < 2:
        raise ValueError("DEV requires matching 1D arrays with at least two observations")
    if not np.isfinite(weights).all() or not np.isfinite(errors).all() or np.any(weights <= 0):
        raise ValueError("DEV weights must be positive and all inputs finite")
    if np.any((errors < 0) | (errors > 1)):
        raise ValueError("This trial uses bounded zero-one classification error")
    weighted = weights * errors
    variance = float(weights.var(ddof=1))
    constant = variance <= 1e-12 * max(1., float(np.mean(weights**2)))
    eta = 0. if constant else -float(np.cov(weighted, weights, ddof=1)[0, 1]) / variance
    risk = float(weighted.mean() + eta * (weights.mean() - 1.))
    corrected = weighted + eta * (weights - 1.)
    return dict(dev_risk=risk, source_error=float(errors.mean()),
                iw_risk=float(weighted.mean()), control_eta=eta,
                constant_weights=constant, corrected_sample_sd=float(corrected.std(ddof=1)),
                weight_mean=float(weights.mean()), weight_min=float(weights.min()),
                weight_p95=float(np.quantile(weights, .95)), weight_max=float(weights.max()),
                effective_sample_size=float(weights.sum()**2 / np.dot(weights, weights)))


def estimate_weights(source_z, target_z, validation_z, *, seed, decays, max_iter, probability_floor):
    n_source, n_target = len(source_z), len(target_z)
    features = np.concatenate((source_z, target_z))
    labels = np.r_[np.ones(n_source, dtype=int), np.zeros(n_target, dtype=int)]
    train, holdout = train_test_split(np.arange(len(labels)), train_size=.8,
                                    random_state=seed, stratify=labels)

    prior_ratio = float(np.sum(labels[train] == 1) / np.sum(labels[train] == 0))
    best, best_accuracy, trials = None, -1., []
    width = features.shape[1]
    for decay in decays:
        model = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(width, width, 2), activation="relu", alpha=decay,
            batch_size=min(256, len(train)), max_iter=max_iter, random_state=seed,
            early_stopping=False))
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(features[train], labels[train])
        probability = model.predict_proba(features[holdout])
        accuracy = float(accuracy_score(labels[holdout], model.predict(features[holdout])))
        report = dict(decay=float(decay), holdout_accuracy=accuracy,
                      holdout_auc=float(roc_auc_score(labels[holdout], probability[:, 1])),
                      holdout_log_loss=float(log_loss(labels[holdout], probability)),
                      converged=not any(issubclass(m.category, ConvergenceWarning) for m in messages),
                      iterations=int(model[-1].n_iter_))
        trials.append(report)
        if accuracy > best_accuracy:
            best, best_accuracy, selected = model, accuracy, report
    probability = best.predict_proba(validation_z)
    source_column = int(np.flatnonzero(best[-1].classes_ == 1)[0])
    source_probability = probability[:, source_column]
    target_probability = probability[:, 1-source_column]
    floored = (source_probability < probability_floor) | (target_probability < probability_floor)

    weights = (np.maximum(target_probability, probability_floor) /
               np.maximum(source_probability, probability_floor) * prior_ratio)
    return weights, dict(selected=selected, trials=trials, prior_ratio=prior_ratio,
                         probability_floor=probability_floor,
                         floored_fraction=float(floored.mean()),
                         train_rows=len(train), holdout_rows=len(holdout),
                         preprocessing="StandardScaler fit only on domain-classifier training rows")


@torch.inference_mode()
def embed(model, features):
    latent, logits = [], []
    model.eval()
    for start in range(0, len(features), 1024):
        z, output = model(torch.from_numpy(features[start:start+1024]).float())
        latent.append(z.numpy())
        logits.append(output.numpy())
    return np.concatenate(latent), np.concatenate(logits)


def candidates(args):
    if args.candidates:
        records = json.loads(args.candidates.read_text())
        if not isinstance(records, list) or not records:
            raise ValueError("--candidates must contain a nonempty JSON list")
        output = []
        for row in records:
            path = Path(row["checkpoint"])
            if not path.is_absolute():
                path = args.candidates.resolve().parent / path
            output.append({**row, "checkpoint": path.resolve()})
    else:
        output = []
        for direction in (DIRECTIONS if args.direction == "both" else [args.direction]):
            for seed in args.seeds:
                for method in args.methods:
                    path = (revision_path("model_root", None)/"source_only_target_val"/direction/f"seed{seed}.pt"
                            if method == "source_only" else output_paths(direction, seed, METHOD_CONFIGS[method])[0])
                    output.append(dict(name=method, direction=direction, seed=seed, checkpoint=path))
    identities = [(r["direction"], r["seed"], r["name"]) for r in output]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate candidate names within direction/seed")
    for direction, seed, name in identities:
        if direction not in DIRECTIONS or type(seed) is not int or not name:
            raise ValueError("Invalid candidate direction, seed, or name")
    return output


def load_candidate(record, revision, locked_hashes, split_hashes, preprocessor_hash):
    path = record["checkpoint"]
    digest = sha256(path)
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (data["direction"] != record["direction"] or data["seed"] != record["seed"] or
            data["features"] != list(COMMON_FEATURES) or
            any(data.get(k) != v for k, v in revision.items()) or
            data["preprocessor_sha256"] != preprocessor_hash or
            data["common_feature_config_sha256"] != sha256(ROOT/"configs/common_features_v2.json")):
        raise ValueError(f"Candidate identity/preprocessing mismatch: {path}")
    for role, expected in split_hashes.items():
        if data["prepared_split_sha256"][role] != expected:
            raise ValueError(f"Candidate split mismatch: {path}: {role}")
    if path.resolve() in locked_hashes and digest != locked_hashes[path.resolve()]:
        raise ValueError(f"Locked candidate changed: {path}")
    model = BaselineMLP(len(COMMON_FEATURES))
    model.load_state_dict(data["model_state_dict"])
    return model, data, digest


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    records = candidates(args)
    outputs = [ROOT/"results/tables"/f"{args.output_prefix}__{name}.csv" for name in ("runs", "selection")]
    manifest_path = ROOT/"results/json"/f"{args.output_prefix}__manifest.json"
    for path in [*outputs, manifest_path]:
        if path.exists():
            raise FileExistsError(f"Choose a new --output-prefix: {path}")
    revision = verify_revision()
    lock = require_development_lock(revision)
    locked_hashes = {Path(r["path"]).resolve():r["sha256"] for r in lock["artifacts"]}
    torch.set_num_threads(args.threads)
    rows, receipts, samples = [], [], []
    from training.proposal_data import split_sha256
    for direction in dict.fromkeys(r["direction"] for r in records):
        source, target = domains(direction)
        base = revision_path("feature_root", None)/direction

        sx, _, source_ids = sample_features(base/f"{source}_train", args.domain_rows,
                                            args.sampling_seed, return_indices=True)
        tx, _, target_ids = sample_features(base/f"{target}_train", args.domain_rows,
                                            args.sampling_seed+1, return_indices=True)
        vx, vy = sample_labeled(base/f"{source}_val", args.validation_rows, args.sampling_seed+2)
        split_hashes = {role:split_sha256(base/name) for role, name in (
            ("source_train",f"{source}_train"),("target_train",f"{target}_train"),("source_val",f"{source}_val"))}
        preprocessor_hash = sha256(revision_path("model_root", None)/direction/"preprocessor.joblib")
        import hashlib
        samples.append(dict(direction=direction, domain_source_rows=len(sx), domain_target_rows=len(tx),
                            validation_source_rows=len(vx), validation_attack_rows=int(vy.sum()),
                            source_indices_sha256=hashlib.sha256(source_ids.tobytes()).hexdigest(),
                            target_indices_sha256=hashlib.sha256(target_ids.tobytes()).hexdigest(),
                            split_sha256=split_hashes, preprocessor_sha256=preprocessor_hash))
        for record in [r for r in records if r["direction"] == direction]:
            model, checkpoint, digest = load_candidate(record, revision, locked_hashes,
                                                        split_hashes, preprocessor_hash)
            sz, _ = embed(model, sx)
            tz, _ = embed(model, tx)
            vz, logits = embed(model, vx)
            error = (logits.argmax(1) != vy).astype(float)
            with threadpool_limits(limits=args.threads):
                weights, domain = estimate_weights(sz, tz, vz, seed=args.sampling_seed+3,
                    decays=args.decays, max_iter=args.max_iter, probability_floor=args.probability_floor)
            risk = dev_risk(weights, error)
            flags = []
            if risk["effective_sample_size"] < .1*len(vx): flags.append("low_effective_sample_size")
            if domain["floored_fraction"]: flags.append("probability_floor_active")
            if not domain["selected"]["converged"]: flags.append("domain_classifier_not_converged")
            if not 0 <= risk["dev_risk"] <= 1: flags.append("risk_outside_0_1")
            row = dict(direction=direction, seed=record["seed"], candidate=record["name"],
                       method=checkpoint.get("method", "source_only"),
                       lambda_mmd=checkpoint.get("lambda_mmd", 0.),
                       best_epoch=checkpoint["best_epoch"], checkpoint=str(record["checkpoint"]),
                       validation_rows=len(vx), **risk,
                       domain_accuracy=domain["selected"]["holdout_accuracy"],
                       domain_auc=domain["selected"]["holdout_auc"],
                       domain_decay=domain["selected"]["decay"],
                       floored_fraction=domain["floored_fraction"], flags=";".join(flags))
            rows.append(row)
            receipts.append({**record, "checkpoint":str(record["checkpoint"]), "checkpoint_sha256":digest,
                             "domain_estimation":domain})
            print(f"{direction} seed{record['seed']} {record['name']}: DEV={risk['dev_risk']:.6f}; "
                  f"ESS={risk['effective_sample_size']:.0f}/{len(vx)}; {row['flags'] or 'no numerical flags'}", flush=True)
    selections = []
    for direction, seed in dict.fromkeys((r["direction"],r["seed"]) for r in rows):
        pool = [r for r in rows if r["direction"] == direction and r["seed"] == seed]
        winner = min(pool, key=lambda r:r["dev_risk"])
        selections.append(dict(direction=direction, seed=seed, candidate_count=len(pool),
                               selected_candidate=winner["candidate"], dev_risk=winner["dev_risk"],
                               checkpoint=winner["checkpoint"], flags=winner["flags"]))
    require_development_lock(verify_revision())
    for path in [*outputs, manifest_path]: path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(outputs[0], rows)
    write_csv(outputs[1], selections)
    manifest = dict(**revision, scope="DEV trial over saved candidates, not a new training/model-selection trajectory",
        selection="minimum raw DEV zero-one risk within direction and seed; flags retained, not filtered",
        target_labels_read=False, target_validation_read=False, target_test_read=False,
        model_parameters_updated=False, source_error="unweighted zero-one error using argmax logits, not AP or Attack F1",
        domain_classifier="reference MLP (d,d,2), alpha selected by held-out domain accuracy; train-only StandardScaler added",
        density_ratio="P(target|z)/P(source|z) * n_source_train/n_target_train",
        control_variate="eta=-Cov(w*error,w)/Var(w); DEV=mean(w*error)+eta*(mean(w)-1)",
        assumptions="covariate shift in learned features and target support covered by source; not verified automatically",
        limitation="Candidates already selected by source AP; ranking is conditional on this pool. A lambda/epoch sweep needs those additional saved candidates. DEV estimates zero-one risk, not target AP.",
        parameters={k:v for k,v in vars(args).items() if k != "candidates"}, samples=samples, candidates=receipts,
        implementation_sha256=sha256(Path(__file__)),
        sources=["https://proceedings.mlr.press/v97/you19a.html", "https://github.com/thuml/Deep-Embedded-Validation/blob/master/dev.py"],
        output_sha256={str(p.relative_to(ROOT)):sha256(p) for p in outputs})
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(f"Saved DEV trial: {outputs[0]} and {outputs[1]}")


def main():
    parser = argparse.ArgumentParser(description='Trial Deep Embedded Validation on saved MLP candidates, without target labels.\n\nDEV estimator: You et al., ICML 2019; reference implementation:\nhttps://github.com/thuml/Deep-Embedded-Validation/blob/master/dev.py\nThis CLI ranks saved candidates; it does not retrain or replace canonical results.\n')
    parser.add_argument("--direction", choices=["both", *DIRECTIONS], default="unsw_to_cicids")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--methods", nargs="+", choices=["source_only", *METHOD_CONFIGS],
                        default=["source_only", *METHOD_CONFIGS])
    parser.add_argument("--candidates", type=Path, help="JSON list: name, direction, seed, checkpoint; relative paths resolved beside this JSON")
    parser.add_argument("--domain-rows", type=int, default=2000, help="Natural-prevalence rows per training domain")
    parser.add_argument("--validation-rows", type=int, default=10000, help="Natural-prevalence labeled SOURCE validation rows")
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--decays", nargs="+", type=float, default=list(DECAYS))
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--probability-floor", type=float, default=1e-8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-prefix", default="dev_trial")
    args = parser.parse_args()
    if (min(args.domain_rows,args.validation_rows) < 10 or min(args.max_iter,args.threads) < 1 or
            len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods) or
            any(not np.isfinite(d) or d <= 0 for d in args.decays) or
            not 0 < args.probability_floor < .5 or not args.output_prefix or
            Path(args.output_prefix).name != args.output_prefix or args.output_prefix in (".","..")):
        parser.error("Invalid sample sizes, iterations, seeds, decays, probability floor, or output prefix")
    run(args)


if __name__ == "__main__":
    main()
