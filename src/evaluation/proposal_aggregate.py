"""Aggregate five-seed proposal_v1 development results without reading test data."""
import argparse
import json
from pathlib import Path

import numpy as np

from training.proposal_mmd import output_paths, sha256

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results/proposal_v1"
CONFIGS = {
    "marginal_mmd": ROOT / "configs/proposal_mmd_v1.json",
    "mk_mmd": ROOT / "configs/proposal_mkmmd_v1.json",
    "class_aware_mmd": ROOT / "configs/proposal_class_aware_v1.json",
}
METRICS = ("pr_auc", "macro_f1", "recall", "fpr", "roc_auc")


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing proposal result: {path}")
    return json.loads(path.read_text())


def result_path(method, direction, seed):
    if method == "source_only":
        return RESULT_ROOT / "source_only_target_val" / direction / f"seed{seed}.json"
    return output_paths(direction, seed, CONFIGS[method])[1]


def diagnostic_path(method, direction, seed):
    name = f"seed{seed}.json" if method == "marginal_mmd" else f"seed{seed}_{CONFIGS[method].stem}.json"
    return RESULT_ROOT / "diagnostics_target_val" / direction / name


def aggregate(directions=("unsw_to_cicids", "cicids_to_unsw"), seeds=(42, 43, 44, 45, 46)):
    rows = []
    summaries = []
    for direction in directions:
        for seed in seeds:
            baseline = read_json(result_path("source_only", direction, seed))
            source_ap = baseline["target_development"]["pr_auc"]
            source_hash = baseline["checkpoint"]
            for method in ("source_only", *CONFIGS):
                result = baseline if method == "source_only" else read_json(result_path(method, direction, seed))
                expected_split = "cicids_val" if direction == "unsw_to_cicids" else "unsw_val"
                if result["direction"] != direction or result["seed"] != seed or result["target_development_split"] != expected_split:
                    raise ValueError(f"Protocol mismatch: {method} {direction} seed{seed}")
                metrics = result["target_development"] if method == "source_only" else result["cross_domain"]
                row = {"direction": direction, "seed": seed, "method": method,
                       **{metric: float(metrics[metric]) for metric in METRICS},
                       "confusion_matrix": {key: int(metrics[key]) for key in ("tn", "fp", "fn", "tp")},
                       "adaptation_gain_pr_auc": float(metrics["pr_auc"] - source_ap)}
                if method != "source_only":
                    if result["source_checkpoint"] != source_hash:
                        raise ValueError(f"Different source checkpoint: {method} {direction} seed{seed}")
                    if result["config_sha256"] != sha256(CONFIGS[method]):
                        raise ValueError(f"Stale method config: {method} {direction} seed{seed}")
                    if (result["common_feature_config_sha256"] != baseline["common_feature_config_sha256"]
                            or result["preprocessor_sha256"] != baseline["preprocessor_sha256"]
                            or result["prepared_split_sha256"] != baseline["prepared_split_sha256"]):
                        raise ValueError(f"Different prepared data: {method} {direction} seed{seed}")
                    row["selected_stage"] = result["selected_stage"]
                    diagnostic_file = diagnostic_path(method, direction, seed)
                    if diagnostic_file.is_file():
                        diagnostic = read_json(diagnostic_file)
                        if (diagnostic["direction"] != direction or diagnostic["seed"] != seed
                                or diagnostic["config_sha256"] != result["config_sha256"]
                                or diagnostic["artifacts"]["adapted_checkpoint"] != result["checkpoint"]):
                            raise ValueError(f"Stale diagnostic: {method} {direction} seed{seed}")
                        row["post_hoc"] = {
                            "mmd2_before": diagnostic["marginal_alignment"]["before"]["mmd2_mean"],
                            "mmd2_after": diagnostic["marginal_alignment"]["after"]["mmd2_mean"],
                            "domain_auc_before": diagnostic["domain_separability"]["before"]["auc_mean"],
                            "domain_auc_after": diagnostic["domain_separability"]["after"]["auc_mean"],
                        }
                rows.append(row)
        for method in ("source_only", *CONFIGS):
            group = [row for row in rows if row["direction"] == direction and row["method"] == method]
            if len(group) != len(seeds):
                raise ValueError(f"Incomplete seeds: {direction} {method}")
            metrics = {}
            for metric in (*METRICS, "adaptation_gain_pr_auc"):
                values = np.asarray([row[metric] for row in group], dtype=np.float64)
                metrics[metric] = {"mean": float(values.mean()),
                                   "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
            confusion = {key: sum(row["confusion_matrix"][key] for row in group)
                         for key in ("tn", "fp", "fn", "tp")}
            summaries.append({"direction": direction, "method": method, "seeds": list(seeds),
                              "metrics": metrics, "confusion_matrix_sum": confusion})
            diagnostic_rows = [row["post_hoc"] for row in group if "post_hoc" in row]
            if diagnostic_rows:
                summaries[-1]["post_hoc"] = {"runs": len(diagnostic_rows), **{
                    key: float(np.mean([row[key] for row in diagnostic_rows]))
                    for key in ("mmd2_before", "mmd2_after", "domain_auc_before", "domain_auc_after")
                }}
    return {"protocol": "proposal_v1_target_val", "data_role": "development_only",
            "directions": list(directions), "seeds": list(seeds),
            "per_run": rows, "summary": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=["unsw_to_cicids", "cicids_to_unsw", "both"], default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / "aggregate/summary.json")
    args = parser.parse_args()
    directions = ("unsw_to_cicids", "cicids_to_unsw") if args.direction == "both" else (args.direction,)
    result = aggregate(directions, tuple(args.seeds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for row in result["summary"]:
        ap = row["metrics"]["pr_auc"]
        gain = row["metrics"]["adaptation_gain_pr_auc"]
        print(f"{row['direction']:17s} {row['method']:16s} AP={ap['mean']:.4f}±{ap['std']:.4f} gain={gain['mean']:+.4f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
