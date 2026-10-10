"""Aggregate five-seed proposal_v2 development results without reading test data."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from training.proposal_mmd import output_paths, sha256
from features.common_features import COMMON_FEATURES

ROOT = Path(__file__).resolve().parents[2]
from training.data_revision import revision_path

RESULT_ROOT = revision_path("result_root", ROOT / "results/proposal_v2")
CONFIGS = {
    "marginal_mmd": ROOT / "configs/proposal_mmd_v2.json",
    "mk_mmd": ROOT / "configs/proposal_mkmmd_v2.json",
    "class_aware_mmd": ROOT / "configs/proposal_class_aware_v2.json",
}
METRICS = ("pr_auc", "macro_f1", "recall", "fpr", "roc_auc")


def seed_statistics(values):
    """Two-sided t interval for the mean across independent run seeds."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite seed observations")
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half_width = (
        float(student_t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
        if len(values) > 1
        else None
    )
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "ci95": (
            [mean - half_width, mean + half_width] if half_width is not None else None
        ),
        "ci_method": "student_t_across_seeds",
    }


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing proposal result: {path}")
    return json.loads(path.read_text())


def result_path(method, direction, seed):
    if method == "source_only":
        return RESULT_ROOT / "source_only_target_val" / direction / f"seed{seed}.json"
    return output_paths(direction, seed, CONFIGS[method])[1]


def diagnostic_path(method, direction, seed):
    name = (
        f"seed{seed}.json"
        if method == "marginal_mmd"
        else f"seed{seed}_{CONFIGS[method].stem}.json"
    )
    return RESULT_ROOT / "diagnostics_target_val" / direction / name


def aggregate(
    directions=("unsw_to_cicids", "cicids_to_unsw"), seeds=(42, 43, 44, 45, 46)
):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    rows = []
    summaries = []
    for direction in directions:
        for seed in seeds:
            baseline = read_json(result_path("source_only", direction, seed))
            if (
                baseline.get("features") != list(COMMON_FEATURES)
                or baseline.get("feature_count") != len(COMMON_FEATURES)
                or baseline["common_feature_config_sha256"]
                != sha256(ROOT / "configs/common_features_v2.json")
            ):
                raise ValueError(
                    "Baseline does not match the current five-feature v2 schema"
                )
            source_ap = baseline["target_development"]["pr_auc"]
            source_hash = baseline["checkpoint"]
            for method in ("source_only", *CONFIGS):
                result = (
                    baseline
                    if method == "source_only"
                    else read_json(result_path(method, direction, seed))
                )
                expected_split = (
                    "cicids_val" if direction == "unsw_to_cicids" else "unsw_val"
                )
                if (
                    result["direction"] != direction
                    or result["seed"] != seed
                    or result["target_development_split"] != expected_split
                ):
                    raise ValueError(
                        f"Protocol mismatch: {method} {direction} seed{seed}"
                    )
                if result.get("features") != list(COMMON_FEATURES) or result.get(
                    "feature_count"
                ) != len(COMMON_FEATURES):
                    raise ValueError("Result feature schema mismatch")
                metrics = (
                    result["target_development"]
                    if method == "source_only"
                    else result["cross_domain"]
                )
                row = {
                    "direction": direction,
                    "seed": seed,
                    "method": method,
                    **{metric: float(metrics[metric]) for metric in METRICS},
                    "confusion_matrix": {
                        key: int(metrics[key]) for key in ("tn", "fp", "fn", "tp")
                    },
                    "adaptation_gain_pr_auc": float(metrics["pr_auc"] - source_ap),
                }
                if method != "source_only":
                    if result["source_checkpoint"] != source_hash:
                        raise ValueError(
                            f"Different source checkpoint: {method} {direction} seed{seed}"
                        )
                    if result["config_sha256"] != sha256(CONFIGS[method]):
                        raise ValueError(
                            f"Stale method config: {method} {direction} seed{seed}"
                        )
                    if (
                        result["common_feature_config_sha256"]
                        != baseline["common_feature_config_sha256"]
                        or result["preprocessor_sha256"]
                        != baseline["preprocessor_sha256"]
                        or result["prepared_split_sha256"]
                        != baseline["prepared_split_sha256"]
                    ):
                        raise ValueError(
                            f"Different prepared data: {method} {direction} seed{seed}"
                        )
                    row["selected_stage"] = result["selected_stage"]
                    diagnostic_file = diagnostic_path(method, direction, seed)
                    if diagnostic_file.is_file():
                        diagnostic = read_json(diagnostic_file)
                        if (
                            diagnostic["direction"] != direction
                            or diagnostic["seed"] != seed
                            or diagnostic["config_sha256"] != result["config_sha256"]
                            or diagnostic["artifacts"]["adapted_checkpoint"]
                            != result["checkpoint"]
                        ):
                            raise ValueError(
                                f"Stale diagnostic: {method} {direction} seed{seed}"
                            )
                        row["post_hoc"] = {
                            "mmd2_before": diagnostic["marginal_alignment"]["before"][
                                "mmd2_mean"
                            ],
                            "mmd2_after": diagnostic["marginal_alignment"]["after"][
                                "mmd2_mean"
                            ],
                            "domain_auc_before": diagnostic["domain_separability"][
                                "before"
                            ]["auc_mean"],
                            "domain_auc_after": diagnostic["domain_separability"][
                                "after"
                            ]["auc_mean"],
                        }
                rows.append(row)
        for method in ("source_only", *CONFIGS):
            group = [
                row
                for row in rows
                if row["direction"] == direction and row["method"] == method
            ]
            if len(group) != len(seeds):
                raise ValueError(f"Incomplete seeds: {direction} {method}")
            metrics = {}
            for metric in (*METRICS, "adaptation_gain_pr_auc"):
                values = np.asarray([row[metric] for row in group], dtype=np.float64)
                metrics[metric] = seed_statistics(values)
            confusion = {
                key: sum(row["confusion_matrix"][key] for row in group)
                for key in ("tn", "fp", "fn", "tp")
            }
            summaries.append(
                {
                    "direction": direction,
                    "method": method,
                    "seeds": list(seeds),
                    "metrics": metrics,
                    "confusion_matrix_sum": confusion,
                    "paired_delta_ap": [
                        {"seed": row["seed"], "delta_ap": row["adaptation_gain_pr_auc"]}
                        for row in group
                    ],
                }
            )
            diagnostic_rows = [row["post_hoc"] for row in group if "post_hoc" in row]
            if diagnostic_rows:
                summaries[-1]["post_hoc"] = {
                    "runs": len(diagnostic_rows),
                    **{
                        key: float(np.mean([row[key] for row in diagnostic_rows]))
                        for key in (
                            "mmd2_before",
                            "mmd2_after",
                            "domain_auc_before",
                            "domain_auc_after",
                        )
                    },
                }
    return {
        "protocol": "proposal_v2",
        "data_role": "development_only",
        "directions": list(directions),
        "seeds": list(seeds),
        "per_run": rows,
        "summary": summaries,
        "statistical_reporting": {
            "pairing": "same direction and source checkpoint seed",
            "ci": "95% Student-t interval of seed means; delta CI computed from paired deltas",
            "caveat": "Five seeds give uncertain intervals; normality and seed independence are assumptions. Seed variation does not measure dataset sampling uncertainty.",
            "significance_test": None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction",
        choices=["unsw_to_cicids", "cicids_to_unsw", "both"],
        default="both",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--output", type=Path, default=RESULT_ROOT / "aggregate/summary.json"
    )
    args = parser.parse_args()
    directions = (
        ("unsw_to_cicids", "cicids_to_unsw")
        if args.direction == "both"
        else (args.direction,)
    )
    result = aggregate(directions, tuple(args.seeds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for row in result["summary"]:
        ap = row["metrics"]["pr_auc"]
        gain = row["metrics"]["adaptation_gain_pr_auc"]
        print(
            f"{row['direction']:17s} {row['method']:16s} AP={ap['mean']:.4f}±{ap['std']:.4f} gain={gain['mean']:+.4f}±{gain['std']:.4f} CI95={gain['ci95']}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
