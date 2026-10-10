from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

SEEDS = (42, 43, 44, 45, 46)
FEATURES = ("flow_duration", "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes")
DIRECTIONS = {
    "unsw_to_cicids": ("unsw", "cicids"),
    "cicids_to_unsw": ("cicids", "unsw"),
}
# 97.5th percentile of Student t(4), appropriate to exactly five paired seeds.
T_CRIT_DF4 = 2.7764451051977987
METRICS = ("pr_auc", "roc_auc", "macro_f1", "f1_attack", "recall", "fpr")


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def assert_close(a: float, b: float, where: str, atol: float = 1e-9) -> None:
    if not (math.isfinite(float(a)) and math.isfinite(float(b))) or not math.isclose(
        float(a), float(b), abs_tol=atol, rel_tol=0
    ):
        raise ValueError(f"{where}: mismatch {a} vs {b}")


def check_protocol(root: Path) -> list[dict]:
    checks = []

    def record(label: str, test, details: str) -> None:
        checks.append(
            {"check": label, "status": "PASS" if test else "FAIL", "details": details}
        )

    snap = root / "data/revisions/canonical"
    manifest_path = snap / "manifest.json"
    manifest = load_json(manifest_path)
    checksum_path = snap / "manifest.sha256"
    actual_digest = sha256(manifest_path)
    expected_digest = (
        checksum_path.read_text().split()[0] if checksum_path.exists() else None
    )
    record(
        "Manifest SHA-256",
        actual_digest == expected_digest,
        "manifest.json matches manifest.sha256",
    )
    record(
        "Frozen canonical",
        manifest.get("data_revision") == "canonical"
        and manifest.get("status") == "frozen"
        and manifest.get("quality_gate_passed") is True,
        "Expected canonical/frozen/quality_gate_passed",
    )
    record(
        "Feature order",
        manifest.get("feature_order") == list(FEATURES),
        ", ".join(FEATURES),
    )
    record(
        "Original counts preserved",
        manifest.get("flow_counts_preserved") is True,
        "No flow/label is dropped by freeze",
    )
    record(
        "Conditional byte semantics documented",
        "conditional" in str(manifest.get("semantic_byte_mapping_status", "")).lower(),
        "Feature semantic equivalence is not assumed",
    )

    policy = load_json(root / "configs/proposal_data_policy.json")
    relocation = load_json(snap / "relocation_audit.json")
    record(
        "Relocation gate",
        relocation.get("quality_gate_passed") is True
        and relocation.get("status") == "passed"
        and relocation.get("precision_cross_split_groups_after_final_fit") == 0,
        "Relocation passed; prepared collision groups = 0",
    )
    record(
        "Relocation policy unchanged",
        relocation.get("policy", {}).get("max_relocated_fraction")
        == policy.get("max_relocated_fraction")
        and relocation.get("policy", {}).get("max_class_prevalence_change")
        == policy.get("max_class_prevalence_change"),
        "Canonical values match predeclared config",
    )

    audit_dir = root / "results/data_audit/canonical"
    for name in ("raw_identity.json", "split_replay.json", "data_quality.json"):
        report = load_json(audit_dir / name)
        record(
            name,
            report.get("quality_gate_passed") is True,
            "Required quality_gate_passed=true",
        )

    prepared = load_json(snap / "prepared_quality.json")
    zero_prepared = all(
        prepared.get(direction, {}).get(domain, {}).get(key) == 0
        for direction in DIRECTIONS
        for domain in ("unsw", "cicids")
        for key in ("cross_split_vector_groups", "invalid_rows")
    )
    record(
        "Prepared split overlap/invalid",
        zero_prepared,
        "All two-direction prepared audit cells have zero overlap and invalid rows",
    )

    for direction, (source, _) in DIRECTIONS.items():
        fit_role = manifest.get("preprocessing", {}).get(direction, {}).get("fit_role")
        record(
            f"Preprocessor {direction}",
            fit_role == f"{source}_train_only",
            f"Expected {source}_train_only, got {fit_role}",
        )

    suite = load_json(root / "configs/proposal_suite_v2.json")
    record(
        "Development selection policy",
        suite.get("checkpoint_selection")
        == "source validation AP, including source-pretrained epoch 0"
        and suite.get("threshold_selection") == "source validation F1",
        "Checkpoint selected on source validation AP, threshold on source validation F1",
    )
    record(
        "Target data isolation policy",
        suite.get("target_train") == "features only during adaptation"
        and suite.get("target_val") == "post-hoc development diagnostics only"
        and suite.get("target_test") == "reserved for final evaluation",
        "Target train = unlabeled; target val = diagnostics; target test = final evaluation",
    )
    return checks


def validate_metrics(x: dict, label: str) -> tuple[int, int, int]:
    for k in ("tn", "fp", "fn", "tp", "prevalence", "threshold", *METRICS):
        if k not in x:
            raise ValueError(f"{label} missing {k}")
    counts = [int(x[k]) for k in ("tn", "fp", "fn", "tp")]
    if min(counts) < 0:
        raise ValueError(f"{label}: negative confusion matrix count")
    total = sum(counts)
    attack = counts[2] + counts[3]
    if total == 0 or attack == 0 or attack == total:
        raise ValueError(f"{label}: empty or single-class validation")
    assert_close(x["prevalence"], attack / total, f"{label} prevalence")
    return total, attack, total - attack


def load_source_baselines(root: Path) -> dict:
    baseline_dir = root / "results/canonical/source_only_target_val"
    manifest_digest = sha256(root / "data/revisions/canonical/manifest.json")
    feature_digest = sha256(root / "configs/common_features_v2.json")
    out = {}
    for direction, (source, target) in DIRECTIONS.items():
        for seed in SEEDS:
            path = baseline_dir / direction / f"seed{seed}.json"
            r = load_json(path)
            expected = {
                "protocol": "proposal_v2",
                "data_revision": "canonical",
                "direction": direction,
                "seed": seed,
                "feature_count": 5,
                "target_development_split": f"{target}_val",
                "data_manifest_sha256": manifest_digest,
                "common_feature_config_sha256": feature_digest,
            }
            for key, wanted in expected.items():
                if r.get(key) != wanted:
                    raise ValueError(
                        f"{path}: {key}: expected {wanted}, got {r.get(key)}"
                    )
            if r.get("features") != list(FEATURES):
                raise ValueError(f"{path}: feature order differs from frozen suite")
            within = r["source_val"]
            cross = r["target_development"]
            validate_metrics(within, f"{path} source_val")
            validate_metrics(cross, f"{path} target_development")
            assert_close(
                r["best_source_val_ap"],
                within["pr_auc"],
                f"{path} source best AP",
                atol=1e-6,
            )
            assert_close(
                r["threshold_from_source_val"],
                within["threshold"],
                f"{path} source threshold",
            )
            assert_close(
                r["threshold_from_source_val"],
                cross["threshold"],
                f"{path} cross threshold",
            )
            out[direction, seed] = r
    return out


def rq1_pairs(results: dict) -> list[dict]:
    rows = []
    # Both models are evaluated on the EXACT SAME canonical target validation split.
    for eval_domain in ("cicids", "unsw"):
        in_dir = "cicids_to_unsw" if eval_domain == "cicids" else "unsw_to_cicids"
        out_dir = "unsw_to_cicids" if eval_domain == "cicids" else "cicids_to_unsw"
        train_cross = DIRECTIONS[out_dir][0]
        for seed in SEEDS:
            within = results[in_dir, seed]["source_val"]
            cross = results[out_dir, seed]["target_development"]
            within_counts = validate_metrics(within, f"{eval_domain} within seed{seed}")
            cross_counts = validate_metrics(cross, f"{eval_domain} cross seed{seed}")
            if within_counts != cross_counts:
                raise ValueError(
                    f"{eval_domain} seed{seed}: both evaluations do not match row/class counts"
                )
            row = {
                "eval_domain": eval_domain,
                "seed": seed,
                "within_train_domain": eval_domain,
                "cross_train_domain": train_cross,
                "eval_split": f"{eval_domain}_val",
                "n_rows": within_counts[0],
                "attack_prevalence": within["prevalence"],
                "source_threshold_within": within["threshold"],
                "source_threshold_cross": cross["threshold"],
            }
            for metric in METRICS:
                row[f"within_{metric}"] = within[metric]
                row[f"cross_{metric}"] = cross[metric]
                row[f"drop_{metric}"] = within[metric] - cross[metric]
            rows.append(row)
    return rows


def seed_stats(xs: list[float]) -> tuple[float, float, float, float]:
    m, sd = mean(xs), stdev(xs)
    half = T_CRIT_DF4 * sd / math.sqrt(len(xs))
    return m, sd, m - half, m + half


def summarize(rows: list[dict]) -> list[dict]:
    table = []
    for domain in ("cicids", "unsw"):
        pairs = [r for r in rows if r["eval_domain"] == domain]
        if len(pairs) != 5:
            raise ValueError(f"Expected five paired results for {domain}")
        if len({(r["n_rows"], round(r["attack_prevalence"], 12)) for r in pairs}) != 1:
            raise ValueError(f"{domain}: target evaluation split changed across seeds")
        item = {
            "eval_domain": domain,
            "n_seeds": len(pairs),
            "evaluation_split": pairs[0]["eval_split"],
            "attack_prevalence": pairs[0]["attack_prevalence"],
        }
        for metric in METRICS:
            for part in ("within", "cross", "drop"):
                mm, ss, lo, hi = seed_stats(
                    [float(r[f"{part}_{metric}"]) for r in pairs]
                )
                item[f"{part}_{metric}_mean"] = mm
                item[f"{part}_{metric}_std"] = ss
                if part == "drop":
                    item[f"{part}_{metric}_ci95_low"] = lo
                    item[f"{part}_{metric}_ci95_high"] = hi
        table.append(item)
    return table


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def to_markdown(checks: list[dict], summary: list[dict]) -> str:
    lines = [
        "# Canonical protocol audit + RQ1 development comparison",
        "",
        "## Protocol audit (read-only)",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for item in checks:
        lines.append(f"| {item['check']} | {item['status']} | {item['details']} |")
    lines += [
        "",
        "## RQ1 — Within-domain vs. direct cross-domain",
        "",
        "Both models are evaluated on the SAME domain-specific validation split, paired by seed.",
        "Within-domain reference uses source validation for checkpoint and threshold selection; this is a development-level, potentially optimistic comparison.",
        "",
        "| Eval domain | Within AP (mean ± SD) | Cross AP (mean ± SD) | AP drop (mean; paired 95% CI) | ROC-AUC drop |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        fmt = lambda v: f"{v:.4f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["eval_domain"].upper(),
                    f"{fmt(row['within_pr_auc_mean'])} ± {fmt(row['within_pr_auc_std'])}",
                    f"{fmt(row['cross_pr_auc_mean'])} ± {fmt(row['cross_pr_auc_std'])}",
                    f"{fmt(row['drop_pr_auc_mean'])} [{fmt(row['drop_pr_auc_ci95_low'])}, {fmt(row['drop_pr_auc_ci95_high'])}]",
                    f"{fmt(row['drop_roc_auc_mean'])}",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Positive drop = performance degradation from within-domain to cross-domain.",
        "Random-ranking AP is approximately the attack prevalence of each evaluation split.",
        "IMPORTANT: Compare same-evaluation-domain results; do not interpret AP gaps across different validation domains as pure domain-shift effects.",
        "Target validation labels were used for post-hoc evaluation, NOT checkpoint or threshold selection in these source-only results.",
        "The final confirmatory RQ1 analysis should compare within-trained and cross-trained models on the SAME held-out domain test set after development lock.",
        "",
        "## Provenance",
        "",
        "Canonical files read; no training or target-test access.",
        "Run `python -m experiments.proposal_pipeline --stage verify` separately to validate all hashes and parquet inventory.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="NIDS-DA root directory"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/analysis/rq1"),
        help="Output directory; relative paths are relative to --root",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out if args.out.is_absolute() else root / args.out
    checks = check_protocol(root)
    print("Protocol audit:")
    for r in checks:
        print(f"  {r['status']:4s}  {r['check']}")
    if any(r["status"] == "FAIL" for r in checks):
        raise SystemExit(
            "Protocol audit FAIL: inspect frozen audit reports; RQ1 table not generated"
        )
    source_results = load_source_baselines(root)
    rows = rq1_pairs(source_results)
    summary = summarize(rows)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "rq1_per_seed.csv", rows)
    write_csv(out / "rq1_summary.csv", summary)
    (out / "protocol_audit.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
    )
    (out / "rq1_report.md").write_text(to_markdown(checks, summary), encoding="utf-8")
    print("\nRQ1 summary:")
    for r in summary:
        print(
            f"  {r['eval_domain'].upper()}: within AP={r['within_pr_auc_mean']:.4f}, "
            f"cross AP={r['cross_pr_auc_mean']:.4f}, "
            f"drop={r['drop_pr_auc_mean']:+.4f}, "
            f"paired 95% CI=[{r['drop_pr_auc_ci95_low']:.4f}, {r['drop_pr_auc_ci95_high']:.4f}]"
        )
    print(f"\nSaved 4 reports under {out}")


if __name__ == "__main__":
    main()
