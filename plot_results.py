import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

# Allow running this file directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from evaluation.proposal_results import load_score_reports
from experiments.proposal_protocol import DIRECTIONS, METHOD_CONFIGS, ROOT, SEEDS

DIRECTION_LABELS = {
    "unsw_to_cicids": "UNSW → CICIDS",
    "cicids_to_unsw": "CICIDS → UNSW",
}
METHOD_LABELS = {
    "marginal_mmd": "Marginal MMD",
    "mk_mmd": "MK-MMD",
    "class_aware_mmd": "Class-Aware MMD",
}
METHOD_STYLES = {
    "source_only": ("#252525", "--"),
    "marginal_mmd": ("#0072B2", "-"),
    "mk_mmd": ("#D55E00", "-."),
    "class_aware_mmd": ("#009E73", ":"),
}
CLASS_COLORS = {"benign": "#0072B2", "attack": "#D55E00"}
QUANTILES = ("p05", "p25", "p50", "p75", "p95")


def pr_points(scores):
    precision = np.asarray(scores["pr_curve"]["precision"], dtype=float)
    recall = np.asarray(scores["pr_curve"]["recall"], dtype=float)
    if precision.ndim != 1 or not len(precision) or precision.shape != recall.shape:
        raise ValueError("Invalid saved precision/recall arrays")
    if not all(
        np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        for values in (precision, recall)
    ):
        raise ValueError("Precision and recall must be finite probabilities")
    if np.any(np.diff(recall) > 1e-12):
        raise ValueError("Saved recall must be in decreasing threshold order")
    # The diagnostic omits sklearn's final no-prediction endpoint (recall=0, precision=1).
    return np.r_[recall, 0.0], np.r_[precision, 1.0]


def style_axes(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.2, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_ylim(0, 1.02)


def save_figure(figure, output, name):
    for extension in ("png", "pdf"):
        figure.savefig(output / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(figure)


def plot_precision_recall(reports, selections, seed, output):
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    for axis, direction in zip(axes, DIRECTIONS):
        baseline = reports[direction, "marginal_mmd"][
            "target_validation_score_analysis"
        ]["before"]
        # Draw adaptation first so the dashed baseline remains visible on exact ties.
        for method in METHOD_CONFIGS:
            scores = reports[direction, method]["target_validation_score_analysis"][
                "after"
            ]
            color, line = METHOD_STYLES[method]
            epoch = selections[direction, method]["best_epoch"]
            label = f"{METHOD_LABELS[method]}: AP={scores['average_precision']:.4f}, epoch={epoch}"
            axis.step(
                *pr_points(scores),
                where="post",
                color=color,
                linestyle=line,
                linewidth=1.6,
                label=label,
                rasterized=True,
            )
        color, line = METHOD_STYLES["source_only"]
        axis.step(
            *pr_points(baseline),
            where="post",
            color=color,
            linestyle=line,
            linewidth=1.8,
            label=f"Source-only: AP={baseline['average_precision']:.4f}",
            rasterized=True,
        )
        prevalence = baseline["attack_prevalence"]
        axis.axhline(
            prevalence,
            color="#777777",
            linestyle=(0, (1, 3)),
            linewidth=1,
            label=f"Attack prevalence={prevalence:.3f}",
        )
        style_axes(axis)
        axis.set(
            xlim=(0, 1),
            xlabel="Recall (Attack)",
            ylabel="Precision (Attack)",
            title=DIRECTION_LABELS[direction],
        )
        axis.legend(loc="upper right", fontsize=8, framealpha=0.95)
    figure.suptitle(
        f"Target-validation Precision–Recall curves · seed {seed}", fontsize=14
    )
    figure.text(
        0.5,
        0.025,
        "One fixed seed; AP comes from the saved score diagnostic and may differ slightly from training-table AP.\n"
        "Epoch 0 selects source-pretrained weights. No threshold is tuned here.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.94))
    save_figure(figure, output, "target_precision_recall")


def plot_score_quantiles(reports, selections, seed, output):
    figure, axes = plt.subplots(2, 3, figsize=(13.6, 8.6), sharex=True, sharey=True)
    quantile_percent = np.array([5, 25, 50, 75, 95])
    rows = []
    for direction_index, direction in enumerate(DIRECTIONS):
        for method_index, method in enumerate(METHOD_CONFIGS):
            axis = axes[direction_index, method_index]
            scores = reports[direction, method]["target_validation_score_analysis"]
            for label, color in CLASS_COLORS.items():
                for phase, line in (("before", "--"), ("after", "-")):
                    distribution = scores[phase]["score_distributions"][label]
                    values = np.array([distribution["quantiles"][q] for q in QUANTILES])
                    if (
                        not np.isfinite(values).all()
                        or np.any(np.diff(values) < -1e-12)
                        or np.any((values < 0) | (values > 1))
                    ):
                        raise ValueError(
                            f"Invalid score quantiles: {direction}/{method}/{label}/{phase}"
                        )
                    axis.plot(
                        quantile_percent,
                        values,
                        color=color,
                        linestyle=line,
                        marker="o" if label == "benign" else "^",
                        markersize=4.5,
                        linewidth=1.6,
                        markerfacecolor="white" if phase == "before" else color,
                    )
                    for quantile, value in zip(QUANTILES, values):
                        rows.append(
                            {
                                "direction": direction,
                                "method": method,
                                "seed": seed,
                                "true_label": (
                                    "Normal" if label == "benign" else "Attack"
                                ),
                                "phase": phase,
                                "quantile": quantile,
                                "attack_score": float(value),
                                "class_count": distribution["count"],
                            }
                        )
            epoch = selections[direction, method]["best_epoch"]
            selected = (
                "source-pretrained" if epoch == 0 else f"adaptation, epoch {epoch}"
            )
            axis.set_title(
                f"{DIRECTION_LABELS[direction]} · {METHOD_LABELS[method]}\nSelected: {selected}",
                fontsize=10,
            )
            counts = scores["before"]["score_distributions"]
            axis.text(
                0.03,
                0.97,
                f"Normal n={counts['benign']['count']:,}\nAttack n={counts['attack']['count']:,}",
                transform=axis.transAxes,
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
            style_axes(axis)
            axis.set(xticks=quantile_percent, xlim=(0, 100))
            if direction_index == 1:
                axis.set_xlabel("Within-class percentile")
            if method_index == 0:
                axis.set_ylabel("Predicted Attack score")
    handles = [
        Line2D(
            [],
            [],
            color=color,
            linestyle=line,
            marker="o" if label == "Normal" else "^",
            markersize=4,
            markerfacecolor="white" if phase == "before" else color,
            label=f"{label} · {'Source-only' if phase == 'before' else 'Selected checkpoint'}",
        )
        for label, color in (
            ("Normal", CLASS_COLORS["benign"]),
            ("Attack", CLASS_COLORS["attack"]),
        )
        for phase, line in (("before", "--"), ("after", "-"))
    ]
    figure.suptitle(f"Attack-score quantiles by true label · seed {seed}", fontsize=14)
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    figure.text(
        0.5,
        0.025,
        "Saved p05, p25, p50, p75 and p95; connecting segments are visual guides, not an ECDF.\n"
        "Before/after percentiles are not paired flows. Overlapping quantiles alone do not establish a ranking change.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.88), h_pad=2.2)
    save_figure(figure, output, "target_attack_score_quantiles")
    with (output / "attack_score_quantiles.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=ROOT / "results/canonical/diagnostics_target_val",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New figure directory; defaults to results/analysis/figures/seed<seed>_<timestamp>",
    )
    args = parser.parse_args()
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = ROOT / "results/analysis/figures" / f"seed{args.seed}_{timestamp}"
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    reports, selections, revision, sources, ap_checks = load_score_reports(
        args.diagnostics_root, args.seed
    )
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42}
    )
    plot_precision_recall(reports, selections, args.seed, args.output)
    plot_score_quantiles(reports, selections, args.seed, args.output)
    (args.output / "figure_manifest.json").write_text(
        json.dumps(
            {
                **revision,
                "seed": args.seed,
                "data_role": "target_validation",
                "pr_curve_scope": "full target-validation; one seed; saved AP; sklearn terminal endpoint restored",
                "score_distribution_scope": "five saved quantiles per true class; not full ECDF or paired-flow shifts",
                "selection": list(selections.values()),
                "sources": sources,
                "diagnostic_vs_locked_ap": ap_checks,
                "ap_comparison_absolute_tolerance": 1e-3,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved PNG, PDF, quantile CSV and provenance: {args.output}")


if __name__ == "__main__":
    main()
