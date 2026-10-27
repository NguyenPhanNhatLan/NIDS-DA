import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from evaluation.proposal_results import load_score_reports
from evaluation.plot_style import DIRECTION_LABELS, METHOD_LABELS
from experiments.proposal_protocol import DIRECTIONS, METHOD_CONFIGS, ROOT, SEEDS

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

    return np.r_[recall, 0.0], np.r_[precision, 1.0]


def style_axes(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.2, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_ylim(0, 1.02)


def save_figure(figure, output, name):
    figure.savefig(output / f"{name}.png", dpi=300, facecolor="white")
    plt.close(figure)


def plot_precision_recall(reports, selections, seed, output):
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    for axis, direction in zip(axes, DIRECTIONS):
        baseline = reports[direction, "marginal_mmd"][
            "target_validation_score_analysis"
        ]["before"]

        adaptation_lines = []
        for method in METHOD_CONFIGS:
            scores = reports[direction, method]["target_validation_score_analysis"][
                "after"
            ]
            color, line = METHOD_STYLES[method]
            label = f"{METHOD_LABELS[method]} · AP = {scores['average_precision']:.4f}"
            line_handle, = axis.step(
                *pr_points(scores),
                where="post",
                color=color,
                linestyle=line,
                linewidth=1.6,
                label=label,
            )
            adaptation_lines.append(line_handle)
        color, line = METHOD_STYLES["source_only"]
        baseline_line, = axis.step(
            *pr_points(baseline),
            where="post",
            color=color,
            linestyle=line,
            linewidth=1.8,
            label=f"Source-only · AP = {baseline['average_precision']:.4f}",
        )
        prevalence = baseline["attack_prevalence"]
        prevalence_line = axis.axhline(
            prevalence,
            color="#777777",
            linestyle=(0, (1, 3)),
            linewidth=1,
            label=f"Attack prevalence = {prevalence:.1%}",
        )
        style_axes(axis)
        axis.set(
            xlim=(0, 1),
            xlabel="Recall (Attack)",
            ylabel="Precision (Attack)",
            title=DIRECTION_LABELS[direction],
        )
        axis.legend(
            handles=[baseline_line, *adaptation_lines, prevalence_line],
            loc="upper right", fontsize=8, framealpha=0.95,
        )
        if (
            direction == "unsw_to_cicids"
            and selections[direction, "marginal_mmd"]["selected_stage"] == "source_pretrained"
        ):
            axis.text(
                0.98, 0.64,
                "Marginal MMD retains the source-only model;\ncurves overlap.",
                transform=axis.transAxes, ha="right", va="top", fontsize=8,
                color="#444444",
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
            )
    figure.suptitle(
        f"Target-validation Precision–Recall curves · seed {seed}", fontsize=14
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, output, f"seed{seed}__target_precision_recall")


def plot_score_quantiles(reports, selections, seed, output):
    figure, axes = plt.subplots(2, 3, figsize=(13.6, 8.6), sharex=True, sharey=True)
    quantile_percent = np.array([5, 25, 50, 75, 95])
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
            threshold = scores["after"]["source_threshold_metrics"]["threshold"]
            axis.axhline(threshold, color="#777777", linestyle=":", linewidth=1.1)
            axis.annotate(
                f"Decision threshold = {threshold:.4f}",
                xy=(0.97, threshold), xycoords=axis.get_yaxis_transform(),
                xytext=(0, 5 if threshold < 0.15 else -5), textcoords="offset points",
                ha="right", va="bottom" if threshold < 0.15 else "top",
                fontsize=8, color="#666666",
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 1.5},
            )
            axis.set_title(
                f"{DIRECTION_LABELS[direction]} · {METHOD_LABELS[method]}", fontsize=10,
            )
            if (
                direction == "unsw_to_cicids"
                and method == "marginal_mmd"
                and selections[direction, method]["selected_stage"] == "source_pretrained"
            ):
                axis.text(
                    0.03, 0.80, "Source-only retained",
                    transform=axis.transAxes, va="top", fontsize=8, color="#444444",
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
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
            label=f"{label} · {'Before' if phase == 'before' else 'After'}",
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
    figure.tight_layout(rect=(0, 0, 1, 0.88), h_pad=2.2)
    save_figure(figure, output, f"seed{seed}__target_attack_score_quantiles")


def plot_feature_ecdfs(output):
    import pyarrow.parquet as pq
    from features.common_features import COMMON_FEATURES
    from training.data_revision import revision_path, verify_revision

    verify_revision()
    common = revision_path("common_root", None)
    titles = ("Flow duration", "Fwd packets", "Bwd packets", "Fwd bytes", "Bwd bytes")
    series = []
    colors = {("unsw", 0): "#0066FF", ("unsw", 1): "#FF7F0E",
              ("cicids", 0): "#18B7A2", ("cicids", 1): "#A855FF"}
    for domain, role in (("unsw", "Source"), ("cicids", "Target")):
        table = pq.read_table(common/f"{domain}_val", columns=[*COMMON_FEATURES, "label"])
        labels = table["label"].to_numpy()
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Unexpected validation labels")
        for label, name in ((0, "Benign"), (1, "Attack")):
            values = [table[feature].to_numpy()[labels == label] for feature in COMMON_FEATURES]
            series.append((f"{role} {name}", colors[domain, label], values))


    figure, axes = plt.subplots(3, 2, figsize=(11, 10), sharey=True)
    for index, (axis, title) in enumerate(zip(axes.flat, titles)):
        curves = []
        for label, color, features in series:
            raw = features[index]
            finite = raw[np.isfinite(raw)]
            if not len(finite):
                raise ValueError(f"No finite observations: {label}: {title}")
            logged = np.sign(finite)*np.log1p(np.abs(finite))
            x, counts = np.unique(logged, return_counts=True)
            y = np.cumsum(counts)/len(logged)
            curves.append((label, color, x, y))
            if len(finite) != len(raw):
                print(f"{label}, {title}: excluded {len(raw)-len(finite)} nonfinite values")
        left = min(0., min(x[0] for _, _, x, _ in curves))
        right = max(x[-1] for _, _, x, _ in curves)
        for label, color, x, y in curves:
            axis.step(np.r_[left, x, right], np.r_[0., y, 1.], where="post", color=color,
                      linewidth=1.5, label=label)
        axis.set_title(f"({chr(97+index)}) {title}", fontsize=11, fontweight="bold", pad=9)
        axis.set(xlabel="Signed log1p(feature value)", ylabel="ECDF", ylim=(0, 1.01),
                 yticks=np.linspace(0, 1, 6))
        axis.tick_params(labelsize=9, labelleft=True)
        axis.set_facecolor("#F7F9FC")
        axis.grid(color="#E1E7EF", linewidth=.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#B9C5D5")
        axis.margins(x=.02)
    axes[2, 1].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[2, 1].legend(handles, labels, loc="center", frameon=False,
                      fontsize=11, handlelength=2.8, labelspacing=.65)
    figure.tight_layout(pad=1.5, h_pad=2, w_pad=2)
    output.mkdir(parents=True, exist_ok=True)
    save_figure(figure, output, "class_stratified_feature_ecdf")
    print(f"Saved wider feature ECDF: {output/'class_stratified_feature_ecdf.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=["scores", "mechanisms", "ecdf"], default="scores")
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "results/analysis/section_4_5_canonical")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=ROOT / "results/canonical/diagnostics_target_val",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG output directory; defaults to results/images/",
    )
    args = parser.parse_args()
    if args.section == "ecdf":
        plot_feature_ecdfs(args.output or ROOT / "results/images")
        return
    if args.section == "mechanisms":
        from evaluation.proposal_mechanism_figures import plot_mechanisms
        plot_mechanisms(args.audit_dir, args.output or ROOT / "results/images")
        return
    if args.output is None:
        args.output = ROOT / "results/images"
    reports, selections, *_ = load_score_reports(
        args.diagnostics_root, args.seed
    )
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10}
    )
    plot_precision_recall(reports, selections, args.seed, args.output)
    plot_score_quantiles(reports, selections, args.seed, args.output)
    print(f"Saved 2 PNG files: {args.output}")


if __name__ == "__main__":
    main()
