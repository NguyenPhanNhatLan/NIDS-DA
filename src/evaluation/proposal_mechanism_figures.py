import csv
import json
from pathlib import Path

from project_utils.result_files import result_file

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from evaluation.plot_style import DIRECTION_LABELS, METHOD_LABELS, METHOD_COLORS, METHOD_MARKERS
from evaluation.proposal_development_lock import require_development_lock
from project_utils import sha256
from experiments.proposal_protocol import DIRECTIONS, METHOD_CONFIGS, SEEDS


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    return float(value) if value != "" else np.nan


def finish_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.2, linewidth=0.6)
    axis.set_axisbelow(True)


def save_png(figure, output, name):
    figure.savefig(output/f"mechanisms__{name}.png", dpi=300, facecolor="white")
    plt.close(figure)


def method_handles():
    return [Line2D([], [], linestyle="none", marker=METHOD_MARKERS[method],
                   color=METHOD_COLORS[method], markersize=6, label=METHOD_LABELS[method])
            for method in METHOD_CONFIGS]


def plot_knn(rows, manifest, output):
    figure, axes = plt.subplots(2, 4, figsize=(14.4, 8.5), sharex="col", sharey="col")
    columns = [(task, label) for task in ("target_within", "target_to_source") for label in ("Normal", "Attack")]
    for row_index, direction in enumerate(DIRECTIONS):
        for column, (task, label) in enumerate(columns):
            axis = axes[row_index, column]
            axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
            for method in METHOD_CONFIGS:
                group = [r for r in rows if r["direction"] == direction and r["method"] == method
                         and r["task"] == task and r["class_label"] == label]
                if sorted(int(r["seed"]) for r in group) != list(SEEDS):
                    raise ValueError("Incomplete method–seed matrix in kNN plot")
                axis.scatter([number(r["before"]) for r in group], [number(r["after"]) for r in group],
                             color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                             s=35, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)
            limit = 0.15 if task == "target_within" else 1.0
            if any(number(r[field]) > limit for r in rows if r["task"] == task for field in ("before", "after")):
                raise ValueError("kNN point exceeds the declared panel scale")
            margin = .02*limit
            axis.set(xlim=(-margin, limit+margin), ylim=(-margin, limit+margin), aspect="equal",
                     xticks=np.linspace(0, limit, 4 if task == "target_within" else 5),
                     yticks=np.linspace(0, limit, 4 if task == "target_within" else 5))
            if row_index == 0:
                axis.set_title(f"{task}\n{label}", fontsize=10)
            if column == 0:
                axis.set_ylabel(f"{DIRECTION_LABELS[direction]}\nMixing after")
            if row_index == 1:
                axis.set_xlabel("Mixing before")
            finish_axis(axis)
    figure.suptitle("Class-wise kNN mixing before and after adaptation", fontsize=14)
    figure.legend(handles=method_handles(), loc="upper center", bbox_to_anchor=(.5, .935), ncol=3, frameon=False)
    k, per_class = manifest["knn"]["k"], manifest["knn"]["per_class"]
    figure.text(.5, .022,
                f"k={k}; {per_class:,} Normal + {per_class:,} Attack per domain, without replacement; fixed validation flows across methods/seeds 42–46.\n"
                "Euclidean distance in original latent space; self row excluded within domain. Mixing = fraction of opposite-class neighbors.\n"
                "Each point is one method–seed; below y=x means lower mixing. Within-target axes: 0–0.15; cross-domain axes: 0–1. This is not an MMD separation ratio.",
                ha="center", fontsize=8.5, color="#444444")
    figure.subplots_adjust(left=.08, right=.985, bottom=.16, top=.80, hspace=.38, wspace=.30)
    save_png(figure, output, "figure1_knn_mixing")


def plot_pseudo_labels(rows, manifest, output):
    figure, axes = plt.subplots(2, 3, figsize=(12.8, 7.5), sharex=True, sharey=True)
    metrics = [("attack_precision", "Attack precision"), ("attack_recall", "Attack recall"), ("coverage", "Coverage")]
    phases = {"before": ("#5B6770", "--"), "after": (METHOD_COLORS["class_aware_mmd"], "-")}
    levels = manifest["pseudo_labels"]["thresholds"]
    for row_index, direction in enumerate(DIRECTIONS):
        for column, (metric, title) in enumerate(metrics):
            axis = axes[row_index, column]
            for phase, (color, style) in phases.items():
                group = [r for r in rows if r["direction"] == direction and r["method"] == "class_aware_mmd" and r["phase"] == phase]
                lookup = {(int(r["seed"]), number(r["threshold"])): number(r[metric]) for r in group}
                for seed in SEEDS:
                    axis.plot(levels, [lookup[seed, level] for level in levels], color=color,
                              linestyle=style, linewidth=.8, alpha=.3, marker=".", markersize=3)
                means = []
                for level in levels:
                    values = np.array([lookup[seed, level] for seed in SEEDS])
                    means.append(float(np.nanmean(values)) if np.isfinite(values).any() else np.nan)
                axis.plot(levels, means, color=color, linestyle=style, linewidth=2.3,
                          marker="o" if phase == "before" else "s", markersize=5, zorder=3)
            actual = sorted({number(r["threshold"]) for r in rows if r["direction"] == direction
                             and r["method"] == "class_aware_mmd" and r["is_training_threshold"] == "True"})
            if len(actual) != 1:
                raise ValueError("Expected one canonical training confidence")
            axis.axvline(actual[0], color="#999999", linestyle=":", linewidth=1)
            if column == 0:
                axis.text(actual[0]+.003, .97, f"Training τ={actual[0]:.2f}",
                          transform=axis.get_xaxis_transform(), ha="left", va="top", fontsize=8,
                          color="#666666", bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
            axis.set(xticks=levels, xlim=(min(levels)-.012, max(levels)+.012), ylim=(-.02, 1.04))
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            if row_index == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(DIRECTION_LABELS[direction])
            if row_index == 1:
                axis.set_xlabel("Confidence threshold τ")
            finish_axis(axis)
    handles = [Line2D([], [], color=color, linestyle=style, linewidth=2.3,
                      marker="o" if phase == "before" else "s", label=f"{phase.title()} · mean over 5 seeds")
               for phase, (color, style) in phases.items()]
    figure.suptitle("Target pseudo-label reliability · canonical Class-Aware MMD", fontsize=14)
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .935), ncol=2, frameon=False)
    figure.text(.5, .025,
                "Thin lines: individual seeds 42–46; thick lines: unweighted seed means. Before: source-only; After: selected saved model.\n"
                "Full target-validation audit at fixed models, not online training label errors. Undefined precision (no accepted pseudo-Attack) is omitted.\n"
                "τ is pseudo-label confidence, distinct from the Attack/Normal decision threshold. Canonical training uses τ=0.80.",
                ha="center", fontsize=8.5, color="#444444")
    figure.tight_layout(rect=(0, .12, 1, .88), h_pad=2)
    save_png(figure, output, "figure2_pseudo_label_reliability")


def plot_gradient_conflict(rows, manifest, output):
    figure, axes = plt.subplots(1, 2, figsize=(12.3, 5.6), sharex=True, sharey=True)
    deltas = [number(r["delta_ap"]) for r in rows]
    lower, upper = min(min(deltas), 0)-.035, max(max(deltas), 0)+.045
    for axis, direction in zip(axes, DIRECTIONS):
        axis.axhline(0, color="#777777", linestyle="--", linewidth=1)
        missing = []
        for method in METHOD_CONFIGS:
            group = [r for r in rows if r["direction"] == direction and r["method"] == method]
            if sorted(int(r["seed"]) for r in group) != list(SEEDS):
                raise ValueError("Incomplete method–seed matrix in gradient plot")
            usable = [r for r in group if r["negative_fraction"] != ""]
            if len(usable) < 5:
                missing.append(f"{METHOD_LABELS[method]}: cosine defined for {len(usable)}/5 seeds")
            for row in usable:
                x, y = number(row["negative_fraction"]), number(row["delta_ap"])
                axis.scatter(x, y, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                             s=50, edgecolors="white", linewidths=.5, zorder=3)
                if len(usable) < 5:
                    axis.annotate(f"{row['seed']} (n={row['valid_batches']})", (x, y),
                                  xytext=(5, 4 if y > 0 else -7), textcoords="offset points",
                                  va="bottom" if y > 0 else "top",
                                  fontsize=8, color=METHOD_COLORS[method])
        if missing:
            axis.text(.02, .03, "\n".join(missing), transform=axis.transAxes,
                      ha="left", va="bottom", fontsize=8, color="#555555",
                      bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
        axis.set(title=DIRECTION_LABELS[direction], xlim=(-.025, 1.025), ylim=(lower, upper),
                 xlabel="Fraction of valid batches with gradient cosine < 0")
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        finish_axis(axis)
    axes[0].set_ylabel("ΔAP (After − source-only)")
    figure.suptitle("Classification–alignment gradient conflict and target AP change", fontsize=14)
    figure.legend(handles=method_handles(), loc="upper center", bbox_to_anchor=(.5, .93), ncol=3, frameon=False)
    batches = manifest["gradient"]["batches"]
    size = manifest["gradient"]["batch_size"]
    figure.text(.5, .025,
                f"Actual method-specific alignment loss vs class-balanced source CE at the selected saved model; {batches} fixed paired training batches × {size} rows/domain.\n"
                "Uniform rows from entire source/target training splits; same pairs across methods/seeds. Target-training labels are not read.\n"
                "Shared encoder gradients; dropout active, BN statistics frozen. Cosine requires two nonzero gradients; skipped/zero-gradient counts are in the appendix.\n"
                "Each point is one method–seed; sparse class-aware labels show seed and valid-batch count n. This local audit does not establish training dynamics or causality.",
                ha="center", fontsize=8, color="#444444")
    figure.tight_layout(rect=(0, .16, 1, .86))
    save_png(figure, output, "figure3_gradient_conflict")


def plot_mechanisms(audit_root, output):
    manifest = json.loads(result_file(audit_root/"audit_manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("scope") != "canonical_only":
        raise ValueError("Require a complete canonical-only mechanism audit")
    if manifest["training_seeds"] != list(SEEDS) or manifest["methods"] != list(METHOD_CONFIGS):
        raise ValueError("Audit matrix differs from the canonical suite")
    require_development_lock({key: manifest[key] for key in ("data_revision", "data_manifest_sha256")})
    for name, expected in manifest.get("outputs_sha256", {}).items():
        if sha256(audit_root/name) != expected:
            raise ValueError(f"Audit table changed since measurement: {name}")
    output.mkdir(parents=True, exist_ok=True)
    plot_knn(read_rows(result_file(audit_root/"knn_runs.csv")), manifest, output)
    plot_pseudo_labels(read_rows(result_file(audit_root/"appendix_pseudo_label_runs.csv")), manifest, output)
    plot_gradient_conflict(read_rows(result_file(audit_root/"appendix_gradient_runs.csv")), manifest, output)
    print(f"Saved 3 canonical mechanism PNG figures: {output}")
