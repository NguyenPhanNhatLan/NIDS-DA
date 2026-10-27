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
    figure, axes = plt.subplots(2, 2, figsize=(8.8, 8.5), sharex=True, sharey=True)
    for row_index, direction in enumerate(DIRECTIONS):
        for column, label in enumerate(("Normal", "Attack")):
            axis = axes[row_index, column]
            axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
            for method in METHOD_CONFIGS:
                group = [r for r in rows if r["direction"] == direction and r["method"] == method
                         and r["task"] == "target_to_source" and r["class_label"] == label]
                if sorted(int(r["seed"]) for r in group) != list(SEEDS):
                    raise ValueError("Incomplete method–seed matrix in kNN plot")
                before = [number(r["before"]) for r in group]
                after = [number(r["after"]) for r in group]
                if any(not 0 <= value <= 1 for value in before + after):
                    raise ValueError("kNN mixing must be in [0, 1]")
                axis.scatter(before, after, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                             s=40, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)
            axis.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), aspect="equal",
                     xticks=np.linspace(0, 1, 5), yticks=np.linspace(0, 1, 5))
            if row_index == 0:
                axis.set_title(label, fontsize=11)
            if column == 0:
                axis.set_ylabel(f"{DIRECTION_LABELS[direction]}\nMixing after")
            if row_index == 1:
                axis.set_xlabel("Mixing before")
            finish_axis(axis)
    figure.suptitle("Target-to-source kNN mixing by class", fontsize=14)
    figure.legend(handles=method_handles(), loc="upper center", bbox_to_anchor=(.5, .935),
                  ncol=3, frameon=False)
    figure.subplots_adjust(left=.13, right=.97, bottom=.09, top=.83, hspace=.22, wspace=.20)
    save_png(figure, output, "figure1_knn_mixing")


def plot_pseudo_labels(rows, manifest, output):
    figure, axes = plt.subplots(2, 3, figsize=(12.8, 7.5), sharex=True, sharey=True)
    metrics = [("attack_precision", "(a) Attack precision"), ("attack_recall", "(b) Attack recall"), ("coverage", "(c) Coverage")]
    seed_markers = dict(zip(SEEDS, ("o", "v", "^", "D", "x")))
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
                              linestyle=style, linewidth=1, alpha=.5, marker=seed_markers[seed], markersize=4)
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
    figure.suptitle("Target pseudo-label quality across confidence thresholds", fontsize=14)
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .935), ncol=2, frameon=False)
    seed_handles = [Line2D([], [], color="#666666", linestyle="none", marker=seed_markers[seed],
                           markersize=4, label=f"Seed {seed}") for seed in SEEDS]
    figure.legend(handles=seed_handles, loc="upper center", bbox_to_anchor=(.5, .89),
                  ncol=5, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, .83), h_pad=2)
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
    figure.suptitle("Method-specific gradient conflict at selected models", fontsize=14)
    figure.legend(handles=method_handles(), loc="upper center", bbox_to_anchor=(.5, .93), ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, .86))
    save_png(figure, output, "figure3_gradient_conflict")


def plot_batchnorm_shift(rows, output):
    methods = ("source_only", *METHOD_CONFIGS)
    domains = (("source_shift", "Source", "#0072B2", "o", -.16),
               ("target_shift", "Target", "#D55E00", "^", .16))
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 8), sharex=True, sharey=True)
    values = [number(row[key]) for row in rows for key, *_ in domains]
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("The logarithmic BN shift figure requires finite positive shifts")
    jitter = np.linspace(-.055, .055, len(SEEDS))
    for row_index, direction in enumerate(DIRECTIONS):
        for column, layer in enumerate(("bn1", "bn2")):
            axis = axes[row_index, column]
            for x, method in enumerate(methods):
                group = sorted((r for r in rows if r["direction"] == direction and
                                r["method"] == method and r["layer"] == layer), key=lambda r: int(r["seed"]))
                if [int(r["seed"]) for r in group] != list(SEEDS):
                    raise ValueError("Incomplete method–seed matrix in BatchNorm audit")
                for key, _, color, marker, offset in domains:
                    shifts = np.array([number(r[key]) for r in group])
                    axis.scatter(x+offset+jitter, shifts, color=color, marker=marker,
                                 s=24, alpha=.65, linewidths=.4, edgecolors="white", zorder=3)
                    axis.scatter(x+offset, shifts.mean(), color=color, marker="D", s=62,
                                 edgecolors="#252525", linewidths=.8, zorder=4)
            axis.set_yscale("log")
            axis.set_ylim(min(values)/1.6, max(values)*1.6)
            axis.set_xlim(-.5, len(methods)-.5)
            axis.set_xticks(range(len(methods)), [METHOD_LABELS[m].replace("Class-Aware ", "Class-Aware\n") for m in methods])
            axis.tick_params(axis="x", labelsize=9)
            if row_index == 0:
                axis.set_title(f"{layer} · {256 if layer == 'bn1' else 168} channels", fontsize=11)
            if column == 0:
                axis.set_ylabel(f"{DIRECTION_LABELS[direction]}\nBN shift (log scale)")
            finish_axis(axis)
    handles = [Line2D([], [], color=color, marker=marker, linestyle="none", label=label,
                      markersize=6) for _, label, color, marker, _ in domains]
    handles.append(Line2D([], [], color="#252525", marker="D", linestyle="none",
                          label="Mean over 5 seeds", markersize=6))
    figure.suptitle("BatchNorm activation shift at selected models", fontsize=14)
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .94),
                  ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, .88), h_pad=2, w_pad=2)
    save_png(figure, output, "figure4_batchnorm_shift")


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
    plot_batchnorm_shift(read_rows(result_file(audit_root/"appendix_bn_runs.csv")), output)
    print(f"Saved 4 canonical mechanism PNG figures: {output}")
