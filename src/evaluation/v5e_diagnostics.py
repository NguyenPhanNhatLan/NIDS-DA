"""Read existing fixed-batch MMD diagnostics without loading/training models."""
import argparse
import json
import math
from pathlib import Path


DEFAULT_REPORT = (
    "results/hda_v5e/optimized_v1/affine_fpr_0p02/"
    "development/v5e_vs_v5b_seed42.json"
)
MODELS = ("v5b_initial", "v5d_reference", "v5e_final")
SPACES = ("hidden", "normal", "attack")
KERNELS = ("single_rbf", "mk_rbf")


def summarize(report):
    diagnostics = report["fixed_batch_diagnostics"]
    # Validate before printing any comparison. Bandwidth equality is necessary;
    # identical batch tensors are guaranteed by the report-producing pipeline.
    for space in SPACES:
        bandwidths = []
        for model in MODELS:
            values = diagnostics[model][space]
            for key in (*KERNELS, "fixed_bandwidth_squared"):
                value = values[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Invalid value: {model}/{space}/{key}")
            bandwidths.append(values["fixed_bandwidth_squared"])
        if bandwidths[0] <= 0 or any(value != bandwidths[0] for value in bandwidths):
            raise ValueError(f"Cannot compare {space}: fixed bandwidths differ or are nonpositive")
    lines = []
    for space in SPACES:
        bandwidth = diagnostics[MODELS[0]][space]["fixed_bandwidth_squared"]
        lines.extend([f"\n{space.upper()} | fixed bandwidth squared = {bandwidth:.10g}",
                      f"{'Model':<20} {'Single-RBF':>14} {'MK-MMD':>14}"])
        for model in MODELS:
            values = diagnostics[model][space]
            lines.append(f"{model:<20} {values['single_rbf']:>14.8g} {values['mk_rbf']:>14.8g}")
        for before, after in (("v5b_initial", "v5d_reference"),
                              ("v5d_reference", "v5e_final"),
                              ("v5b_initial", "v5e_final")):
            changes = [diagnostics[after][space][key] - diagnostics[before][space][key] for key in KERNELS]
            lines.append(f"Delta {after} - {before}: Single-RBF={changes[0]:+.8g}, MK-MMD={changes[1]:+.8g}")
    metrics = report.get("metrics", {})
    if metrics:
        lines.extend(["\nClassification (calibrated operating points)", "Model | AP | ROC-AUC | F1 | Recall | FPR"])
        for model in ("v5b", "v5d", "v5e"):
            if model in metrics:
                lines.append(model.upper() + " | " + " | ".join(
                    f"{metrics[model][key]:.6f}" for key in ("pr_auc", "roc_auc", "f1", "recall", "fpr")))
    lines.extend([
        "\nDelta < 0 means lower MMD within the SAME kernel and space.",
        "Compare Normal and Attack separately; do not infer alignment from Hidden alone.",
        "Do not compare loss magnitudes across kernels/spaces. Lower MMD does not guarantee higher AP/F1.",
        "Bandwidth equality is checked here; batch identity relies on the fixed-batch report provenance.",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()
    path = Path(args.report)
    if not path.is_file():
        parser.error(f"Report not found: {path}; provide an existing report with --report")
    try:
        output = summarize(json.loads(path.read_text()))
    except (KeyError, TypeError, ValueError) as error:
        parser.error(f"Invalid diagnostics report: {error}")
    print(f"Report: {path}")
    print(output)


if __name__ == "__main__":
    main()
