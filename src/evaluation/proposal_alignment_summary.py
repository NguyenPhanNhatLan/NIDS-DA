import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from project_utils.result_files import publish_results, result_file

from project_utils import sha256
from experiments.proposal_protocol import ROOT
from evaluation.proposal_results import load_alignment_runs as collect_runs


def summarize_runs(runs, tolerance=1e-12):
    groups = defaultdict(list)
    for row in runs:
        groups[row["direction"], row["method"]].append(row)
    summary = []
    for (direction, method), rows in groups.items():
        reduced = [row for row in rows if row["delta_mmd2"] < -tolerance]
        summary.append(
            {
                "direction": direction,
                "method": method,
                "total_runs": len(rows),
                "adaptation_selected": sum(
                    row["selected_stage"] == "adaptation" for row in rows
                ),
                "mmd_decreased": len(reduced),
                "ap_increased_given_mmd_decrease": sum(
                    row["delta_ap"] > tolerance for row in reduced
                ),
                "ap_unchanged_given_mmd_decrease": sum(
                    abs(row["delta_ap"]) <= tolerance for row in reduced
                ),
                "ap_decreased_given_mmd_decrease": sum(
                    row["delta_ap"] < -tolerance for row in reduced
                ),
            }
        )
    return summary


def export(diagnostics_root, output, tolerance=1e-12):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    if output.exists() or result_file(output / "export_manifest.json").exists():
        raise FileExistsError(f"Choose a new export directory: {output}")
    runs, revision, input_paths = collect_runs(diagnostics_root)
    summary = summarize_runs(runs, tolerance)
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in (
        ("alignment_runs.csv", runs),
        ("alignment_summary.csv", summary),
    ):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "| Direction | Method | Adaptation selected | MMD decreased | AP ↑ / = / ↓ when MMD decreased |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary:
        lines.append(
            f"| {row['direction']} | {row['method']} | "
            f"{row['adaptation_selected']}/{row['total_runs']} | {row['mmd_decreased']} | "
            f"{row['ap_increased_given_mmd_decrease']} / "
            f"{row['ap_unchanged_given_mmd_decrease']} / "
            f"{row['ap_decreased_given_mmd_decrease']} |"
        )
    note = (
        "Target-validation, seeds from the locked suite. MMD before/after compares "
        "source-pretrained with the selected checkpoint, not necessarily the final epoch. "
        "best_epoch=0 means source_pretrained; positive epochs mean adaptation. "
        f"Changes use full precision and absolute tolerance {tolerance:g}. "
        "MMD reduction percent is calculated per run. No inference or training is performed."
    )
    (output / "alignment_summary.md").write_text(
        "\n".join(lines) + "\n\n" + note + "\n"
    )
    (output / "export_manifest.json").write_text(
        json.dumps(
            {
                **revision,
                "data_role": "target_validation",
                "absolute_change_tolerance": tolerance,
                "diagnostics_root": str(diagnostics_root.resolve()),
                "verification_scope": "development lock and its artifacts; diagnostic references and AP checked against locked results; raw/prepared data not reread",
                "inputs": [
                    {"path": str(path), "sha256": sha256(path)}
                    for path in sorted(input_paths)
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print("\n".join(lines))
    print(f"\nSaved {len(runs)} runs and {len(summary)} groups: {output}")


def main():
    parser = argparse.ArgumentParser(description='Export per-seed alignment changes and checkpoint-selection counts from saved JSON.')
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=ROOT / "results/canonical/diagnostics_target_val",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    export(args.diagnostics_root, args.output, args.tolerance)
    publish_results(args.output)


if __name__ == "__main__":
    main()
