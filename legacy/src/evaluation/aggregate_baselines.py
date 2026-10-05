import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_DIR / "results" / "baseline"
SEED = 42

COLUMNS = [
    "Experiment", "Feature space", "Train", "Test",
    "PR-AUC (AP)", "ROC-AUC", "F1", "Recall", "FPR",
]
METRICS = {
    "PR-AUC (AP)": "pr_auc",
    "ROC-AUC": "roc_auc",
    "F1": "f1",
    "Recall": "recall",
    "FPR": "fpr",
}


def result_row(dataset):
    path = RESULT_DIR / f"{dataset}_seed{SEED}.json"
    with path.open(encoding="utf-8") as file:
        result = json.load(file)

    expected = "Source supervised" if dataset == "unsw" else "Target supervised reference"
    if result.get("experiment") != expected or result.get("seed") != SEED:
        raise ValueError(f"Sai experiment hoặc seed trong {path}")
    row = {
        "Experiment": result["experiment"],
        "Feature space": result["feature_space"],
        "Train": result["train_domain"],
        "Test": result["test_domain"],
    }
    for title, key in METRICS.items():
        row[title] = result[key]
    return row


def main():
    rows = [result_row("unsw"), result_row("cicids")]
    direct = {
        "Experiment": "Direct source→target",
        "Feature space": "heterogeneous",
        "Train": "UNSW",
        "Test": "CICIDS",
    }
    for title in METRICS:
        direct[title] = "N/A"
    rows.append(direct)

    table = pd.DataFrame(rows, columns=COLUMNS)
    output_path = RESULT_DIR / "baseline_table.csv"
    table.to_csv(output_path, index=False)
    print(table.to_string(index=False))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
