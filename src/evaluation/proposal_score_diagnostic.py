import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def score_diagnostics(labels, scores, source_threshold):
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)

    if y.ndim != 1 or s.shape != y.shape or not len(y):
        raise ValueError("Labels/scores phải là hai vector cùng kích thước")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("Labels phải là 0/1 và scores phải hữu hạn")
    if not np.isfinite(source_threshold):
        raise ValueError("Threshold phải hữu hạn")
    if len(np.unique(y)) != 2:
        raise ValueError("Diagnostic này cần cả benign và attack")

    def at_threshold(threshold):
        pred = (s >= threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

        return {
            "threshold": float(threshold),
            "recall": float(tp / (tp + fn)),
            "fpr": float(fp / (fp + tn)),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    precision, recall, thresholds = precision_recall_curve(y, s)

    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(
        2 * p * r,
        p + r,
        out=np.zeros_like(p),
        where=(p + r) > 0,
    )
    best = int(np.argmax(f1))

    distributions = {}
    for label, name in ((0, "benign"), (1, "attack")):
        values = s[y == label]
        distributions[name] = {
            "count": int(len(values)),
            "mean": float(values.mean()),
            "quantiles": dict(
                zip(
                    ("p05", "p25", "p50", "p75", "p95"),
                    np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95]).tolist(),
                )
            ),
            "fraction_below_source_threshold": float(
                np.mean(values < source_threshold)
            ),
        }

    return {
        "attack_prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "source_threshold_metrics": at_threshold(source_threshold),
        "oracle_f1_diagnostic_only": at_threshold(thresholds[best]),
        "score_distributions": distributions,
        "pr_curve": {
            "threshold": thresholds.tolist(),
            "precision": p.tolist(),
            "recall": r.tolist(),
            "f1": f1.tolist(),
        },
    }


def history_diagnostics(history):
    if not history:
        raise ValueError("Checkpoint không có training history")

    ap_curve = [
        {
            "epoch": int(row["epoch"]),
            "source_val_ap": float(row["source_val_ap"]),
        }
        for row in history
    ]

    initial_ap = ap_curve[0]["source_val_ap"]
    best = max(ap_curve, key=lambda row: row["source_val_ap"])

    alignment_curve = []
    for row in history:
        alignment = row.get("class_aware_alignment")
        if not alignment:
            continue

        aligned = int(alignment["aligned_batches"])
        skipped = int(alignment["skipped_batches"])
        total = aligned + skipped

        alignment_curve.append(
            {
                "epoch": int(row["epoch"]),
                "aligned_batches": aligned,
                "skipped_batches": skipped,
                "two_class_alignment_rate": aligned / total if total else None,
                "both_classes_eligible_batches": alignment[
                    "both_classes_eligible_batches"
                ],
                "eligible_batch_counts": alignment["eligible_batch_counts"],
                "pseudo_labels": row.get("pseudo_label_acceptance"),
            }
        )

    return {
        "source_ap_curve": ap_curve,
        "initial_source_ap": initial_ap,
        "best_source_ap_epoch": best["epoch"],
        "best_source_ap": best["source_val_ap"],
        "last_source_ap": ap_curve[-1]["source_val_ap"],
        "maximum_drop_from_initial": max(
            0.0,
            initial_ap - min(row["source_val_ap"] for row in ap_curve),
        ),
        "class_aware_alignment_curve": alignment_curve,
    }
