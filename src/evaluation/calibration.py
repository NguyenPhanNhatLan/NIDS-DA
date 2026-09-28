import numpy as np
import torch


def fit_affine_calibrator(source_margin, source_label, target_margin_normal,
                          target_margin_attack):
    source_margin = torch.as_tensor(source_margin, dtype=torch.float64)
    source_label = torch.as_tensor(source_label)
    normal = torch.as_tensor(target_margin_normal, dtype=torch.float64)
    attack = torch.as_tensor(target_margin_attack, dtype=torch.float64)
    for margins in (source_margin, normal, attack):
        if margins.ndim != 1 or margins.numel() == 0 or not torch.isfinite(margins).all():
            raise ValueError("Margins must be nonempty finite vectors")
    if source_label.shape != source_margin.shape or not ((source_label == 0) | (source_label == 1)).all():
        raise ValueError("Source labels must be binary and match source margins")
    if not (source_label == 0).any() or not (source_label == 1).any():
        raise ValueError("Source validation requires both classes")
    source_normal = source_margin[source_label == 0].median().item()
    source_attack = source_margin[source_label == 1].median().item()
    target_normal = normal.median().item()
    target_attack = attack.median().item()
    if source_attack - source_normal <= 1e-6:
        raise ValueError("Source anchors collapsed or reversed")
    if target_attack - target_normal <= 1e-6:
        raise ValueError("Target pseudo anchors collapsed or reversed")
    a = (source_attack - source_normal) / (target_attack - target_normal)
    b = source_normal - a * target_normal
    if not np.isfinite([a, b]).all() or a <= 0:
        raise ValueError("Calibration requires finite a > 0 and finite b")
    return {
        "a": a, "b": b,
        "source_normal": source_normal, "source_attack": source_attack,
        "target_pseudo_normal": target_normal, "target_pseudo_attack": target_attack,
    }


def calibrate_margin(margin, a, b):
    return a * margin + b


def calibrated_probability(margin, a, b):
    margin = torch.as_tensor(margin, dtype=torch.float64)
    return torch.sigmoid(calibrate_margin(margin, a, b))


def select_threshold_by_fpr(labels, scores, max_fpr=0.01):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or labels.shape != scores.shape:
        raise ValueError("Labels and scores must be matching nonempty vectors")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Scores must be finite and labels must be binary")
    normal_count = np.sum(labels == 0)
    attack_count = np.sum(labels == 1)
    if not normal_count or not attack_count or not 0 <= max_fpr <= 1:
        raise ValueError("Both classes and max_fpr in [0, 1] are required")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    thresholds = np.r_[np.nextafter(sorted_scores[0], np.inf), sorted_scores[ends]]
    recall = np.r_[0., np.cumsum(sorted_labels == 1)[ends] / attack_count]
    fpr = np.r_[0., np.cumsum(sorted_labels == 0)[ends] / normal_count]
    valid = np.flatnonzero(fpr <= max_fpr)
    best = valid[np.argmax(recall[valid])]
    return float(thresholds[best])
