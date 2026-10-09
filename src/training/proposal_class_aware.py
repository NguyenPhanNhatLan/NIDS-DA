"""Class-conditional single-RBF MMD using confident target predictions."""
from pathlib import Path
import torch
from training.adaptation import mmd_loss
from training.proposal_data import ParquetBatchStream

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/proposal_class_aware_v2.json'


def pseudo_label_counts(logits, confidence):
    probabilities, labels = logits.detach().softmax(dim=1).max(dim=1)
    accepted = probabilities >= confidence
    counts = torch.bincount(labels[accepted], minlength=2).tolist()
    return int(accepted.sum()), len(labels), counts


def class_aware_mmd_loss(source, source_labels, target, target_logits, confidence):
    probabilities, labels = target_logits.detach().softmax(dim=1).max(dim=1)
    accepted = probabilities >= confidence
    losses, bandwidths = [], []
    for label in (0, 1):
        source_class = source[source_labels == label]
        target_class = target[accepted & (labels == label)]
        if len(source_class) >= 2 and len(target_class) >= 2:
            loss, bandwidth = mmd_loss(source_class, target_class)
            losses.append(loss)
            bandwidths.append(bandwidth)
    if not losses:
        zero = (source.sum() + target.sum()) * 0.0
        return zero, zero.detach()
    return torch.stack(losses).mean(), torch.stack(bandwidths).mean()


@torch.no_grad()
def audit_pseudo_labels(model, path, confidence=0.8):
    model.eval()
    device = next(model.parameters()).device
    rows = accepted = 0
    counts = [0, 0]
    for features in ParquetBatchStream(path, 1024, False, 0, False):
        _, logits = model(features.to(device))
        batch_accepted, batch_rows, batch_counts = pseudo_label_counts(logits, confidence)
        rows += batch_rows
        accepted += batch_accepted
        counts = [a + b for a, b in zip(counts, batch_counts)]
    return {'target_rows': rows, 'accepted': accepted, 'pseudo_normal': counts[0],
            'pseudo_attack': counts[1], 'accepted_rate': accepted / rows if rows else 0.0,
            'target_labels_used': False}
