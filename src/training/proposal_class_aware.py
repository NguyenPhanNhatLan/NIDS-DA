"""Class-conditional single-RBF MMD using confident target predictions."""
import torch
from training.adaptation import mmd_loss
from training.proposal_data import ParquetBatchStream

def pseudo_label_counts(logits, confidence):
    probabilities, labels = logits.detach().softmax(dim=1).max(dim=1)
    accepted = probabilities >= confidence
    counts = torch.bincount(labels[accepted], minlength=2).tolist()
    return int(accepted.sum()), len(labels), counts


def class_aware_mmd_loss(source, source_labels, target, target_logits, confidence):
    probabilities, labels = target_logits.detach().softmax(dim=1).max(dim=1)
    accepted = probabilities >= confidence
    # Frozen policy: align both classes or skip the entire class-aware term.
    # Never lower confidence or manufacture minority pseudo-labels.
    if any(int((source_labels == label).sum()) < 2
           or int((accepted & (labels == label)).sum()) < 2 for label in (0, 1)):
        zero = (source.sum() + target.sum()) * 0.0
        return zero, zero.detach()
    losses, bandwidths = [], []
    for label in (0, 1):
        source_class = source[source_labels == label]
        target_class = target[accepted & (labels == label)]
        loss, bandwidth = mmd_loss(source_class, target_class)
        losses.append(loss)
        bandwidths.append(bandwidth)
    return torch.stack(losses).mean(), torch.stack(bandwidths).mean()


def class_aware_batch_stats(source_labels, target_logits, confidence):
    probabilities, labels = target_logits.detach().softmax(dim=1).max(dim=1)
    accepted = probabilities >= confidence
    predicted = torch.bincount(labels, minlength=2).tolist()
    counts = torch.bincount(labels[accepted], minlength=2).tolist()
    source_counts = torch.bincount(source_labels, minlength=2).tolist()
    eligible = [label for label in (0, 1) if source_counts[label] >= 2 and counts[label] >= 2]
    return {'predicted_per_class': predicted, 'accepted_per_class': counts,
            'eligible_classes': eligible, 'active_classes': eligible if len(eligible) == 2 else []}


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


def class_separation_loss(source, source_labels, target, target_logits, confidence, margin):
    """Keep opposite-class unit centroids apart in both domains and across domains.

    Margin is a minimum squared Euclidean distance. Target groups use detached
    confident predictions; skip the batch unless both classes have >= 2 samples
    in both domains, matching the class-aware MMD eligibility policy.
    """
    probabilities, labels = target_logits.detach().softmax(dim=1).max(dim=1)
    accepted = probabilities >= confidence
    if any(int((source_labels == label).sum()) < 2
           or int((accepted & (labels == label)).sum()) < 2 for label in (0, 1)):
        return (source.sum() + target.sum()) * 0.0
    source_centers = torch.stack([source[source_labels == c].mean(0) for c in (0, 1)])
    target_centers = torch.stack([target[accepted & (labels == c)].mean(0) for c in (0, 1)])
    source_centers = torch.nn.functional.normalize(source_centers, dim=1)
    target_centers = torch.nn.functional.normalize(target_centers, dim=1)
    distances = torch.stack([
        (source_centers[0] - source_centers[1]).square().sum(),
        (target_centers[0] - target_centers[1]).square().sum(),
        (source_centers[0] - target_centers[1]).square().sum(),
        (source_centers[1] - target_centers[0]).square().sum(),
    ])
    return torch.relu(margin - distances).mean()
