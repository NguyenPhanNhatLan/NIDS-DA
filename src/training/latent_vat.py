"""Latent-space Virtual Adversarial Training (VAT) for frozen classifiers."""
import math

import torch
from torch.nn import functional as F


def _l2_normalize_per_sample(x, eps=1e-12):
    if x.ndim != 2:
        raise ValueError("Expected a [batch, dim] tensor")
    norm = torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(eps)
    return x / norm


def latent_vat_loss(
    latent,
    classifier,
    epsilon_ratio=0.05,
    xi_ratio=1e-3,
    power_iterations=1,
):
    """VAT in latent space using a perturbation radius relative to ||z||_2.

    The adversarial direction is estimated on detached latent vectors, so the
    direction search does not create second-order gradients through the target
    adapter. The final KL loss is evaluated on the original latent tensor and
    therefore backpropagates into the target adapter.

    Parameters
    ----------
    latent:
        Post-ReLU shared target latent, shape [B, D].
    classifier:
        Frozen source classifier mapping latent -> logits.
    epsilon_ratio:
        Final perturbation norm as a fraction of each sample's latent L2 norm.
    xi_ratio:
        Small finite-difference radius used while estimating the adversarial
        direction, also relative to each sample's latent norm.
    power_iterations:
        Number of VAT direction-refinement iterations. Use 1 for the first
        controlled experiment.
    """
    if latent.ndim != 2 or len(latent) < 2 or latent.shape[1] < 1:
        raise ValueError("latent must be a nonempty [B,D] tensor with B >= 2")
    if not torch.isfinite(latent).all():
        raise ValueError("latent contains NaN/Inf")
    if (not math.isfinite(epsilon_ratio) or epsilon_ratio < 0
            or not math.isfinite(xi_ratio) or xi_ratio <= 0):
        raise ValueError("epsilon_ratio must be >=0 and xi_ratio must be >0")
    if type(power_iterations) is not int or power_iterations < 1:
        raise ValueError("power_iterations must be a positive integer")

    # Clean target prediction is the consistency target. Detach it: VAT should
    # move the perturbed branch, not chase a moving target within one step.
    with torch.no_grad():
        clean_logits = classifier(latent)
        clean_prob = torch.softmax(clean_logits, dim=1)

    # Search for the locally most prediction-changing direction without
    # backpropagating the direction-search graph into the adapter.
    z0 = latent.detach()
    # Relative radius is more meaningful than a fixed absolute epsilon because
    # the frozen source latent scale is not standardized to unit norm.
    latent_scale = torch.linalg.vector_norm(z0, dim=1, keepdim=True).clamp_min(1.0)

    d = _l2_normalize_per_sample(torch.randn_like(z0))
    for _ in range(power_iterations):
        d = d.detach().requires_grad_(True)
        search_r = xi_ratio * latent_scale * _l2_normalize_per_sample(d)
        adv_logits = classifier(z0 + search_r)
        search_kl = F.kl_div(
            F.log_softmax(adv_logits, dim=1),
            clean_prob,
            reduction="batchmean",
        )
        grad = torch.autograd.grad(search_kl, d, only_inputs=True)[0]
        if not torch.isfinite(grad).all():
            raise ValueError("VAT direction gradient contains NaN/Inf")
        d = _l2_normalize_per_sample(grad).detach()

    r_adv = epsilon_ratio * latent_scale * d
    adv_logits = classifier(latent + r_adv)
    loss = F.kl_div(
        F.log_softmax(adv_logits, dim=1),
        clean_prob,
        reduction="batchmean",
    )
    if not torch.isfinite(loss):
        raise ValueError("VAT loss contains NaN/Inf")
    return loss
