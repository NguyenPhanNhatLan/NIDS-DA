from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from torch import nn



from features.common_schema import COMMON_FEATURES
from training.dataloaders import create_dataloader


Array = np.ndarray


@dataclass(frozen=True)
class AdaptationConfig:
    hidden_layers: tuple[int, int] = (64, 32)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 100
    mmd_kernel_scales: tuple[float, ...] = (1.0,)
    patience: int = 10
    min_epochs: int = 10
    min_delta: float = 1e-4
    mmd_lambda: float = 0.1
    coral_lambda: float = 0.0
    mmd_bandwidth: float | None = None
    alignment: str = "marginal"
    warmup_epochs: int = 0
    ramp_epochs: int = 0
    adaptive_bandwidth: bool = False
    pseudo_confidence: float = 0.95
    random_state: int = 42


def estimate_rbf_bandwidth(
    source: Array,
    target: Array,
    *,
    random_state: int,
    max_samples: int = 1024,
) -> float:
    rng = np.random.default_rng(random_state)
    combined = np.vstack([source, target])
    sample_size = min(max_samples, len(combined))
    combined = combined[
        rng.choice(len(combined), size=sample_size, replace=False)
    ]
    first = combined[rng.integers(0, sample_size, size=max_samples)]
    second = combined[rng.integers(0, sample_size, size=max_samples)]
    distances = np.linalg.norm(first - second, axis=1)
    positive = distances[distances > 1e-12]
    return float(max(np.median(positive), 1e-3)) if len(positive) else 1.0


def linear_rbf_mmd(source: torch.Tensor, target: torch.Tensor, bandwidth: float):
    usable = min(len(source), len(target))
    usable -= usable % 2
    if usable < 2:
        return (source.sum() + target.sum()) * 0.0

    source = source[:usable]
    target = target[:usable]

    def kernel(left, right):
        distance_sq = (left - right).square().sum(dim=1)
        return torch.exp(-distance_sq / (2.0 * bandwidth**2))

    return (
        kernel(source[0::2], source[1::2])
        + kernel(target[0::2], target[1::2])
        - kernel(source[0::2], target[1::2])
        - kernel(source[1::2], target[0::2])
    ).mean()

def block_wise_rbf_mmd(source: torch.Tensor, target: torch.Tensor, bandwidth: float, block_size: int = 128, *, kernel_scales: tuple[float, ...] = (1.0,)):
    
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be finite and positive.")
    if not kernel_scales or any(not np.isfinite(scale) or scale <= 0.0 for scale in kernel_scales):
        raise ValueError("kernel_scales must be a non-empty tuple of positive finite values.")
    
    
    usable = min(len(source), len(target))
    usable -= usable % block_size
    if usable < block_size:
        return (source.sum() + target.sum()) * 0.0

    idx_s = torch.randperm(usable, device=source.device)
    idx_t = torch.randperm(usable, device=target.device)

    s_blocks = source[idx_s].view(-1, block_size, source.size(1)).contiguous()
    t_blocks = target[idx_t].view(-1, block_size, target.size(1)).contiguous()

    bandwidth_sq = float(bandwidth ** 2)

    def compute_kernel(x, y):
        x_sq = x.square().sum(dim=-1, keepdim=True)
        y_sq = y.square().sum(dim=-1).unsqueeze(1)
        xy = torch.bmm(x, y.transpose(1, 2))
        
        dist_sq = x_sq + y_sq - 2.0 * xy
        dist_sq = torch.clamp(dist_sq, min=0.0)
        
        kernel_sum = torch.zeros_like(dist_sq)
        for scale in kernel_scales:
            kernel_sum = kernel_sum + torch.exp(-dist_sq / (2.0 * (bandwidth_sq * scale**2)))
        return kernel_sum / len(kernel_scales)
    

    k_xx = compute_kernel(s_blocks, s_blocks)
    k_yy = compute_kernel(t_blocks, t_blocks)
    k_xy = compute_kernel(s_blocks, t_blocks)

    mask = torch.eye(block_size, dtype=torch.bool, device=source.device).unsqueeze(0)
    k_xx_sum = k_xx.masked_fill(mask, 0.0).sum(dim=(1, 2)) / (block_size * (block_size - 1))
    k_yy_sum = k_yy.masked_fill(mask, 0.0).sum(dim=(1, 2)) / (block_size * (block_size - 1))
    k_xy_sum = k_xy.sum(dim=(1, 2)) / (block_size * block_size)

    return (k_xx_sum + k_yy_sum - 2.0 * k_xy_sum).mean()

def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Deep CORAL: squared covariance distance / (4*d*d), Sun & Saenko 2016.

    Covariances use n-1; unequal batch sizes are supported. A singleton
    cannot estimate covariance and contributes a differentiable zero.
    """
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("CORAL requires two matrices with the same feature dimension")
    if min(len(source), len(target)) < 2:
        return (source.sum() + target.sum()) * 0.0
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = source_centered.T @ source_centered / (len(source) - 1)
    target_cov = target_centered.T @ target_centered / (len(target) - 1)
    return (source_cov - target_cov).square().sum() / (4 * source.shape[1] ** 2)


def conditional_rbf_mmd(source, target, labels, probabilities, bandwidth, *, kernel_scales=(1.0,), confidence=.95):
    """Class-wise unbiased MMD; supports unequal small groups without padding."""
    probabilities = probabilities.detach()
    accepted = torch.maximum(probabilities, 1 - probabilities) >= confidence
    pseudo = probabilities >= .5
    losses = []
    counts = []
    for c in (0, 1):
        x = source[labels == c][:128]
        selected = accepted & (pseudo == c)
        counts.append(int(selected.sum()))
        y = target[selected][:128]
        if len(x) < 2 or len(y) < 2:
            continue
        def kernel(a, b):
            d = torch.cdist(a, b).square()
            return sum(torch.exp(-d / (2 * (bandwidth * scale)**2)) for scale in kernel_scales) / len(kernel_scales)
        xx, yy, xy = kernel(x, x), kernel(y, y), kernel(x, y)
        losses.append((xx.sum()-xx.diagonal().sum())/(len(x)*(len(x)-1))
                      + (yy.sum()-yy.diagonal().sum())/(len(y)*(len(y)-1)) - 2*xy.mean())
    loss = torch.stack(losses).mean() if losses else (source.sum()+target.sum())*0
    return loss, counts, len(losses)


def rbf_mmd_squared(source: Array, target: Array, bandwidth: float) -> float:
    bandwidth_sq = float(bandwidth**2)

    def mean_kernel(left: Array, right: Array) -> float:
        left_sq = np.sum(left * left, axis=1)[:, None]
        right_sq = np.sum(right * right, axis=1)[None, :]
        distance_sq = np.maximum(left_sq + right_sq - 2.0 * left @ right.T, 0.0)
        return float(np.exp(-distance_sq / (2.0 * bandwidth_sq)).mean())

    value = (
        mean_kernel(source, source)
        + mean_kernel(target, target)
        - 2.0 * mean_kernel(source, target)
    )
    return float(max(value, 0.0))


class MMDNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: tuple[int, int]):
        super().__init__()
        hidden_1, hidden_2 = hidden_layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_2, 1)

    def forward(self, features):
        representation = self.feature_extractor(features)
        logits = self.classifier(representation).squeeze(1)
        return logits, representation


class TorchMMDClassifier:
    def __init__(self, input_dim: int, config: AdaptationConfig):
        if len(config.hidden_layers) != 2:
            raise ValueError("Exactly two hidden layers are required.")
        if any(not np.isfinite(v) or v < 0 for v in (config.mmd_lambda, config.coral_lambda)):
            raise ValueError("Alignment coefficients must be finite and non-negative.")

        if config.alignment not in {"marginal", "conditional"}:
            raise ValueError("Unknown alignment")
        if config.warmup_epochs < 0 or config.ramp_epochs < 0 or not .5 < config.pseudo_confidence <= 1:
            raise ValueError("Invalid schedule or confidence")
        torch.manual_seed(config.random_state)
        self.config = config
        self.model = MMDNetwork(input_dim, config.hidden_layers)
        self.bandwidth_: float | None = None
        self.history_: list[dict[str, float | int]] = []
        self.best_epoch_: int | None = None
        self.best_validation_pr_auc_: float | None = None

    @staticmethod
    def _next_batch(loader, iterator):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def fit(
        self,
        source_features: Array,
        source_labels: Array,
        target_features: Array | None,
        validation_features: Array,
        validation_labels: Array,
        *,
        class_weights: tuple[float, float],
        target_size: int | None = None,
        epoch_callback=None,
    ) -> "TorchMMDClassifier":
        source_features = np.asarray(source_features, dtype=np.float32)
        source_labels = np.asarray(source_labels, dtype=np.float32)
        if (self.config.mmd_lambda > 0.0 or self.config.coral_lambda > 0.0) and target_features is None:
            raise ValueError("target_features are required for MMD or CORAL.")
        if target_features is not None:
            target_features = np.asarray(target_features, dtype=np.float32)
            target_size = len(target_features)
        if target_size is None or target_size <= 0:
            raise ValueError("target_size must be positive.")
        sample_weights = np.where(
            source_labels == 1.0, class_weights[1], class_weights[0]
        ).astype(np.float32)

        source_loader = create_dataloader(
            source_features,
            source_labels,
            batch_size=self.config.batch_size,
            shuffle=True,
            sample_weights=sample_weights,
            seed=self.config.random_state,
        )
        target_loader = (
            create_dataloader(
                target_features,
                batch_size=self.config.batch_size,
                shuffle=True,
                seed=self.config.random_state + 1,
            )
            if target_features is not None
            else None
        )

        if target_features is not None:
            rng = np.random.default_rng(self.config.random_state)
            source_sample = source_features[
                rng.choice(len(source_features), min(2048, len(source_features)), False)
            ]
            target_sample = target_features[
                rng.choice(len(target_features), min(2048, len(target_features)), False)
            ]
            self.bandwidth_ = self.config.mmd_bandwidth or estimate_rbf_bandwidth(
                self.transform(source_sample),
                self.transform(target_sample),
                random_state=self.config.random_state,
            )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        best_score = -np.inf
        best_state = deepcopy(self.model.state_dict())
        best_bandwidth = self.bandwidth_
        epochs_without_improvement = 0

        if epoch_callback is not None:
            epoch_callback(self, 0, "initial")

        for epoch in range(1, self.config.max_epochs + 1):
            self.model.train()
            source_iterator = iter(source_loader)
            target_iterator = iter(target_loader) if target_loader is not None else None
            effective_lambda = self.config.mmd_lambda * (
                0.0 if epoch <= self.config.warmup_epochs else
                min(1.0, (epoch-self.config.warmup_epochs)/max(1, self.config.ramp_epochs)))
            if self.config.adaptive_bandwidth and target_features is not None and effective_lambda > 0:
                measured = estimate_rbf_bandwidth(self.transform(source_sample), self.transform(target_sample), random_state=self.config.random_state)
                self.bandwidth_ = measured if epoch == self.config.warmup_epochs+1 else .9*self.bandwidth_+.1*measured
                self.model.train()
            effective_coral = self.config.coral_lambda * (
                0.0 if epoch <= self.config.warmup_epochs else
                min(1.0, (epoch-self.config.warmup_epochs)/max(1, self.config.ramp_epochs)))
            coral_total = 0.0
            pseudo_counts = [0, 0]
            active_classes = 0
            classification_total = 0.0
            mmd_total = 0.0
            target_steps = (target_size + self.config.batch_size - 1) // self.config.batch_size
            steps = max(len(source_loader), target_steps)

            for _ in range(steps):
                source_batch, source_iterator = self._next_batch(
                    source_loader, source_iterator
                )
                source_x, source_y, weights = source_batch

                optimizer.zero_grad()
                source_logits, source_representation = self.model(source_x)
                classification_losses = criterion(source_logits, source_y)
                classification_loss = (
                    classification_losses * weights
                ).sum() / weights.sum()

                mmd_loss = source_logits.new_tensor(0.0)
                correlation_loss = source_logits.new_tensor(0.0)
                if effective_lambda > 0.0 or effective_coral > 0.0:
                    target_batch, target_iterator = self._next_batch(target_loader, target_iterator)
                    target_logits, target_representation = self.model(target_batch)
                    if effective_coral > 0.0:
                        correlation_loss = coral_loss(source_representation, target_representation)
                    if effective_lambda > 0.0 and self.config.alignment == "conditional":
                        mmd_loss, counts, active = conditional_rbf_mmd(
                            source_representation, target_representation, source_y,
                            torch.sigmoid(target_logits), self.bandwidth_,
                            kernel_scales=self.config.mmd_kernel_scales,
                            confidence=self.config.pseudo_confidence)
                        pseudo_counts = [a+b for a,b in zip(pseudo_counts, counts)]
                        active_classes += active
                    elif effective_lambda > 0.0:
                        mmd_loss = block_wise_rbf_mmd(source_representation, target_representation,
                            self.bandwidth_, kernel_scales=self.config.mmd_kernel_scales)
                loss = classification_loss + effective_lambda * mmd_loss + effective_coral * correlation_loss
                loss.backward()
                optimizer.step()
                classification_total += classification_loss.item()
                mmd_total += mmd_loss.item()
                coral_total += correlation_loss.item()

            validation_score = float(
                average_precision_score(
                    validation_labels,
                    self.predict_proba(validation_features)[:, 1],
                )
            )
            self.history_.append(
                {
                    "epoch": epoch,
                    "effective_lambda": effective_lambda,
                    "bandwidth": self.bandwidth_,
                    "pseudo_benign": pseudo_counts[0],
                    "pseudo_attack": pseudo_counts[1],
                    "active_classes_per_step": active_classes / steps,
                    "classification_loss": classification_total / steps,
                    "mmd_loss": mmd_total / steps,
                    "coral_loss": coral_total / steps,
                    "effective_coral_lambda": effective_coral,
                    "total_loss": (
                        classification_total
                        + effective_lambda * mmd_total
                        + effective_coral * coral_total
                    )
                    / steps,
                    "source_validation_pr_auc": validation_score,
                }
            )

            if epoch_callback is not None:
                epoch_callback(self, epoch, "epoch")

            improvement = validation_score - best_score
            if validation_score > best_score:
                best_score = validation_score
                best_state = deepcopy(self.model.state_dict())
                best_bandwidth = self.bandwidth_
                self.best_epoch_ = epoch
                self.best_validation_pr_auc_ = validation_score

            if improvement > self.config.min_delta:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                epoch >= max(self.config.min_epochs, self.config.warmup_epochs + self.config.ramp_epochs)
                and epochs_without_improvement >= self.config.patience
            ):
                break

        self.model.load_state_dict(best_state)
        self.bandwidth_ = best_bandwidth
        if epoch_callback is not None:
            epoch_callback(self, self.best_epoch_, "selected")
        return self

    def transform(self, features: Array) -> Array:
        loader = create_dataloader(
            np.asarray(features, dtype=np.float32),
            batch_size=self.config.batch_size,
        )
        representations = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                representations.append(self.model.feature_extractor(batch).numpy())
        return np.vstack(representations)

    def predict_proba(self, features: Array) -> Array:
        loader = create_dataloader(
            np.asarray(features, dtype=np.float32),
            batch_size=self.config.batch_size,
        )
        probabilities = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                logits, _ = self.model(batch)
                probabilities.append(torch.sigmoid(logits).numpy())
        attack_probability = np.concatenate(probabilities)
        return np.column_stack([1.0 - attack_probability, attack_probability])


class AdaptationPipeline:
    def __init__(
        self,
        processor: Pipeline,
        network: TorchMMDClassifier,
        feature_names: tuple[str, ...] = tuple(COMMON_FEATURES),
    ):
        self.processor = processor
        self.network = network
        self.feature_names = feature_names

    def _prepare(self, features: pd.DataFrame | Array) -> Array:
        if isinstance(features, pd.DataFrame):
            features = features[list(self.feature_names)]
        return self.processor.transform(features)

    def predict_proba(self, features: pd.DataFrame | Array) -> Array:
        return self.network.predict_proba(self._prepare(features))

    def transform(self, features: pd.DataFrame | Array) -> Array:
        return self.network.transform(self._prepare(features))
