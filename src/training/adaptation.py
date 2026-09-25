"""Huấn luyện encoder CICIDS bằng MMD; giữ nguyên model UNSW."""
import torch

from models.target_encoder import TargetEncoder, TargetModel


def mmd_loss(source_features, target_features):
    """MMD với RBF kernel; bandwidth lấy từ batch nguồn, không dùng nhãn đích."""
    source_distance = torch.cdist(source_features, source_features).square()
    bandwidth = source_distance.detach().mean().clamp_min(1e-6)
    target_distance = torch.cdist(target_features, target_features).square()
    cross_distance = torch.cdist(source_features, target_features).square()

    source_kernel = torch.exp(-source_distance / (2 * bandwidth))
    target_kernel = torch.exp(-target_distance / (2 * bandwidth))
    cross_kernel = torch.exp(-cross_distance / (2 * bandwidth))
    return source_kernel.mean() + target_kernel.mean() - 2 * cross_kernel.mean()


def train_adaptation(source_model, source_loader, target_loader, target_dim,
                     epochs=10, learning_rate=0.001):
    device = next(source_model.parameters()).device
    source_model.eval()
    for parameter in source_model.parameters():
        parameter.requires_grad = False

    target_model = TargetModel(
        TargetEncoder(input_dim=target_dim), source_model.classifier
    ).to(device)
    optimizer = torch.optim.Adam(target_model.encoder.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        target_model.encoder.train()
        target_model.classifier.eval()
        total_loss = 0.0
        steps = 0
        source_batches = iter(source_loader)

        for target_x, _ in target_loader:
            # Đọc lại nguồn nếu hết; không dùng cycle() vì sẽ giữ batch trong RAM.
            try:
                source_x, _ = next(source_batches)
            except StopIteration:
                source_batches = iter(source_loader)
                try:
                    source_x, _ = next(source_batches)
                except StopIteration:
                    raise ValueError("UNSW train không đủ một batch.") from None

            source_x = source_x.to(device)
            target_x = target_x.to(device)
            with torch.no_grad():
                source_features, _ = source_model(source_x)

            target_features = target_model.encoder(target_x)
            loss = mmd_loss(source_features, target_features)
            if not torch.isfinite(loss):
                raise ValueError("MMD loss có NaN/Inf.")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        if steps == 0:
            raise ValueError("CICIDS adaptation không đủ một batch. Giảm BATCH_SIZE.")
        print(f"Epoch {epoch + 1}/{epochs} | MMD: {total_loss / steps:.6f}")

    target_model.eval()
    return target_model
