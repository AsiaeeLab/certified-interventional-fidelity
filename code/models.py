"""MNIST MLP model -- standalone reimplementation for CIF experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from utils import ensure_dir, set_seed


class MLPClassifier(nn.Module):
    """784 -> hidden_dim -> hidden_dim -> out_dim with ReLU."""

    def __init__(self, in_dim: int = 784, hidden_dim: int = 512, out_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.hidden_dim = hidden_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer activations (after ReLU)."""
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.forward_features(x)
        return self.fc_out(a)


@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    test_acc: float


def _train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
) -> float:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(1, total)


def train_mnist(
    *,
    seed: int,
    artifacts_dir: Path,
    data_dir: Path,
    device: torch.device,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> TrainResult:
    """Train MNIST MLP and save checkpoint."""

    from torchvision import datasets, transforms

    set_seed(seed)
    ensure_dir(artifacts_dir)
    ensure_dir(data_dir)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_ds = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin
    )

    model = MLPClassifier(in_dim=784, hidden_dim=512, out_dim=10)
    test_acc = _train_loop(model, train_loader, test_loader, device=device, epochs=epochs, lr=lr)

    model_path = artifacts_dir / f"mnist_mlp_seed{seed}.pt"
    torch.save({"model_state_dict": model.state_dict(), "seed": seed}, model_path)
    return TrainResult(model_path=model_path, test_acc=test_acc)


def load_model(
    model_path: Path,
    *,
    in_dim: int = 784,
    hidden_dim: int = 512,
    out_dim: int = 10,
    device: torch.device,
) -> MLPClassifier:
    ckpt = torch.load(model_path, map_location="cpu")
    model = MLPClassifier(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model

