from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


def mnist_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def load_mnist_datasets(data_dir: Path):
    from torchvision import datasets

    transform = mnist_transform()
    train_ds = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)
    return train_ds, test_ds


def make_loader(ds, *, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    pin = device.type == "cuda"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=pin)


@torch.no_grad()
def compute_penultimate_acts(
    model,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    acts: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for xb, yb in loader:
        xb = xb.to(device)
        a = model.forward_features(xb)
        acts.append(a.detach().cpu())
        ys.append(yb.detach().cpu())
    return torch.cat(acts, dim=0), torch.cat(ys, dim=0)


@torch.no_grad()
def compute_test_cache(
    model,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (a_test, logits_base, y_test) on CPU."""
    acts: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for xb, yb in loader:
        xb = xb.to(device)
        a = model.forward_features(xb)
        logit = model.fc_out(a)
        acts.append(a.detach().cpu())
        logits.append(logit.detach().cpu())
        ys.append(yb.detach().cpu())
    return torch.cat(acts, dim=0), torch.cat(logits, dim=0), torch.cat(ys, dim=0)


def sample_calibration_subset(train_ds, *, n_cal: int, seed: int) -> Subset:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(train_ds), generator=g)[:n_cal].tolist()
    return Subset(train_ds, idx)

