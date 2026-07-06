from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from cif import cs_radius
from compile import CompiledHead, compile_const
from mnist_utils import load_mnist_datasets, make_loader, sample_calibration_subset
from models import load_model, train_mnist
from pruning import compute_pruning_scores, make_prune_plan
from utils import artifacts_dir, ensure_dir, get_device, results_dir, set_seed


@torch.no_grad()
def _eval_logits_batch(
    *,
    a_test: torch.Tensor,
    logits_base: torch.Tensor,
    W_out: torch.Tensor,
    head: CompiledHead,
    i: torch.Tensor,
    j: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kept = head.kept_idx
    a_i_kept = a_test[i][:, kept]
    a_j_kept = a_test[j][:, kept]
    mask_f = mask.to(dtype=a_test.dtype)
    a_h = torch.where(mask, a_j_kept, a_i_kept)
    delta = (a_j_kept - a_i_kept) * mask_f
    W_kept = W_out[:, kept]
    logits_dense = logits_base[i] + delta @ W_kept.T
    logits_red = a_h @ head.W_new.T + head.b_new
    return logits_dense, logits_red


def _sample_i_j(N: int, batch_size: int, rng: torch.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    i = torch.randint(0, N, (batch_size,), generator=rng, device=device)
    j = torch.randint(0, N, (batch_size,), generator=rng, device=device)
    neq = j != i
    while not bool(neq.all()):
        j2 = torch.randint(0, N, (batch_size,), generator=rng, device=device)
        j = torch.where(neq, j, j2)
        neq = j != i
    return i, j


def run_e3() -> None:
    ensure_dir(results_dir())
    device = get_device()

    # Reuse the E1-trained model (train if missing).
    seed = 0
    set_seed(seed)
    art_dir = artifacts_dir()
    data_dir = Path(__file__).resolve().parent / "data"
    ckpt_path = art_dir / "mnist_mlp_seed0.pt"
    if not ckpt_path.exists():
        res = train_mnist(seed=seed, artifacts_dir=art_dir, data_dir=data_dir, device=device)
        print(f"[E3] Trained MNIST MLP: test_acc={res.test_acc:.4f} ({res.model_path})")
    model = load_model(ckpt_path, device=device)

    train_ds, test_ds = load_mnist_datasets(data_dir)

    # Calibration for VBP plan (keep=256)
    cal_subset = sample_calibration_subset(train_ds, n_cal=2000, seed=seed)
    cal_loader = make_loader(cal_subset, batch_size=512, shuffle=False, device=device)
    a_cal_cpu, y_cal_cpu = [], []
    with torch.no_grad():
        for xb, yb in cal_loader:
            xb = xb.to(device)
            a = model.forward_features(xb)
            a_cal_cpu.append(a.detach().cpu())
            y_cal_cpu.append(yb.detach().cpu())
    a_cal = torch.cat(a_cal_cpu, dim=0)
    y_cal = torch.cat(y_cal_cpu, dim=0)

    W_out_cpu = model.fc_out.weight.detach().cpu()
    b_out_cpu = model.fc_out.bias.detach().cpu()
    scores = compute_pruning_scores(a_cal, y_cal, W_out_cpu, b_out_cpu)
    kept_idx, pruned_idx, constants = make_prune_plan("vbp", 256, scores, seed=0)
    head_cpu = compile_const(W_out_cpu, b_out_cpu, kept_idx, pruned_idx, constants)
    head = CompiledHead(
        kept_idx=head_cpu.kept_idx.to(device),
        W_new=head_cpu.W_new.to(device),
        b_new=head_cpu.b_new.to(device),
    )

    # Test cache
    test_loader = make_loader(test_ds, batch_size=1024, shuffle=False, device=device)
    a_test_cpu, logits_base_cpu = [], []
    with torch.no_grad():
        for xb, _yb in test_loader:
            xb = xb.to(device)
            a = model.forward_features(xb)
            logits = model.fc_out(a)
            a_test_cpu.append(a.detach().cpu())
            logits_base_cpu.append(logits.detach().cpu())
    a_test = torch.cat(a_test_cpu, dim=0).to(device)
    logits_base = torch.cat(logits_base_cpu, dim=0).to(device)
    W_out = model.fc_out.weight.detach().to(device)

    mask_ps = [0.05, 0.1, 0.2, 0.5]
    metrics = ["zero_one", "clipped_kl", "clipped_l2"]
    n = 10_000
    delta = 0.05
    radius = cs_radius(n, delta)

    rows: list[dict] = []
    for mask_p in mask_ps:
        for metric in metrics:
            rng = torch.Generator(device=device).manual_seed(123)
            z_out = torch.empty((n,), dtype=torch.float32, device="cpu")
            offset = 0
            while offset < n:
                b = min(256, n - offset)
                i, j = _sample_i_j(a_test.shape[0], b, rng, device)
                mask = torch.rand((b, head.kept_idx.numel()), generator=rng, device=device) < float(mask_p)
                logits_dense, logits_red = _eval_logits_batch(
                    a_test=a_test,
                    logits_base=logits_base,
                    W_out=W_out,
                    head=head,
                    i=i,
                    j=j,
                    mask=mask,
                )

                if metric == "zero_one":
                    z = (logits_dense.argmax(dim=-1) != logits_red.argmax(dim=-1)).to(dtype=torch.float32)
                elif metric == "clipped_kl":
                    log_p = F.log_softmax(logits_dense, dim=-1)
                    log_q = F.log_softmax(logits_red, dim=-1)
                    p = log_p.exp()
                    kl = (p * (log_p - log_q)).sum(dim=-1)
                    z = kl.clamp(max=1.0).to(dtype=torch.float32)
                elif metric == "clipped_l2":
                    p = F.softmax(logits_dense, dim=-1)
                    q = F.softmax(logits_red, dim=-1)
                    l2 = torch.linalg.vector_norm(p - q, ord=2, dim=-1)
                    z = l2.clamp(max=1.0).to(dtype=torch.float32)
                else:
                    raise ValueError(metric)

                z_out[offset : offset + b] = z.detach().cpu()
                offset += b

            r_hat = float(z_out.mean().item())
            f_hat = 1.0 - r_hat
            lcb_f = 1.0 - (r_hat + radius)
            ucb_f = 1.0 - max(0.0, r_hat - radius)
            rows.append(
                {
                    "mask_p": float(mask_p),
                    "metric": metric,
                    "r_hat": r_hat,
                    "radius": float(radius),
                    "f_hat": float(f_hat),
                    "lcb_f": float(lcb_f),
                    "ucb_f": float(ucb_f),
                }
            )

    pd.DataFrame(rows).to_csv(results_dir() / "e3_sensitivity.csv", index=False)

