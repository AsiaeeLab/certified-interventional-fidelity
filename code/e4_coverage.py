from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cif import BettingCSTracker, cs_radius
from compile import CompiledHead, compile_const
from mnist_utils import load_mnist_datasets, make_loader, sample_calibration_subset
from models import load_model, train_mnist
from pruning import compute_pruning_scores, make_prune_plan
from utils import artifacts_dir, ensure_dir, get_device, results_dir, set_seed


@torch.no_grad()
def _eval_z_batch(
    *,
    a_test: torch.Tensor,
    logits_base: torch.Tensor,
    W_out: torch.Tensor,
    head: CompiledHead,
    i: torch.Tensor,
    j: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    kept = head.kept_idx
    a_i_kept = a_test[i][:, kept]
    a_j_kept = a_test[j][:, kept]
    mask_f = mask.to(dtype=a_test.dtype)
    a_h = torch.where(mask, a_j_kept, a_i_kept)
    delta = (a_j_kept - a_i_kept) * mask_f
    W_kept = W_out[:, kept]
    logits_dense = logits_base[i] + delta @ W_kept.T
    logits_red = a_h @ head.W_new.T + head.b_new
    return (logits_dense.argmax(dim=-1) != logits_red.argmax(dim=-1)).to(dtype=torch.float32)


def _sample_i_j(N: int, batch_size: int, rng: torch.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    i = torch.randint(0, N, (batch_size,), generator=rng, device=device)
    j = torch.randint(0, N, (batch_size,), generator=rng, device=device)
    neq = j != i
    while not bool(neq.all()):
        j2 = torch.randint(0, N, (batch_size,), generator=rng, device=device)
        j = torch.where(neq, j, j2)
        neq = j != i
    return i, j


def _stress_probs_from_sensitivity(sens: torch.Tensor, *, target_sum: float) -> torch.Tensor:
    K = sens.numel()
    target_sum = float(target_sum)
    if target_sum <= 0.0:
        return torch.zeros_like(sens)
    if target_sum >= float(K):
        return torch.ones_like(sens)

    s = sens.detach().to(dtype=torch.float64).clamp_min(0.0)
    p = torch.zeros((K,), dtype=torch.float64, device=sens.device)
    remaining = torch.ones((K,), dtype=torch.bool, device=sens.device)
    remaining_target = target_sum

    while True:
        idx = remaining.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            break
        w = s[idx]
        w_sum = float(w.sum().item())
        if w_sum <= 0.0:
            p[idx] = remaining_target / float(idx.numel())
            break
        p_prov = remaining_target * (w / w.sum())
        exceed = p_prov > 1.0
        if not bool(exceed.any()):
            p[idx] = p_prov
            break
        exceed_idx = idx[exceed]
        p[exceed_idx] = 1.0
        remaining[exceed_idx] = False
        remaining_target = target_sum - float(p.sum().item())
        if remaining_target <= 0.0:
            break

    return p.to(dtype=sens.dtype)


@torch.no_grad()
def _sample_mask_adaptive_and_weight(
    *,
    batch_size: int,
    p: float,
    p_k: torch.Tensor,
    alpha: float,
    rng: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep = p_k.numel()
    use_tilde = torch.rand((batch_size,), generator=rng, device=device) < float(alpha)
    mask_pi = torch.rand((batch_size, keep), generator=rng, device=device) < float(p)
    mask_tilde = torch.rand((batch_size, keep), generator=rng, device=device) < p_k.view(1, -1)
    mask = torch.where(use_tilde.view(-1, 1), mask_tilde, mask_pi)

    eps = 1e-6
    p_kc = p_k.clamp(min=eps, max=1.0 - eps).to(dtype=torch.float64)
    log_p = math.log(float(p))
    log_1mp = math.log(1.0 - float(p))
    log_pk = torch.log(p_kc)
    log_1mpk = torch.log1p(-p_kc)

    mask_f = mask.to(dtype=torch.float64)
    m_sum = mask_f.sum(dim=1)
    log_pi = m_sum * log_p + (keep - m_sum) * log_1mp
    log_qtilde = (mask_f * log_pk.view(1, -1) + (1.0 - mask_f) * log_1mpk.view(1, -1)).sum(dim=1)

    log_q = torch.logaddexp(
        math.log(1.0 - float(alpha)) + log_pi,
        math.log(float(alpha)) + log_qtilde,
    )
    w = torch.exp(log_pi - log_q).to(dtype=torch.float32)
    W_max = 1.0 / (1.0 - float(alpha))
    w = w.clamp(min=0.0, max=W_max)
    return mask, w


def _cs_radius_array(n_max: int, delta: float, W_max: float) -> np.ndarray:
    ns = np.arange(1, n_max + 1, dtype=np.float64)
    delta_n = 6.0 * float(delta) / (math.pi**2 * ns * ns)
    rad = np.sqrt(np.log(2.0 / delta_n) / (2.0 * ns)) * float(W_max)
    return rad


@torch.no_grad()
def run_e4() -> None:
    ensure_dir(results_dir())
    device = get_device()

    # Setup: model + VBP abstraction at keep=256.
    seed = 0
    set_seed(seed)
    art_dir = artifacts_dir()
    data_dir = Path(__file__).resolve().parent / "data"
    ckpt_path = art_dir / "mnist_mlp_seed0.pt"
    if not ckpt_path.exists():
        res = train_mnist(seed=seed, artifacts_dir=art_dir, data_dir=data_dir, device=device)
        print(f"[E4] Trained MNIST MLP: test_acc={res.test_acc:.4f} ({res.model_path})")
    model = load_model(ckpt_path, device=device)

    train_ds, test_ds = load_mnist_datasets(data_dir)

    # Calibration subset for VBP plan
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

    mask_p = 0.5
    deltas = [0.01, 0.05, 0.10]
    alpha = 0.3
    W_max = 1.0 / (1.0 - alpha)

    # Ground truth estimate: 100,000 IID trials
    n_true = 100_000
    rng_true = torch.Generator(device=device).manual_seed(999)
    z_true = torch.empty((n_true,), dtype=torch.float32, device="cpu")
    offset = 0
    while offset < n_true:
        b = min(512, n_true - offset)
        i, j = _sample_i_j(a_test.shape[0], b, rng_true, device)
        mask = torch.rand((b, head.kept_idx.numel()), generator=rng_true, device=device) < float(mask_p)
        z = _eval_z_batch(a_test=a_test, logits_base=logits_base, W_out=W_out, head=head, i=i, j=j, mask=mask)
        z_true[offset : offset + b] = z.detach().cpu()
        offset += b
    R_true = float(z_true.mean().item())
    print(f"[E4] R_true (approx, n=100000): {R_true:.6f}")

    # Precompute stress distribution p_k for adaptive
    sens = torch.linalg.vector_norm(W_out[:, head.kept_idx], ord=2, dim=0)
    p_k = _stress_probs_from_sensitivity(sens, target_sum=float(mask_p) * float(head.kept_idx.numel()))
    p_k = p_k.clamp(min=0.0, max=1.0).to(device=device, dtype=torch.float32)

    n_reps = 500
    n_trials = 2000

    rad_iid_by_delta = {d: _cs_radius_array(n_trials, d, W_max=1.0) for d in deltas}
    rad_is_by_delta = {d: _cs_radius_array(n_trials, d, W_max=W_max) for d in deltas}

    covered_hoeff = {(d, p): 0 for d in deltas for p in ["iid", "adaptive", "aggressive_peeking"]}
    covered_bet = {(d, p): 0 for d in deltas for p in ["iid", "adaptive", "aggressive_peeking"]}
    check_every = 10
    check_idx = (np.arange(check_every, n_trials + 1, check_every, dtype=np.int64) - 1)

    for r in range(n_reps):
        seed_r = 10_000 + r

        # IID protocol
        rng = torch.Generator(device=device).manual_seed(seed_r)
        z = torch.empty((n_trials,), dtype=torch.float32, device="cpu")
        offset = 0
        while offset < n_trials:
            b = min(256, n_trials - offset)
            i, j = _sample_i_j(a_test.shape[0], b, rng, device)
            mask = torch.rand((b, head.kept_idx.numel()), generator=rng, device=device) < float(mask_p)
            zb = _eval_z_batch(a_test=a_test, logits_base=logits_base, W_out=W_out, head=head, i=i, j=j, mask=mask)
            z[offset : offset + b] = zb.detach().cpu()
            offset += b

        z_np = z.numpy()
        z_list = z_np.astype(np.float64, copy=False).tolist()

        # Precompute IID running means for Hoeffding peeking checks.
        cumsum = np.cumsum(z_np, dtype=np.float64)
        ns = np.arange(1, n_trials + 1, dtype=np.float64)
        r_hat_path = cumsum / ns

        for d in deltas:
            rad_iid = rad_iid_by_delta[d]
            log_thresh = math.log(1.0 / float(d))

            # IID (final step)
            r_hat = float(r_hat_path[-1])
            lo = max(0.0, r_hat - float(rad_iid[-1]))
            hi = min(1.0, r_hat + float(rad_iid[-1]))
            if lo <= R_true <= hi:
                covered_hoeff[(d, "iid")] += 1

            logK = BettingCSTracker._ons_log_capital(z_list, R_true, c=0.5)
            if logK < log_thresh:
                covered_bet[(d, "iid")] += 1

            # Aggressive peeking (check every 10 steps)
            lo_path = np.maximum(0.0, r_hat_path - rad_iid)
            hi_path = np.minimum(1.0, r_hat_path + rad_iid)
            ok = np.all((lo_path[check_idx] <= R_true) & (R_true <= hi_path[check_idx]))
            if bool(ok):
                covered_hoeff[(d, "aggressive_peeking")] += 1

            max_logK = BettingCSTracker._ons_max_log_capital(
                z_list, R_true, check_every=check_every, c=0.5
            )
            if max_logK < log_thresh:
                covered_bet[(d, "aggressive_peeking")] += 1

        # Adaptive IS protocol
        rng = torch.Generator(device=device).manual_seed(seed_r)
        wz = torch.empty((n_trials,), dtype=torch.float32, device="cpu")
        offset = 0
        while offset < n_trials:
            b = min(256, n_trials - offset)
            i, j = _sample_i_j(a_test.shape[0], b, rng, device)
            mask, w = _sample_mask_adaptive_and_weight(
                batch_size=b, p=mask_p, p_k=p_k, alpha=alpha, rng=rng, device=device
            )
            zb = _eval_z_batch(a_test=a_test, logits_base=logits_base, W_out=W_out, head=head, i=i, j=j, mask=mask)
            wz[offset : offset + b] = (w * zb).detach().cpu()
            offset += b

        wz_np = wz.numpy().astype(np.float64, copy=False)
        y_np = (np.clip(wz_np, 0.0, float(W_max)) / float(W_max)).tolist()
        for d in deltas:
            rad_is = rad_is_by_delta[d]
            log_thresh = math.log(1.0 / float(d))

            r_hat_is = float(wz_np.mean())
            lo = max(0.0, r_hat_is - float(rad_is[-1]))
            hi = min(1.0, r_hat_is + float(rad_is[-1]))
            if lo <= R_true <= hi:
                covered_hoeff[(d, "adaptive")] += 1

            logK = BettingCSTracker._ons_log_capital(y_np, R_true / float(W_max), c=0.5)
            if logK < log_thresh:
                covered_bet[(d, "adaptive")] += 1

    rows = []
    for d in deltas:
        for protocol in ["iid", "adaptive", "aggressive_peeking"]:
            cov_h = covered_hoeff[(d, protocol)] / float(n_reps)
            cov_b = covered_bet[(d, protocol)] / float(n_reps)
            rows.append(
                {
                    "protocol": protocol,
                    "n_reps": n_reps,
                    "n_trials": n_trials,
                    "delta": float(d),
                    "R_true": float(R_true),
                    "empirical_coverage_hoeffding": float(cov_h),
                    "coverage_std_hoeffding": float(math.sqrt(cov_h * (1.0 - cov_h) / float(n_reps))),
                    "empirical_coverage_betting": float(cov_b),
                    "coverage_std_betting": float(math.sqrt(cov_b * (1.0 - cov_b) / float(n_reps))),
                }
            )

    pd.DataFrame(rows).to_csv(results_dir() / "e4_coverage.csv", index=False)
