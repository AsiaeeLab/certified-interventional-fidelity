from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cif import BettingCSTracker, PairedBettingCSTracker, cs_radius, cs_radius_is, cs_radius_paired
from compile import CompiledHead, compile_affine, compile_const
from mnist_utils import load_mnist_datasets, make_loader, sample_calibration_subset
from models import load_model, train_mnist
from pruning import compute_affine_map, compute_pruning_scores, make_prune_plan
from utils import artifacts_dir, ensure_dir, get_device, results_dir, set_seed


@torch.no_grad()
def _eval_z_batch(
    *,
    a_test: torch.Tensor,  # (N, d)
    logits_base: torch.Tensor,  # (N, C)
    W_out: torch.Tensor,  # (C, d)
    head: CompiledHead,
    i: torch.Tensor,  # (B,)
    j: torch.Tensor,  # (B,)
    mask: torch.Tensor,  # (B, keep) bool
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
    """Scale nonnegative sens to probabilities in [0,1] summing to target_sum."""

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
def _sample_mask_iid(batch_size: int, keep: int, p: float, rng: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.rand((batch_size, keep), generator=rng, device=device) < float(p)


@torch.no_grad()
def _sample_mask_adaptive_and_weight(
    *,
    batch_size: int,
    p: float,
    p_k: torch.Tensor,  # (keep,)
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


@dataclass(frozen=True)
class CIFRunResult:
    z: np.ndarray  # shape (n_max,)
    w: np.ndarray  # shape (n_max,)


@torch.no_grad()
def _run_iia_trials(
    *,
    a_test: torch.Tensor,
    logits_base: torch.Tensor,
    W_out: torch.Tensor,
    head: CompiledHead,
    mask_p: float,
    sampling: str,
    seed: int,
    n_max: int = 10_000,
    batch_size: int = 256,
    alpha: float = 0.3,
) -> CIFRunResult:
    device = a_test.device
    N = a_test.shape[0]
    keep = head.kept_idx.numel()

    rng = torch.Generator(device=device).manual_seed(seed)

    if sampling == "adaptive":
        sens = torch.linalg.vector_norm(W_out[:, head.kept_idx], ord=2, dim=0)
        p_k = _stress_probs_from_sensitivity(sens, target_sum=float(mask_p) * float(keep))
        p_k = p_k.clamp(min=0.0, max=1.0).to(device=device, dtype=torch.float32)
    else:
        p_k = None

    z_out = torch.empty((n_max,), dtype=torch.float32, device="cpu")
    w_out = torch.empty((n_max,), dtype=torch.float32, device="cpu")
    offset = 0
    while offset < n_max:
        b = min(batch_size, n_max - offset)
        i, j = _sample_i_j(N, b, rng, device)
        if sampling == "iid":
            mask = _sample_mask_iid(b, keep, mask_p, rng, device)
            w = torch.ones((b,), device=device, dtype=torch.float32)
        elif sampling == "adaptive":
            assert p_k is not None
            mask, w = _sample_mask_adaptive_and_weight(
                batch_size=b, p=mask_p, p_k=p_k, alpha=alpha, rng=rng, device=device
            )
        else:
            raise ValueError(f"Unknown sampling: {sampling}")

        z = _eval_z_batch(a_test=a_test, logits_base=logits_base, W_out=W_out, head=head, i=i, j=j, mask=mask)
        z_cpu = z.detach().cpu().to(dtype=torch.float32)
        w_cpu = w.detach().cpu().to(dtype=torch.float32)
        z_out[offset : offset + b] = z_cpu
        w_out[offset : offset + b] = w_cpu
        offset += b

    return CIFRunResult(z=z_out.numpy().copy(), w=w_out.numpy().copy())


def _summarize_cif_run(
    *,
    z: np.ndarray,
    w: np.ndarray,
    delta: float,
    W_max: float,
    trace_ns: list[int],
    F0s: list[float],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    n_max = int(z.shape[0])
    wz = (z.astype(np.float64) * w.astype(np.float64)).astype(np.float64, copy=False)

    ns = np.arange(1, n_max + 1, dtype=np.float64)
    cumsum = np.cumsum(wz, dtype=np.float64)
    r_hat = cumsum / ns

    delta_n = 6.0 * float(delta) / (math.pi**2 * ns * ns)
    radius = np.sqrt(np.log(2.0 / delta_n) / (2.0 * ns)) * float(W_max)

    lcb_f = 1.0 - (r_hat + radius)
    ucb_f = 1.0 - np.maximum(0.0, r_hat - radius)

    trace_rows_hoeff: list[dict] = []
    for n in trace_ns:
        idx = n - 1
        trace_rows_hoeff.append(
            {
                "cs_type": "hoeffding",
                "n": int(n),
                "r_hat": float(r_hat[idx]),
                "radius": float(radius[idx]),
                "lcb_f": float(lcb_f[idx]),
                "ucb_f": float(ucb_f[idx]),
            }
        )

    cert_rows_hoeff: list[dict] = []
    for F0 in F0s:
        ok = lcb_f >= float(F0)
        where = np.where(ok)[0]
        if where.size > 0:
            n_cert = int(where[0] + 1)
            certified = True
        else:
            n_cert = n_max
            certified = False
        cert_rows_hoeff.append(
            {"cs_type": "hoeffding", "F_0": float(F0), "n_certify": n_cert, "certified": certified}
        )

    # Betting CS: run on rescaled observations y_t = (w_t z_t)/W_max in [0,1], then scale back.
    y = np.clip(wz, 0.0, float(W_max)) / float(W_max)

    trace_rows_bet: list[dict] = []
    for n in trace_ns:
        tracker = BettingCSTracker(delta=delta, W_max=W_max)
        tracker.n = int(n)
        tracker._observations = y[:n].tolist()
        lo_R, hi_R = tracker.cs_interval
        trace_rows_bet.append(
            {
                "cs_type": "betting",
                "n": int(n),
                "r_hat": float(tracker.r_hat),
                "radius": float(0.5 * (hi_R - lo_R)),
                "lcb_f": float(1.0 - hi_R),
                "ucb_f": float(1.0 - lo_R),
            }
        )

    cert_rows_bet: list[dict] = []
    eta = 2.0 / (2.0 - math.log(3.0))
    c = 0.5
    log_thresh = math.log(1.0 / float(delta))

    def first_time_upper_below(target_R: float) -> tuple[int, bool]:
        """Return first n where betting U_R(n) <= target_R (certify F>=1-target_R)."""
        target_R = float(target_R)
        m = max(0.001, min(target_R / float(W_max), 0.999))

        lam = 0.0
        A = 1.0
        logK = 0.0
        sum_y = 0.0

        for t, x in enumerate(y.tolist(), 1):
            x = float(x)
            sum_y += x
            diff = x - m
            factor = 1.0 + lam * diff
            if factor <= 0.0:
                # Conservatively treat as rejection.
                logK = float("inf")
            else:
                logK += math.log(factor)

            y_mean = sum_y / float(t)
            if (y_mean <= m) and (logK >= log_thresh):
                return t, True

            if factor > 0.0:
                g = diff / factor
                A += g * g
                lam += eta * g / A
                if lam > c:
                    lam = c
                elif lam < -c:
                    lam = -c

        return n_max, False

    for F0 in F0s:
        target_R = 1.0 - float(F0)
        n_cert, ok = first_time_upper_below(target_R)
        cert_rows_bet.append({"cs_type": "betting", "F_0": float(F0), "n_certify": int(n_cert), "certified": bool(ok)})

    return trace_rows_hoeff, cert_rows_hoeff, trace_rows_bet, cert_rows_bet


def run_e1() -> None:
    ensure_dir(results_dir())
    device = get_device()

    # Step 0: Train/load MNIST MLP
    seed = 0
    set_seed(seed)
    art_dir = artifacts_dir()
    data_dir = Path(__file__).resolve().parent / "data"
    ckpt_path = art_dir / "mnist_mlp_seed0.pt"

    if not ckpt_path.exists():
        res = train_mnist(seed=seed, artifacts_dir=art_dir, data_dir=data_dir, device=device)
        print(f"[E1] Trained MNIST MLP: test_acc={res.test_acc:.4f} ({res.model_path})")
    model = load_model(ckpt_path, device=device)

    # Data loaders
    train_ds, test_ds = load_mnist_datasets(data_dir)
    test_loader = make_loader(test_ds, batch_size=1024, shuffle=False, device=device)

    # Step 1: Calibration activations + pruning scores
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

    # Step 2: Precompute test cache
    a_test_cpu, logits_base_cpu, _y_test = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            a = model.forward_features(xb)
            logits = model.fc_out(a)
            a_test_cpu.append(a.detach().cpu())
            logits_base_cpu.append(logits.detach().cpu())
            _y_test.append(yb.detach().cpu())
    a_test = torch.cat(a_test_cpu, dim=0).to(device)
    logits_base_cpu_t = torch.cat(logits_base_cpu, dim=0)
    y_test_cpu_t = torch.cat(_y_test, dim=0)
    test_acc = (logits_base_cpu_t.argmax(dim=-1) == y_test_cpu_t).to(dtype=torch.float32).mean().item()
    print(f"[E1] MNIST MLP test_acc={test_acc:.4f}")

    logits_base = logits_base_cpu_t.to(device)
    W_out = model.fc_out.weight.detach().to(device)
    b_out = model.fc_out.bias.detach().to(device)

    methods = ["cip-affine", "cip-const", "vbp", "random"]
    keep_sizes = [512, 384, 256, 128, 64, 32]
    mask_ps = [0.1, 0.2, 0.5]
    trace_ns = [50, 100, 200, 500, 1000, 2000, 5000, 10_000]
    F0s = [0.90, 0.95, 0.99]
    delta = 0.05

    # Precompute compiled heads per (method, keep) on CPU, then move to device.
    heads: dict[tuple[str, int], CompiledHead] = {}
    for method in methods:
        for keep in keep_sizes:
            kept_idx, pruned_idx, constants = make_prune_plan(method, keep, scores, seed=0)
            if method == "cip-affine":
                affine_map = compute_affine_map(
                    A=a_cal,
                    h=scores["h"],
                    pruned_idx=pruned_idx,
                    keep_idx=kept_idx,
                    r=8,
                    ridge=1e-4,
                )
                head_cpu = compile_affine(W_out_cpu, b_out_cpu, kept_idx, affine_map)
            else:
                head_cpu = compile_const(W_out_cpu, b_out_cpu, kept_idx, pruned_idx, constants)
            heads[(method, keep)] = CompiledHead(
                kept_idx=head_cpu.kept_idx.to(device),
                W_new=head_cpu.W_new.to(device),
                b_new=head_cpu.b_new.to(device),
            )

    trace_out: list[dict] = []
    cert_out: list[dict] = []

    for method in methods:
        for keep in keep_sizes:
            head = heads[(method, keep)]
            for mask_p in mask_ps:
                for sampling in ["iid", "adaptive"]:
                    seed_iia = 42
                    run = _run_iia_trials(
                        a_test=a_test,
                        logits_base=logits_base,
                        W_out=W_out,
                        head=head,
                        mask_p=mask_p,
                        sampling=sampling,
                        seed=seed_iia,
                        n_max=10_000,
                        batch_size=256,
                        alpha=0.3,
                    )
                    W_max = 1.0 if sampling == "iid" else 1.0 / (1.0 - 0.3)
                    tr_h, ce_h, tr_b, ce_b = _summarize_cif_run(
                        z=run.z, w=run.w, delta=delta, W_max=W_max, trace_ns=trace_ns, F0s=F0s
                    )
                    for r in tr_h + tr_b:
                        trace_out.append(
                            {
                                "method": method,
                                "keep": keep,
                                "mask_p": float(mask_p),
                                "sampling": sampling,
                                **r,
                            }
                        )
                    for r in ce_h + ce_b:
                        cert_out.append(
                            {
                                "method": method,
                                "keep": keep,
                                "mask_p": float(mask_p),
                                "sampling": sampling,
                                **r,
                            }
                        )

    df_trace = pd.DataFrame(trace_out)
    df_cert = pd.DataFrame(cert_out)
    df_trace.to_csv(results_dir() / "e1_cs_traces.csv", index=False)
    df_cert.to_csv(results_dir() / "e1_certification.csv", index=False)

    # Step 4: Paired comparison at keep=256, mask_p=0.5
    pairs = [
        ("cip-affine", "cip-const"),
        ("cip-const", "vbp"),
        ("vbp", "random"),
        ("cip-affine", "random"),
    ]
    pair_rows: list[dict] = []
    n_max = 10_000
    keep = 256
    mask_p = 0.5

    for m1, m2 in pairs:
        head1 = heads[(m1, keep)]
        head2 = heads[(m2, keep)]
        rng = torch.Generator(device=device).manual_seed(42)

        sum_z1 = 0.0
        sum_z2 = 0.0
        sum_delta = 0.0
        bet_tracker = PairedBettingCSTracker(delta=delta)
        offset = 0
        while offset < n_max:
            b = min(256, n_max - offset)
            i, j = _sample_i_j(a_test.shape[0], b, rng, device)
            mask = _sample_mask_iid(b, keep=head1.kept_idx.numel(), p=mask_p, rng=rng, device=device)
            z1 = _eval_z_batch(
                a_test=a_test, logits_base=logits_base, W_out=W_out, head=head1, i=i, j=j, mask=mask
            )
            z2 = _eval_z_batch(
                a_test=a_test, logits_base=logits_base, W_out=W_out, head=head2, i=i, j=j, mask=mask
            )
            delta_z = (z1 - z2).to(dtype=torch.float64)
            sum_z1 += float(z1.sum().item())
            sum_z2 += float(z2.sum().item())
            sum_delta += float(delta_z.sum().item())

            # Update paired betting tracker (vectorized)
            dy = (0.5 * (delta_z.clamp(-1.0, 1.0) + 1.0)).detach().cpu().numpy()
            bet_tracker.n += int(dy.shape[0])
            bet_tracker._observations.extend(dy.tolist())

            offset += b

        mu_hat = sum_delta / float(n_max)
        method_1_r_hat = float(sum_z1 / float(n_max))
        method_2_r_hat = float(sum_z2 / float(n_max))

        radius_paired = cs_radius_paired(n_max, delta)
        significant = (mu_hat - radius_paired) > 0.0 or (mu_hat + radius_paired) < 0.0
        pair_rows.append(
            {
                "cs_type": "hoeffding",
                "pair": f"{m1}_vs_{m2}",
                "keep": keep,
                "mask_p": float(mask_p),
                "n": n_max,
                "mu_hat": float(mu_hat),
                "radius": float(radius_paired),
                "significant": bool(significant),
                "method_1_r_hat": method_1_r_hat,
                "method_2_r_hat": method_2_r_hat,
            }
        )

        bet_lo, bet_hi = bet_tracker.cs_interval
        pair_rows.append(
            {
                "cs_type": "betting",
                "pair": f"{m1}_vs_{m2}",
                "keep": keep,
                "mask_p": float(mask_p),
                "n": n_max,
                "mu_hat": float(bet_tracker.mu_hat),
                "radius": float(0.5 * (bet_hi - bet_lo)),
                "significant": bool(bet_tracker.significant),
                "method_1_r_hat": method_1_r_hat,
                "method_2_r_hat": method_2_r_hat,
            }
        )

    pd.DataFrame(pair_rows).to_csv(results_dir() / "e1_paired.csv", index=False)


@torch.no_grad()
def run_e1_paired_only() -> None:
    """Regenerate only `results/e1_paired.csv` (paired comparisons), without rerunning full E1."""
    ensure_dir(results_dir())
    device = get_device()

    seed = 0
    set_seed(seed)
    art_dir = artifacts_dir()
    data_dir = Path(__file__).resolve().parent / "data"
    ckpt_path = art_dir / "mnist_mlp_seed0.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path} (refusing to retrain in paired-only mode)")
    model = load_model(ckpt_path, device=device)

    train_ds, test_ds = load_mnist_datasets(data_dir)

    # Calibration activations + pruning scores
    cal_subset = sample_calibration_subset(train_ds, n_cal=2000, seed=seed)
    cal_loader = make_loader(cal_subset, batch_size=512, shuffle=False, device=device)
    a_cal_cpu, y_cal_cpu = [], []
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

    # Test cache
    test_loader = make_loader(test_ds, batch_size=1024, shuffle=False, device=device)
    a_test_cpu, logits_base_cpu = [], []
    for xb, _yb in test_loader:
        xb = xb.to(device)
        a = model.forward_features(xb)
        logits = model.fc_out(a)
        a_test_cpu.append(a.detach().cpu())
        logits_base_cpu.append(logits.detach().cpu())
    a_test = torch.cat(a_test_cpu, dim=0).to(device)
    logits_base = torch.cat(logits_base_cpu, dim=0).to(device)
    W_out = model.fc_out.weight.detach().to(device)
    b_out = model.fc_out.bias.detach().to(device)

    delta = 0.05
    keep = 256
    mask_p = 0.5
    n_max = 10_000

    # Compile heads for the paired methods at keep=256.
    methods = ["cip-affine", "cip-const", "vbp", "random"]
    heads: dict[str, CompiledHead] = {}
    for method in methods:
        kept_idx, pruned_idx, constants = make_prune_plan(method, keep, scores, seed=0)
        if method == "cip-affine":
            affine_map = compute_affine_map(
                A=a_cal,
                h=scores["h"],
                pruned_idx=pruned_idx,
                keep_idx=kept_idx,
                r=8,
                ridge=1e-4,
            )
            head_cpu = compile_affine(W_out_cpu, b_out_cpu, kept_idx, affine_map)
        else:
            head_cpu = compile_const(W_out_cpu, b_out_cpu, kept_idx, pruned_idx, constants)
        heads[method] = CompiledHead(
            kept_idx=head_cpu.kept_idx.to(device),
            W_new=head_cpu.W_new.to(device),
            b_new=head_cpu.b_new.to(device),
        )

    pairs = [
        ("cip-affine", "cip-const"),
        ("cip-const", "vbp"),
        ("vbp", "random"),
        ("cip-affine", "random"),
    ]

    pair_rows: list[dict] = []
    for m1, m2 in pairs:
        head1 = heads[m1]
        head2 = heads[m2]
        rng = torch.Generator(device=device).manual_seed(42)

        sum_z1 = 0.0
        sum_z2 = 0.0
        sum_delta = 0.0
        bet_tracker = PairedBettingCSTracker(delta=delta)
        offset = 0
        while offset < n_max:
            b = min(256, n_max - offset)
            i, j = _sample_i_j(a_test.shape[0], b, rng, device)
            mask = _sample_mask_iid(b, keep=head1.kept_idx.numel(), p=mask_p, rng=rng, device=device)
            z1 = _eval_z_batch(a_test=a_test, logits_base=logits_base, W_out=W_out, head=head1, i=i, j=j, mask=mask)
            z2 = _eval_z_batch(a_test=a_test, logits_base=logits_base, W_out=W_out, head=head2, i=i, j=j, mask=mask)
            delta_z = (z1 - z2).to(dtype=torch.float64)
            sum_z1 += float(z1.sum().item())
            sum_z2 += float(z2.sum().item())
            sum_delta += float(delta_z.sum().item())

            dy = (0.5 * (delta_z.clamp(-1.0, 1.0) + 1.0)).detach().cpu().numpy()
            bet_tracker.n += int(dy.shape[0])
            bet_tracker._observations.extend(dy.tolist())

            offset += b

        mu_hat = sum_delta / float(n_max)
        method_1_r_hat = float(sum_z1 / float(n_max))
        method_2_r_hat = float(sum_z2 / float(n_max))

        radius_paired = cs_radius_paired(n_max, delta)
        significant = (mu_hat - radius_paired) > 0.0 or (mu_hat + radius_paired) < 0.0
        pair_rows.append(
            {
                "cs_type": "hoeffding",
                "pair": f"{m1}_vs_{m2}",
                "keep": keep,
                "mask_p": float(mask_p),
                "n": n_max,
                "mu_hat": float(mu_hat),
                "radius": float(radius_paired),
                "significant": bool(significant),
                "method_1_r_hat": method_1_r_hat,
                "method_2_r_hat": method_2_r_hat,
            }
        )

        bet_lo, bet_hi = bet_tracker.cs_interval
        pair_rows.append(
            {
                "cs_type": "betting",
                "pair": f"{m1}_vs_{m2}",
                "keep": keep,
                "mask_p": float(mask_p),
                "n": n_max,
                "mu_hat": float(bet_tracker.mu_hat),
                "radius": float(0.5 * (bet_hi - bet_lo)),
                "significant": bool(bet_tracker.significant),
                "method_1_r_hat": method_1_r_hat,
                "method_2_r_hat": method_2_r_hat,
            }
        )

    pd.DataFrame(pair_rows).to_csv(results_dir() / "e1_paired.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="E1: Certified IIA experiments")
    parser.add_argument("--paired-only", action="store_true", help="Only regenerate results/e1_paired.csv")
    args = parser.parse_args()

    if args.paired_only:
        run_e1_paired_only()
    else:
        run_e1()
