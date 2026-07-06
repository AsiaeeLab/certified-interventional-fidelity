from __future__ import annotations

from collections.abc import Iterable

import torch


@torch.no_grad()
def compute_pruning_scores(
    a_cal: torch.Tensor,
    y_cal: torch.Tensor,
    W_out: torch.Tensor,
    b_out: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute pruning scores and constants for CIF MNIST abstractions.

    Returns dict with keys:
      - scores_cip_const: (d,) CIP-Const quadratic proxy score (Qmin / n_cal)
      - scores_vbp: (d,) variance score Var(a_j)
      - scores_random: (d,) random scores (seed=0)
      - c_star: (d,) optimal curvature-weighted constant
      - mean_a: (d,)
      - var_a: (d,)
      - g: (n_cal, d) gradient proxy
      - h: (n_cal, d) curvature proxy (Hessian diagonal)
    """

    device = a_cal.device
    n_cal, d = a_cal.shape
    C = W_out.shape[0]

    logits = a_cal @ W_out.T + b_out  # (n, C)
    p = torch.softmax(logits, dim=-1)
    y_onehot = torch.zeros((n_cal, C), device=device, dtype=a_cal.dtype)
    y_onehot.scatter_(1, y_cal.view(-1, 1), 1.0)

    g = (p - y_onehot) @ W_out  # (n, d)
    u = p @ W_out  # (n, d)
    v = p @ (W_out * W_out)  # (n, d)
    h = v - u * u  # (n, d)

    a = a_cal
    mean_a = a.mean(dim=0)
    var_a = a.var(dim=0, unbiased=False)

    H = h.sum(dim=0)
    G = g.sum(dim=0)
    Ha = (h * a).sum(dim=0)
    Ga = (g * a).sum(dim=0)
    Haa = (h * a * a).sum(dim=0)

    eps = 1e-10
    good = H > eps

    c_star = torch.empty((d,), device=device, dtype=a.dtype)
    c_star[good] = (Ha[good] - G[good]) / H[good]
    c_star[~good] = mean_a[~good]

    # Minimized quadratic proxy value Qmin (sum over samples, not averaged)
    Qmin = torch.empty((d,), device=device, dtype=a.dtype)
    Qmin[good] = (0.5 * Haa[good] - Ga[good]) - (G[good] - Ha[good]).pow(2) / (2.0 * H[good])
    # Fallback: evaluate proxy at c = mean when H is tiny.
    c_fb = mean_a[~good]
    Q_fb = 0.5 * H[~good] * c_fb * c_fb + c_fb * (G[~good] - Ha[~good]) + (0.5 * Haa[~good] - Ga[~good])
    Qmin[~good] = Q_fb

    scores_cip_const = Qmin / float(n_cal)

    g0 = torch.Generator(device=device).manual_seed(0)
    scores_random = torch.rand((d,), generator=g0, device=device, dtype=a.dtype)

    return {
        "scores_cip_const": scores_cip_const.detach(),
        "scores_vbp": var_a.detach(),
        "scores_random": scores_random.detach(),
        "c_star": c_star.detach(),
        "mean_a": mean_a.detach(),
        "var_a": var_a.detach(),
        "g": g.detach(),
        "h": h.detach(),
    }


def make_prune_plan(
    method: str,
    keep: int,
    scores_dict: dict[str, torch.Tensor],
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (kept_idx, pruned_idx, constants) tensors."""

    scores = scores_dict["scores_cip_const"]
    d = scores.numel()
    if keep > d or keep <= 0:
        raise ValueError(f"Invalid keep={keep} for d={d}")
    B = d - keep

    if method in ("cip-const", "cip-affine"):
        ranking = torch.argsort(scores_dict["scores_cip_const"], descending=False)
        pruned = ranking[:B]
        constants = scores_dict["c_star"]
    elif method == "vbp":
        ranking = torch.argsort(scores_dict["var_a"], descending=False)
        pruned = ranking[:B]
        constants = scores_dict["mean_a"]
    elif method == "random":
        g = torch.Generator(device=scores.device).manual_seed(seed)
        pruned = torch.randperm(d, generator=g, device=scores.device)[:B]
        constants = scores_dict["mean_a"]
    else:
        raise ValueError(f"Unknown method: {method}")

    kept_mask = torch.ones(d, dtype=torch.bool, device=scores.device)
    kept_mask[pruned] = False
    kept = torch.arange(d, device=scores.device)[kept_mask]
    kept, _ = torch.sort(kept)
    pruned, _ = torch.sort(pruned)
    return kept.to(dtype=torch.long), pruned.to(dtype=torch.long), constants


def _top_correlated(A: torch.Tensor, j: int, keep_idx: torch.Tensor, r: int) -> torch.Tensor:
    a_j = A[:, j]
    A_keep = A[:, keep_idx]
    a_jc = a_j - a_j.mean()
    A_kc = A_keep - A_keep.mean(dim=0, keepdim=True)
    cov = (A_kc * a_jc.view(-1, 1)).mean(dim=0)
    std_j = a_jc.std(unbiased=False) + 1e-8
    std_k = A_kc.std(dim=0, unbiased=False) + 1e-8
    corr = cov / (std_k * std_j)
    topk = torch.topk(corr.abs(), k=min(r, corr.numel())).indices
    return keep_idx[topk]


def _weighted_affine_fit(
    X: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    ridge: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Weighted least squares with centered features
    w = w.clamp_min(1e-8)
    w_sum = w.sum()
    x_mean = (w.view(-1, 1) * X).sum(dim=0) / w_sum
    y_mean = (w * y).sum() / w_sum
    Xc = X - x_mean
    yc = y - y_mean
    sqrt_w = torch.sqrt(w)
    Xw = Xc * sqrt_w.view(-1, 1)
    yw = yc * sqrt_w
    A_mat = Xw.t() @ Xw + ridge * torch.eye(Xw.shape[1], device=X.device, dtype=X.dtype)
    b_vec = Xw.t() @ yw
    try:
        coef = torch.linalg.solve(A_mat, b_vec)
    except RuntimeError:
        coef = torch.linalg.lstsq(A_mat, b_vec).solution
    beta = y_mean - (x_mean @ coef)
    return beta, coef


def compute_affine_map(
    A: torch.Tensor,
    h: torch.Tensor,
    pruned_idx: Iterable[int] | torch.Tensor,
    keep_idx: Iterable[int] | torch.Tensor,
    r: int = 8,
    ridge: float = 1e-4,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """For CIP-Affine: map pruned unit -> (beta, parent_idx, coef)."""

    if isinstance(keep_idx, torch.Tensor):
        keep_idx_t = keep_idx.to(device=A.device, dtype=torch.long)
    else:
        keep_idx_t = torch.tensor(list(keep_idx), device=A.device, dtype=torch.long)

    if isinstance(pruned_idx, torch.Tensor):
        pruned_list = pruned_idx.to(device="cpu", dtype=torch.long).tolist()
    else:
        pruned_list = list(pruned_idx)

    affine_map: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for j in pruned_list:
        pj = _top_correlated(A, int(j), keep_idx_t, r)
        X = A[:, pj]
        y = A[:, int(j)]
        w = h[:, int(j)]
        beta, coef = _weighted_affine_fit(X, y, w, ridge=ridge)
        affine_map[int(j)] = (beta, pj, coef)
    return affine_map

