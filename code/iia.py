from __future__ import annotations

import torch

from compile import CompiledHead


@torch.no_grad()
def sample_iia_single(
    a_test: torch.Tensor,  # (N, d)
    logits_base: torch.Tensor,  # (N, C)
    W_out: torch.Tensor,  # (C, d)
    head: CompiledHead,
    bernoulli_p: float,
    rng: torch.Generator,
) -> float:
    """Sample ONE interchange intervention trial, return Z_t in {0, 1}.

    Z_t = 0 if argmax(logits_dense) == argmax(logits_red), else 1.
    So R = E[Z] = 1 - IIA.
    """
    N = a_test.shape[0]
    kept = head.kept_idx

    i = torch.randint(0, N, (1,), generator=rng, device=a_test.device).item()
    j = torch.randint(0, N, (1,), generator=rng, device=a_test.device).item()
    while j == i:
        j = torch.randint(0, N, (1,), generator=rng, device=a_test.device).item()

    a_i_kept = a_test[i, kept]  # (keep,)
    a_j_kept = a_test[j, kept]  # (keep,)

    mask = torch.rand(kept.numel(), generator=rng, device=a_test.device) < float(bernoulli_p)
    a_h = torch.where(mask, a_j_kept, a_i_kept)

    W_kept = W_out[:, kept]
    logits_dense = logits_base[i] + (a_h - a_i_kept) @ W_kept.T
    logits_red = a_h @ head.W_new.T + head.b_new

    agree = (logits_dense.argmax() == logits_red.argmax()).item()
    return 0.0 if agree else 1.0


@torch.no_grad()
def sample_iia_batch(
    a_test: torch.Tensor,
    logits_base: torch.Tensor,
    W_out: torch.Tensor,
    head: CompiledHead,
    bernoulli_p: float,
    rng: torch.Generator,
    batch_size: int = 256,
) -> torch.Tensor:
    """Sample batch_size IIA trials, return Z_t tensor of shape (batch_size,)."""

    device = a_test.device
    N = a_test.shape[0]
    kept = head.kept_idx
    keep = kept.numel()
    C = W_out.shape[0]
    if logits_base.shape[1] != C:
        raise ValueError(f"logits_base has C={logits_base.shape[1]}, but W_out has C={C}")

    i = torch.randint(0, N, (batch_size,), generator=rng, device=device)
    j = torch.randint(0, N, (batch_size,), generator=rng, device=device)
    neq = j != i
    while not bool(neq.all()):
        j2 = torch.randint(0, N, (batch_size,), generator=rng, device=device)
        j = torch.where(neq, j, j2)
        neq = j != i

    a_i_kept = a_test[i][:, kept]  # (B, keep)
    a_j_kept = a_test[j][:, kept]  # (B, keep)

    mask = torch.rand((batch_size, keep), generator=rng, device=device) < float(bernoulli_p)
    mask_f = mask.to(dtype=a_test.dtype)

    a_h = torch.where(mask, a_j_kept, a_i_kept)  # (B, keep)
    delta = (a_j_kept - a_i_kept) * mask_f

    W_kept = W_out[:, kept]  # (C, keep)
    logits_dense = logits_base[i] + delta @ W_kept.T  # (B, C)
    logits_red = a_h @ head.W_new.T + head.b_new  # (B, C)

    z = (logits_dense.argmax(dim=-1) != logits_red.argmax(dim=-1)).to(dtype=torch.float32)
    return z

