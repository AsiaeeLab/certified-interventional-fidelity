from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompiledHead:
    kept_idx: torch.Tensor  # (keep,) int64
    W_new: torch.Tensor  # (C, keep)
    b_new: torch.Tensor  # (C,)


def compile_const(
    W_out: torch.Tensor,
    b_out: torch.Tensor,
    kept_idx: torch.Tensor,
    pruned_idx: torch.Tensor,
    constants: torch.Tensor,
) -> CompiledHead:
    """Compile constant-replacement abstraction via bias folding."""
    b_new = b_out + W_out[:, pruned_idx] @ constants[pruned_idx]
    W_new = W_out[:, kept_idx]
    return CompiledHead(kept_idx=kept_idx, W_new=W_new, b_new=b_new)


def compile_affine(
    W_out: torch.Tensor,
    b_out: torch.Tensor,
    kept_idx: torch.Tensor,
    affine_map: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> CompiledHead:
    """Compile affine-replacement abstraction. Fold affine maps into classifier weights."""

    kept_idx = kept_idx.to(dtype=torch.long)
    kept_idx, _ = torch.sort(kept_idx)

    W_orig = W_out.clone()
    W_new_full = W_out.clone()
    b_new = b_out.clone()

    for j, (beta, parent_idx, coef) in affine_map.items():
        Wj = W_orig[:, int(j)]  # (C,)
        parent_idx = parent_idx.to(dtype=torch.long)
        coef = coef.to(dtype=W_out.dtype)
        W_new_full[:, parent_idx] += Wj.view(-1, 1) * coef.view(1, -1)
        b_new += beta.to(dtype=W_out.dtype) * Wj

    W_new = W_new_full[:, kept_idx]
    return CompiledHead(kept_idx=kept_idx, W_new=W_new, b_new=b_new)

