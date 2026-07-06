from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
TABLE_DIR = PROJECT_ROOT / "results"
FIG_DIR = PROJECT_ROOT / "figures"

from cif import BettingCSTracker  # noqa: E402
from compile import CompiledHead, compile_affine, compile_const  # noqa: E402
from e1_certified_iia import _run_iia_trials  # noqa: E402
from mnist_utils import load_mnist_datasets, make_loader, sample_calibration_subset  # noqa: E402
from models import load_model  # noqa: E402
from pruning import compute_affine_map, compute_pruning_scores, make_prune_plan  # noqa: E402
from utils import artifacts_dir, get_device, set_seed, set_torch_num_threads_from_env  # noqa: E402


DELTA = 0.05
ALPHA_DEFAULT = 0.3


@dataclass
class E1Context:
    a_test: torch.Tensor
    logits_base: torch.Tensor
    w_out: torch.Tensor
    scores: dict[str, torch.Tensor]
    a_cal: torch.Tensor
    w_out_cpu: torch.Tensor
    b_out_cpu: torch.Tensor
    device: torch.device


def _ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_e1_context() -> E1Context:
    set_torch_num_threads_from_env(default=8)
    device = get_device()
    set_seed(0)

    ckpt_path = artifacts_dir() / "mnist_mlp_seed0.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing cached MNIST MLP checkpoint: {ckpt_path}")

    model = load_model(ckpt_path, device=device)
    data_dir = CODE_DIR / "data"
    train_ds, test_ds = load_mnist_datasets(data_dir)

    cal_subset = sample_calibration_subset(train_ds, n_cal=2000, seed=0)
    cal_loader = make_loader(cal_subset, batch_size=512, shuffle=False, device=device)
    a_cal_parts: list[torch.Tensor] = []
    y_cal_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, yb in cal_loader:
            xb = xb.to(device)
            a_cal_parts.append(model.forward_features(xb).detach().cpu())
            y_cal_parts.append(yb.detach().cpu())
    a_cal = torch.cat(a_cal_parts, dim=0)
    y_cal = torch.cat(y_cal_parts, dim=0)

    w_out_cpu = model.fc_out.weight.detach().cpu()
    b_out_cpu = model.fc_out.bias.detach().cpu()
    scores = compute_pruning_scores(a_cal, y_cal, w_out_cpu, b_out_cpu)

    test_loader = make_loader(test_ds, batch_size=1024, shuffle=False, device=device)
    a_test_parts: list[torch.Tensor] = []
    logits_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, _yb in test_loader:
            xb = xb.to(device)
            a = model.forward_features(xb)
            a_test_parts.append(a.detach().cpu())
            logits_parts.append(model.fc_out(a).detach().cpu())

    return E1Context(
        a_test=torch.cat(a_test_parts, dim=0).to(device),
        logits_base=torch.cat(logits_parts, dim=0).to(device),
        w_out=model.fc_out.weight.detach().to(device),
        scores=scores,
        a_cal=a_cal,
        w_out_cpu=w_out_cpu,
        b_out_cpu=b_out_cpu,
        device=device,
    )


def _compile_head(ctx: E1Context, method: str, keep: int) -> CompiledHead:
    kept_idx, pruned_idx, constants = make_prune_plan(method, keep, ctx.scores, seed=0)
    if method == "cip-affine":
        affine_map = compute_affine_map(
            A=ctx.a_cal,
            h=ctx.scores["h"],
            pruned_idx=pruned_idx,
            keep_idx=kept_idx,
            r=8,
            ridge=1e-4,
        )
        head_cpu = compile_affine(ctx.w_out_cpu, ctx.b_out_cpu, kept_idx, affine_map)
    else:
        head_cpu = compile_const(ctx.w_out_cpu, ctx.b_out_cpu, kept_idx, pruned_idx, constants)
    return CompiledHead(
        kept_idx=head_cpu.kept_idx.to(ctx.device),
        W_new=head_cpu.W_new.to(ctx.device),
        b_new=head_cpu.b_new.to(ctx.device),
    )


def _betting_interval(z: np.ndarray, w: np.ndarray, W_max: float, n: int | None = None) -> tuple[float, float, float]:
    tracker = BettingCSTracker(delta=DELTA, W_max=W_max)
    n_use = int(len(z) if n is None else n)
    for zi, wi in zip(z[:n_use], w[:n_use], strict=False):
        tracker.update(float(zi), float(wi))
    lo_r, hi_r = tracker.cs_interval
    return float(lo_r), float(hi_r), float(tracker.r_hat)


def _first_time_betting_one_sided(
    z: np.ndarray,
    w: np.ndarray,
    *,
    W_max: float,
    target_R: float,
    direction: str,
) -> tuple[int, bool]:
    """One-sided betting rejection at mean target_R.

    direction='below' certifies R <= target_R, equivalently F >= 1-target_R.
    direction='above' certifies R >= target_R, equivalently F <= 1-target_R.
    """

    if direction not in {"below", "above"}:
        raise ValueError("direction must be 'below' or 'above'")

    y = np.clip(z.astype(np.float64) * w.astype(np.float64), 0.0, float(W_max)) / float(W_max)
    m = max(0.001, min(float(target_R) / float(W_max), 0.999))
    eta = 2.0 / (2.0 - math.log(3.0))
    c = 0.5
    log_thresh = math.log(1.0 / DELTA)

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
            logK = float("inf")
        else:
            logK += math.log(factor)

        y_mean = sum_y / float(t)
        if direction == "below" and y_mean <= m and logK >= log_thresh:
            return t, True
        if direction == "above" and y_mean >= m and logK >= log_thresh:
            return t, True

        if factor > 0.0:
            g = diff / factor
            A += g * g
            lam += eta * g / A
            lam = min(c, max(-c, lam))

    return len(z), False


def _run_config(
    ctx: E1Context,
    head: CompiledHead,
    *,
    method: str,
    keep: int,
    mask_p: float,
    sampling: str,
    alpha: float,
    seed: int,
    n_max: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    run = _run_iia_trials(
        a_test=ctx.a_test,
        logits_base=ctx.logits_base,
        W_out=ctx.w_out,
        head=head,
        mask_p=mask_p,
        sampling=sampling,
        seed=seed,
        n_max=n_max,
        batch_size=512,
        alpha=alpha,
    )
    W_max = 1.0 if sampling == "iid" else 1.0 / (1.0 - float(alpha))
    print(
        f"[run] method={method} keep={keep} p={mask_p} sampling={sampling} "
        f"alpha={alpha} seed={seed} n={n_max} F_hat={1.0 - np.mean(run.z * run.w):.4f}",
        flush=True,
    )
    return run.z, run.w, W_max


def run_adaptive_vs_iid(ctx: E1Context) -> pd.DataFrame:
    rows: list[dict] = []
    trace_rows: list[dict] = []

    head_cip = _compile_head(ctx, "cip-affine", 256)
    trace_ns = sorted(set(np.rint(np.geomspace(20, 5000, 45)).astype(int).tolist() + [1, 5000]))
    for sampling, alpha in [("iid", 0.0), ("adaptive", ALPHA_DEFAULT)]:
        z, w, W_max = _run_config(
            ctx,
            head_cip,
            method="cip-affine",
            keep=256,
            mask_p=0.5,
            sampling=sampling,
            alpha=alpha if sampling == "adaptive" else ALPHA_DEFAULT,
            seed=42,
            n_max=5000,
        )
        target_R = 1.0 - 0.70
        n_cert, certified = _first_time_betting_one_sided(
            z, w, W_max=W_max, target_R=target_R, direction="above"
        )
        lo_R, hi_R, r_hat = _betting_interval(z, w, W_max)
        rows.append(
            {
                "experiment": "lowF_exclusion_cip_affine_k256_p0.5_F0_0.70",
                "sampling": sampling,
                "alpha": float(alpha),
                "seed": 42,
                "n_certify": int(n_cert),
                "F_hat_final": float(1.0 - r_hat),
                "radius_final": float(0.5 * (hi_R - lo_R)),
                "certified": bool(certified),
                "n_max": 5000,
            }
        )

        for n in trace_ns:
            lo_n, hi_n, r_hat_n = _betting_interval(z, w, W_max, n=n)
            trace_rows.append(
                {
                    "experiment": "lowF_exclusion_cip_affine_k256_p0.5_F0_0.70",
                    "sampling": sampling,
                    "alpha": float(alpha),
                    "seed": 42,
                    "n": int(n),
                    "F_hat": float(1.0 - r_hat_n),
                    "lcb_f": float(1.0 - hi_n),
                    "ucb_f": float(1.0 - lo_n),
                    "radius": float(0.5 * (hi_n - lo_n)),
                    "F_0": 0.70,
                }
            )

    head_vbp = _compile_head(ctx, "vbp", 128)
    for seed in [101, 102, 103, 104, 105]:
        for sampling, alpha in [("iid", 0.0), ("adaptive", ALPHA_DEFAULT)]:
            z, w, W_max = _run_config(
                ctx,
                head_vbp,
                method="vbp",
                keep=128,
                mask_p=0.1,
                sampling=sampling,
                alpha=alpha if sampling == "adaptive" else ALPHA_DEFAULT,
                seed=seed,
                n_max=10000,
            )
            n_cert, certified = _first_time_betting_one_sided(
                z, w, W_max=W_max, target_R=1.0 - 0.90, direction="below"
            )
            lo_R, hi_R, r_hat = _betting_interval(z, w, W_max)
            rows.append(
                {
                    "experiment": "borderline_cert_vbp_k128_p0.1_F0_0.90",
                    "sampling": sampling,
                    "alpha": float(alpha),
                    "seed": int(seed),
                    "n_certify": int(n_cert),
                    "F_hat_final": float(1.0 - r_hat),
                    "radius_final": float(0.5 * (hi_R - lo_R)),
                    "certified": bool(certified),
                    "n_max": 10000,
                }
            )

    df = pd.DataFrame(rows)
    trace = pd.DataFrame(trace_rows)
    df.to_csv(TABLE_DIR / "adaptive_vs_iid.csv", index=False)
    trace.to_csv(TABLE_DIR / "adaptive_vs_iid_trace.csv", index=False)
    _plot_adaptive_vs_iid(df, trace)
    return df


def _plot_adaptive_vs_iid(df: pd.DataFrame, trace: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
        }
    )
    colors = {"iid": "#355C7D", "adaptive": "#C06C84"}
    labels = {"iid": "i.i.d.", "adaptive": "adaptive (alpha=0.3)"}

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), constrained_layout=True)

    ax = axes[0]
    for sampling in ["iid", "adaptive"]:
        s = trace[trace["sampling"] == sampling].sort_values("n")
        ax.plot(s["n"], s["ucb_f"], marker="o", markersize=2.5, linewidth=1.6, color=colors[sampling], label=labels[sampling])
    ax.axhline(0.70, color="0.25", linestyle="--", linewidth=1.0, label="exclusion target F0=0.70")
    ax.set_xscale("log")
    ax.set_xlabel("samples n")
    ax.set_ylabel("UCB(F)")
    ax.set_title("Low-fidelity exclusion")
    ax.grid(True)
    ax.legend(frameon=True, fontsize=8)

    ax = axes[1]
    sub = df[df["experiment"] == "borderline_cert_vbp_k128_p0.1_F0_0.90"].copy()
    positions = {"iid": 0, "adaptive": 1}
    for sampling in ["iid", "adaptive"]:
        vals = sub[sub["sampling"] == sampling]["n_certify"].astype(float).to_numpy()
        ax.boxplot(
            vals,
            positions=[positions[sampling]],
            widths=0.45,
            patch_artist=True,
            boxprops={"facecolor": colors[sampling], "alpha": 0.28, "edgecolor": colors[sampling]},
            medianprops={"color": colors[sampling], "linewidth": 1.8},
            whiskerprops={"color": colors[sampling]},
            capprops={"color": colors[sampling]},
            flierprops={"marker": "o", "markersize": 3, "markerfacecolor": colors[sampling], "markeredgecolor": colors[sampling]},
        )
        jitter = np.linspace(-0.07, 0.07, num=len(vals)) if len(vals) else np.array([])
        ax.scatter(np.full_like(vals, positions[sampling], dtype=float) + jitter, vals, s=18, color=colors[sampling], zorder=3)
    ax.set_xticks([0, 1], ["i.i.d.", "adaptive"])
    ax.set_ylabel("n to certify F >= 0.90")
    ax.set_title("Borderline certification")
    ax.grid(True, axis="y")

    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"adaptive_vs_iid_lowF.{ext}", dpi=300)
    plt.close(fig)


def run_alpha_sweep(ctx: E1Context) -> pd.DataFrame:
    rows: list[dict] = []
    head = _compile_head(ctx, "vbp", 256)
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]:
        sampling = "iid" if alpha == 0.0 else "adaptive"
        W_max_expected = 1.0 if alpha == 0.0 else 1.0 / (1.0 - alpha)
        for seed in [201, 202, 203]:
            z, w, W_max = _run_config(
                ctx,
                head,
                method="vbp",
                keep=256,
                mask_p=0.1,
                sampling=sampling,
                alpha=alpha if sampling == "adaptive" else ALPHA_DEFAULT,
                seed=seed,
                n_max=10000,
            )
            if not np.isclose(W_max, W_max_expected):
                raise RuntimeError(f"Unexpected W_max for alpha={alpha}: got {W_max}, expected {W_max_expected}")
            _lo_R, _hi_R, r_hat = _betting_interval(z, w, W_max)
            for F0 in [0.90, 0.95]:
                n_cert, certified = _first_time_betting_one_sided(
                    z, w, W_max=W_max, target_R=1.0 - F0, direction="below"
                )
                rows.append(
                    {
                        "alpha": float(alpha),
                        "seed": int(seed),
                        "F_0": float(F0),
                        "n_certify": int(n_cert),
                        "F_hat_final": float(1.0 - r_hat),
                        "W_max": float(W_max),
                        "certified": bool(certified),
                        "n_max": 10000,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "alpha_sweep.csv", index=False)
    _plot_alpha_sweep(df)
    return df


def _plot_alpha_sweep(df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
        }
    )
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    colors = {0.90: "#355C7D", 0.95: "#C06C84"}
    xvals = sorted(df["alpha"].unique())
    for F0 in [0.90, 0.95]:
        medians = []
        lo_err = []
        hi_err = []
        for alpha in xvals:
            vals = df[(df["F_0"] == F0) & (df["alpha"] == alpha)]["n_certify"].astype(float).to_numpy()
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            medians.append(med)
            lo_err.append(med - q1)
            hi_err.append(q3 - med)
        ax.errorbar(
            xvals,
            medians,
            yerr=[lo_err, hi_err],
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=colors[F0],
            label=f"F0={F0:.2f}",
        )
    ax.set_xlabel("mixture rate alpha (0 = i.i.d.)")
    ax.set_ylabel("n to certify")
    ax.set_title("Alpha sweep, vbp k=256 p=0.1")
    ax.set_xticks(xvals)
    ax.grid(True)
    ax.legend(frameon=True)
    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"alpha_sweep.{ext}", dpi=300)
    plt.close(fig)



def main() -> None:
    start_time = time.time()
    _ensure_dirs()
    print(f"[setup] project={PROJECT_ROOT}", flush=True)
    ctx = _load_e1_context()
    print(f"[setup] device={ctx.device} n_test={ctx.a_test.shape[0]}", flush=True)
    adaptive_df = run_adaptive_vs_iid(ctx)
    alpha_df = run_alpha_sweep(ctx)
    print(f"[done] runtime={(time.time() - start_time) / 60.0:.1f} min")
    print(f"[done] wrote {TABLE_DIR} and {FIG_DIR}", flush=True)


if __name__ == "__main__":
    main()
