from __future__ import annotations

import sys
import time
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

from compile import CompiledHead  # noqa: E402
from e1_certified_iia import (  # noqa: E402
    _eval_z_batch,
    _sample_i_j,
    _sample_mask_adaptive_and_weight,
    _stress_probs_from_sensitivity,
)
from adaptive_study import (  # noqa: E402
    ALPHA_DEFAULT,
    DELTA,
    E1Context,
    _betting_interval,
    _compile_head,
    _ensure_dirs,
    _first_time_betting_one_sided,
    _load_e1_context,
)


def _run_history_adaptive_trials(
    ctx: E1Context,
    head: CompiledHead,
    *,
    mask_p: float,
    alpha: float,
    seed: int,
    n_max: int,
    batch_size: int = 100,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run bounded-mixture sampling with cumulative per-coordinate failure rates."""

    device = ctx.a_test.device
    N = ctx.a_test.shape[0]
    keep = int(head.kept_idx.numel())
    rng = torch.Generator(device=device).manual_seed(seed)

    success_counts = torch.zeros((keep,), dtype=torch.float32, device=device)
    exposure_counts = torch.zeros((keep,), dtype=torch.float32, device=device)
    p_k = torch.full((keep,), float(mask_p), dtype=torch.float32, device=device)

    z_out = torch.empty((n_max,), dtype=torch.float32, device="cpu")
    w_out = torch.empty((n_max,), dtype=torch.float32, device="cpu")

    offset = 0
    while offset < n_max:
        b = min(batch_size, n_max - offset)
        i, j = _sample_i_j(N, b, rng, device)
        mask, w = _sample_mask_adaptive_and_weight(
            batch_size=b,
            p=mask_p,
            p_k=p_k,
            alpha=alpha,
            rng=rng,
            device=device,
        )
        z = _eval_z_batch(
            a_test=ctx.a_test,
            logits_base=ctx.logits_base,
            W_out=ctx.w_out,
            head=head,
            i=i,
            j=j,
            mask=mask,
        )

        mask_f = mask.to(dtype=torch.float32)
        exposure_counts += mask_f.sum(dim=0)
        success_counts += (mask_f * z.view(-1, 1)).sum(dim=0)
        failure_rate = (success_counts + prior_alpha) / (exposure_counts + prior_alpha + prior_beta)
        p_k = _stress_probs_from_sensitivity(failure_rate, target_sum=float(mask_p) * float(keep))
        p_k = p_k.clamp(min=0.0, max=1.0).to(device=device, dtype=torch.float32)

        z_out[offset : offset + b] = z.detach().cpu().to(dtype=torch.float32)
        w_out[offset : offset + b] = w.detach().cpu().to(dtype=torch.float32)
        offset += b

    W_max = 1.0 / (1.0 - float(alpha))
    print(
        f"[history] keep={keep} p={mask_p} alpha={alpha} seed={seed} "
        f"n={n_max} F_hat={1.0 - np.mean(z_out.numpy() * w_out.numpy()):.4f}",
        flush=True,
    )
    return z_out.numpy().copy(), w_out.numpy().copy(), W_max


def _summarize_run(
    *,
    z: np.ndarray,
    w: np.ndarray,
    W_max: float,
    experiment: str,
    seed: int,
    n_max: int,
    target_F: float,
    direction: str,
) -> dict:
    n_cert, certified = _first_time_betting_one_sided(
        z,
        w,
        W_max=W_max,
        target_R=1.0 - float(target_F),
        direction=direction,
    )
    lo_R, hi_R, r_hat = _betting_interval(z, w, W_max)
    _lo_stop, _hi_stop, r_hat_stop = _betting_interval(z, w, W_max, n=n_cert)
    return {
        "experiment": experiment,
        "sampling": "adaptive",
        "proposal": "history",
        "alpha": ALPHA_DEFAULT,
        "seed": int(seed),
        "n_certify": int(n_cert),
        "F_hat_final": float(1.0 - r_hat),
        "radius_final": float(0.5 * (hi_R - lo_R)),
        "F_hat_stop": float(1.0 - r_hat_stop),
        "certified": bool(certified),
        "n_max": int(n_max),
    }


def _trace_run(
    *,
    z: np.ndarray,
    w: np.ndarray,
    W_max: float,
    experiment: str,
    seed: int,
    proposal: str,
    alpha: float,
    trace_ns: list[int],
) -> list[dict]:
    out: list[dict] = []
    for n in trace_ns:
        lo_R, hi_R, r_hat = _betting_interval(z, w, W_max, n=n)
        out.append(
            {
                "experiment": experiment,
                "sampling": "iid" if proposal == "n/a" else "adaptive",
                "proposal": proposal,
                "alpha": float(alpha),
                "seed": int(seed),
                "n": int(n),
                "F_hat": float(1.0 - r_hat),
                "lcb_f": float(1.0 - hi_R),
                "ucb_f": float(1.0 - lo_R),
                "radius": float(0.5 * (hi_R - lo_R)),
            }
        )
    return out


def _baseline_rows() -> pd.DataFrame:
    path = TABLE_DIR / "adaptive_vs_iid.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing v1 baseline table: {path}")
    df = pd.read_csv(path)
    df.insert(
        2,
        "proposal",
        np.where(df["sampling"].eq("adaptive"), "static", "n/a"),
    )
    if "F_hat_stop" not in df.columns:
        df["F_hat_stop"] = np.nan
    return df


def _baseline_trace_rows() -> pd.DataFrame:
    path = TABLE_DIR / "adaptive_vs_iid_trace.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing v1 baseline trace table: {path}")
    trace = pd.read_csv(path)
    trace.insert(
        2,
        "proposal",
        np.where(trace["sampling"].eq("adaptive"), "static", "n/a"),
    )
    return trace


def run_adaptive_history_v2(ctx: E1Context) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    trace_rows: list[dict] = []
    low_exp = "lowF_exclusion_cip_affine_k256_p0.5_F0_0.70"
    border_exp = "borderline_cert_vbp_k128_p0.1_F0_0.90"

    trace_ns = sorted(set(np.rint(np.geomspace(20, 5000, 45)).astype(int).tolist() + [1, 5000]))

    head_cip = _compile_head(ctx, "cip-affine", 256)
    for seed in [42, 43, 44, 45, 46]:
        z, w, W_max = _run_history_adaptive_trials(
            ctx,
            head_cip,
            mask_p=0.5,
            alpha=ALPHA_DEFAULT,
            seed=seed,
            n_max=5000,
        )
        rows.append(
            _summarize_run(
                z=z,
                w=w,
                W_max=W_max,
                experiment=low_exp,
                seed=seed,
                n_max=5000,
                target_F=0.70,
                direction="above",
            )
        )
        trace_rows.extend(
            _trace_run(
                z=z,
                w=w,
                W_max=W_max,
                experiment=low_exp,
                seed=seed,
                proposal="history",
                alpha=ALPHA_DEFAULT,
                trace_ns=trace_ns,
            )
        )

    head_vbp = _compile_head(ctx, "vbp", 128)
    for seed in [101, 102, 103, 104, 105]:
        z, w, W_max = _run_history_adaptive_trials(
            ctx,
            head_vbp,
            mask_p=0.1,
            alpha=ALPHA_DEFAULT,
            seed=seed,
            n_max=10000,
        )
        rows.append(
            _summarize_run(
                z=z,
                w=w,
                W_max=W_max,
                experiment=border_exp,
                seed=seed,
                n_max=10000,
                target_F=0.90,
                direction="below",
            )
        )

    baseline = _baseline_rows()
    history = pd.DataFrame(rows)
    out = pd.concat([baseline, history], ignore_index=True, sort=False)
    out.to_csv(TABLE_DIR / "adaptive_vs_iid_v2.csv", index=False)

    trace = pd.concat([_baseline_trace_rows(), pd.DataFrame(trace_rows)], ignore_index=True, sort=False)
    trace.to_csv(TABLE_DIR / "adaptive_vs_iid_v2_history_trace.csv", index=False)
    _plot_three_way(out, trace)
    return out, trace


def _summary_stats(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    rows = []
    labels = [("iid", "n/a"), ("adaptive", "static"), ("adaptive", "history")]
    for sampling, proposal in labels:
        vals = df[
            (df["experiment"] == experiment)
            & (df["sampling"] == sampling)
            & (df["proposal"] == proposal)
        ]["n_certify"].astype(float)
        if vals.empty:
            continue
        arr = vals.to_numpy()
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        rows.append({"sampling": sampling, "proposal": proposal, "q1": q1, "median": med, "q3": q3, "n": len(arr)})
    return pd.DataFrame(rows)


def _plot_three_way(df: pd.DataFrame, history_trace: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
        }
    )
    colors = {"n/a": "#355C7D", "static": "#C06C84", "history": "#6C9A8B"}
    display = {"n/a": "i.i.d.", "static": "static adaptive", "history": "history adaptive"}
    order = ["n/a", "static", "history"]
    low_exp = "lowF_exclusion_cip_affine_k256_p0.5_F0_0.70"
    border_exp = "borderline_cert_vbp_k128_p0.1_F0_0.90"

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), constrained_layout=True)

    ax = axes[0]
    for proposal in order:
        s = history_trace[(history_trace["experiment"] == low_exp) & (history_trace["proposal"] == proposal)].copy()
        if s.empty:
            continue
        if proposal == "history":
            grouped = s.groupby("n")["ucb_f"]
            n = np.array(sorted(grouped.groups.keys()), dtype=float)
            q1 = grouped.quantile(0.25).reindex(n).to_numpy()
            med = grouped.quantile(0.50).reindex(n).to_numpy()
            q3 = grouped.quantile(0.75).reindex(n).to_numpy()
            ax.fill_between(n, q1, q3, color=colors[proposal], alpha=0.16, linewidth=0)
            ax.plot(n, med, marker="o", markersize=2.4, linewidth=1.6, color=colors[proposal], label=display[proposal])
        else:
            ax.plot(
                s["n"],
                s["ucb_f"],
                marker="o",
                markersize=2.4,
                linewidth=1.6,
                color=colors[proposal],
                label=display[proposal],
            )
    ax.axhline(0.70, color="0.25", linestyle="--", linewidth=1.0, label="target F0=0.70")
    ax.set_xscale("log")
    ax.set_xlabel("samples n")
    ax.set_ylabel("UCB(F)")
    ax.set_title("Low-fidelity exclusion")
    ax.grid(True)
    ax.legend(frameon=True, fontsize=7.5)

    ax = axes[1]
    sub = df[df["experiment"] == border_exp].copy()
    for pos, proposal in enumerate(order):
        vals = sub[sub["proposal"] == proposal]["n_certify"].astype(float).to_numpy()
        if vals.size == 0:
            continue
        ax.boxplot(
            vals,
            positions=[pos],
            widths=0.42,
            patch_artist=True,
            boxprops={"facecolor": colors[proposal], "alpha": 0.28, "edgecolor": colors[proposal]},
            medianprops={"color": colors[proposal], "linewidth": 1.8},
            whiskerprops={"color": colors[proposal]},
            capprops={"color": colors[proposal]},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": colors[proposal],
                "markeredgecolor": colors[proposal],
            },
        )
        jitter = np.linspace(-0.07, 0.07, num=len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(np.full_like(vals, pos, dtype=float) + jitter, vals, s=18, color=colors[proposal], zorder=3)
    ax.set_xticks(range(len(order)), [display[p] for p in order], rotation=12, ha="right")
    ax.set_ylabel("n to certify F >= 0.90")
    ax.set_title("Borderline certification")
    ax.grid(True, axis="y")

    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"adaptive_three_way.{ext}", dpi=300)
    plt.close(fig)

    stats = pd.concat([_summary_stats(df, low_exp), _summary_stats(df, border_exp)], keys=[low_exp, border_exp])
    stats.to_csv(TABLE_DIR / "adaptive_vs_iid_v2_summary.csv")



def main() -> None:
    start_time = time.time()
    _ensure_dirs()
    print(f"[setup-v2] repo={REPO_ROOT}", flush=True)
    ctx = _load_e1_context()
    print(f"[setup-v2] device={ctx.device} n_test={ctx.a_test.shape[0]}", flush=True)
    df, _trace = run_adaptive_history_v2(ctx)
    print(f"[done] runtime={(time.time() - start_time) / 60.0:.1f} min")
    print(f"[done-v2] wrote {TABLE_DIR / 'adaptive_vs_iid_v2.csv'}", flush=True)


if __name__ == "__main__":
    main()
