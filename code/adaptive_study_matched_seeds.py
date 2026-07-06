from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
TABLE_DIR = PROJECT_ROOT / "results"
FIG_DIR = PROJECT_ROOT / "figures"

from adaptive_study import (  # noqa: E402
    ALPHA_DEFAULT,
    _betting_interval,
    _compile_head,
    _ensure_dirs,
    _first_time_betting_one_sided,
    _load_e1_context,
    _run_config,
)


LOW_EXP = "lowF_exclusion_cip_affine_k256_p0.5_F0_0.70"
BORDER_EXP = "borderline_cert_vbp_k128_p0.1_F0_0.90"


def _summarize_lowf(z: np.ndarray, w: np.ndarray, W_max: float, *, sampling: str, proposal: str, seed: int) -> dict:
    n_cert, certified = _first_time_betting_one_sided(
        z,
        w,
        W_max=W_max,
        target_R=1.0 - 0.70,
        direction="above",
    )
    lo_R, hi_R, r_hat = _betting_interval(z, w, W_max)
    _lo_stop, _hi_stop, r_hat_stop = _betting_interval(z, w, W_max, n=n_cert)
    return {
        "experiment": LOW_EXP,
        "sampling": sampling,
        "proposal": proposal,
        "alpha": 0.0 if sampling == "iid" else ALPHA_DEFAULT,
        "seed": int(seed),
        "n_certify": int(n_cert),
        "F_hat_final": float(1.0 - r_hat),
        "radius_final": float(0.5 * (hi_R - lo_R)),
        "certified": bool(certified),
        "n_max": 5000,
        "F_hat_stop": float(1.0 - r_hat_stop),
    }


def run_lowf_matched_baselines() -> pd.DataFrame:
    _ensure_dirs()
    table_path = TABLE_DIR / "adaptive_vs_iid_v2.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing v2 table: {table_path}")

    ctx = _load_e1_context()
    head = _compile_head(ctx, "cip-affine", 256)

    rows: list[dict] = []
    for seed in [43, 44, 45, 46]:
        z, w, W_max = _run_config(
            ctx,
            head,
            method="cip-affine",
            keep=256,
            mask_p=0.5,
            sampling="iid",
            alpha=ALPHA_DEFAULT,
            seed=seed,
            n_max=5000,
        )
        rows.append(_summarize_lowf(z, w, W_max, sampling="iid", proposal="n/a", seed=seed))

        z, w, W_max = _run_config(
            ctx,
            head,
            method="cip-affine",
            keep=256,
            mask_p=0.5,
            sampling="adaptive",
            alpha=ALPHA_DEFAULT,
            seed=seed,
            n_max=5000,
        )
        rows.append(_summarize_lowf(z, w, W_max, sampling="adaptive", proposal="static", seed=seed))

    old = pd.read_csv(table_path, keep_default_na=False)
    new = pd.DataFrame(rows)

    # Keep the script idempotent while preserving all existing non-target rows.
    target = (
        old["experiment"].eq(LOW_EXP)
        & old["proposal"].isin(["n/a", "static"])
        & old["seed"].astype(int).isin([43, 44, 45, 46])
    )
    combined = pd.concat([old.loc[~target], new], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["experiment", "proposal", "seed"],
        key=lambda s: s.map({"n/a": 0, "static": 1, "history": 2}).fillna(s) if s.name == "proposal" else s,
    ).reset_index(drop=True)
    combined.to_csv(table_path, index=False)
    _write_summary(combined)
    _plot_three_way_boxplots(combined)
    return combined


def _write_summary(df: pd.DataFrame) -> None:
    rows = []
    for experiment in [LOW_EXP, BORDER_EXP]:
        for sampling, proposal in [("iid", "n/a"), ("adaptive", "static"), ("adaptive", "history")]:
            s = df[
                df["experiment"].eq(experiment)
                & df["sampling"].eq(sampling)
                & df["proposal"].eq(proposal)
            ].copy()
            if s.empty:
                continue
            vals = s["n_certify"].astype(float).to_numpy()
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            rows.append(
                {
                    "experiment": experiment,
                    "sampling": sampling,
                    "proposal": proposal,
                    "n": int(len(vals)),
                    "q1": float(q1),
                    "median": float(med),
                    "q3": float(q3),
                    "F_hat_final_mean": float(s["F_hat_final"].astype(float).mean()),
                    "radius_final_mean": float(s["radius_final"].astype(float).mean()),
                }
            )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "adaptive_vs_iid_v2_summary.csv", index=False)


def _plot_three_way_boxplots(df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
        }
    )
    colors = {"n/a": "#355C7D", "static": "#C06C84", "history": "#6C9A8B"}
    labels = {"n/a": "i.i.d.", "static": "static adaptive", "history": "history adaptive"}
    order = ["n/a", "static", "history"]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), constrained_layout=True)
    panels = [
        (LOW_EXP, "Low-fidelity exclusion", "n to exclude F >= 0.70"),
        (BORDER_EXP, "Borderline certification", "n to certify F >= 0.90"),
    ]

    for ax, (experiment, title, ylabel) in zip(axes, panels, strict=True):
        sub = df[df["experiment"].eq(experiment)].copy()
        for pos, proposal in enumerate(order):
            vals = sub[sub["proposal"].eq(proposal)]["n_certify"].astype(float).to_numpy()
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
        ax.set_xticks(range(len(order)), [labels[p] for p in order], rotation=12, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y")

    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"adaptive_three_way.{ext}", dpi=300)
    plt.close(fig)



def main() -> None:
    start_time = time.time()
    df = run_lowf_matched_baselines()
    print(f"[done] runtime={(time.time() - start_time) / 60.0:.1f} min")
    print(f"[done-v3] rows={len(df)} wrote {TABLE_DIR / 'adaptive_vs_iid_v2.csv'}", flush=True)


if __name__ == "__main__":
    main()
