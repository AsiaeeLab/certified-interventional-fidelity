"""Plot the adaptive-sampling study figures (paper Appendix E) from CSV results.

Inputs (produced by adaptive_study*.py): results/adaptive_vs_iid_v2.csv, results/alpha_sweep.csv.
Outputs: figures/adaptive_three_way.pdf, figures/alpha_sweep.pdf.
"""
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

LOW_EXP = "lowF_exclusion_cip_affine_k256_p0.5_F0_0.70"
BORDER_EXP = "borderline_cert_vbp_k128_p0.1_F0_0.90"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.size": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
})


def plot_three_way() -> None:
    df = pd.read_csv(TABLE_DIR / "adaptive_vs_iid_v2.csv")
    df["proposal"] = df["proposal"].fillna("n/a")
    colors = {"n/a": "#355C7D", "static": "#C06C84", "history": "#6C9A8B"}
    labels = {"n/a": "i.i.d.", "static": "static adaptive", "history": "history adaptive"}
    order = ["n/a", "static", "history"]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), constrained_layout=True)
    panels = [
        (LOW_EXP, "Low-fidelity exclusion", r"$n$ to exclude $F \geq 0.70$"),
        (BORDER_EXP, "Borderline certification", r"$n$ to certify $F \geq 0.90$"),
    ]
    for ax, (experiment, title, ylabel) in zip(axes, panels):
        sub = df[df["experiment"].eq(experiment)]
        for pos, proposal in enumerate(order):
            vals = sub[sub["proposal"].eq(proposal)]["n_certify"].astype(float).to_numpy()
            if vals.size == 0:
                continue
            ax.boxplot(
                vals, positions=[pos], widths=0.42, patch_artist=True,
                boxprops={"facecolor": colors[proposal], "alpha": 0.28, "edgecolor": colors[proposal]},
                medianprops={"color": colors[proposal], "linewidth": 1.8},
                whiskerprops={"color": colors[proposal]},
                capprops={"color": colors[proposal]},
                flierprops={"marker": "o", "markersize": 3,
                            "markerfacecolor": colors[proposal], "markeredgecolor": colors[proposal]},
            )
            jitter = np.linspace(-0.07, 0.07, num=len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full_like(vals, pos, dtype=float) + jitter, vals, s=18,
                       color=colors[proposal], zorder=3)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([labels[p] for p in order], rotation=12, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y")
    fig.savefig(FIG_DIR / "adaptive_three_way.pdf", dpi=300)
    plt.close(fig)


def plot_alpha_sweep() -> None:
    adf = pd.read_csv(TABLE_DIR / "alpha_sweep.csv")
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    colors = {0.90: "#355C7D", 0.95: "#C06C84"}
    xvals = sorted(adf["alpha"].unique())
    for f0 in [0.90, 0.95]:
        medians, lo_err, hi_err = [], [], []
        for alpha in xvals:
            vals = adf[(adf["F_0"] == f0) & (adf["alpha"] == alpha)]["n_certify"].astype(float).to_numpy()
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            medians.append(med)
            lo_err.append(med - q1)
            hi_err.append(q3 - med)
        ax.errorbar(xvals, medians, yerr=[lo_err, hi_err], marker="o", linewidth=1.8,
                    capsize=3, color=colors[f0], label=rf"threshold $F_0={f0:.2f}$")
    ax.set_xlabel(r"mixture rate $\alpha$ ($\alpha=0$: i.i.d.)")
    ax.set_ylabel(r"$n$ to certify")
    ax.set_xticks(xvals)
    ax.grid(True)
    ax.legend(frameon=True)
    fig.savefig(FIG_DIR / "alpha_sweep.pdf", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_three_way()
    plot_alpha_sweep()
    print(f"wrote figures to {FIG_DIR}")
