"""
Publication-quality plotting script (v2) for the CIF paper.

Produces 5 main figures + 4 supplementary figures from CSV results.
All outputs go to doc/figures_v2/ as both PDF and PNG (300 dpi).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
RESULTS_DIR = _HERE.parent / "results"
FIGURES_DIR = _HERE.parent / "figures"

# ---------------------------------------------------------------------------
# Method display-name mapping
# ---------------------------------------------------------------------------
METHOD_DISPLAY = {
    "cip-const": "Hard pruning",
    "cip-affine": "Soft interv.",
    "vbp": "Variance-based",
    "random": "Random",
}

METRIC_DISPLAY = {
    "zero_one": r"0--1 loss",
    "clipped_kl": "Clipped KL",
    "clipped_l2": r"Clipped L$_2$",
}

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
def _set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "text.usetex": False,
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.frameon": True,
        "legend.edgecolor": "0.8",
        "legend.fancybox": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.3,
        "figure.dpi": 150,
    })


# Qualitative color palette (colorblind-safe)
PALETTE = sns.color_palette("colorblind", n_colors=10)


def _save(fig: plt.Figure, stem: str) -> None:
    """Save figure as PDF + PNG to FIGURES_DIR."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = FIGURES_DIR / f"{stem}.{ext}"
        dpi = 300 if ext == "png" else None
        fig.savefig(out, format=ext, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  -> {stem}.pdf / .png")


# ===================================================================
# Figure 1: e1_cs_width_vs_n  --  CS width convergence
# ===================================================================
def plot_e1_cs_width_vs_n() -> None:
    print("[Fig 1] CS width vs n ...")
    df = pd.read_csv(RESULTS_DIR / "e1_cs_traces.csv")
    sub = df[
        (df["method"] == "cip-const")
        & (df["keep"] == 256)
        & (df["mask_p"] == 0.5)
    ].copy()
    assert not sub.empty, "No data for cip-const / keep=256 / mask_p=0.5"

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    # Color scheme: blue for iid, orange for adaptive
    c_iid, c_ada = PALETTE[0], PALETTE[1]

    line_specs = [
        ("iid", "hoeffding", "-",  c_iid,  "i.i.d. / Hoeffding"),
        ("iid", "betting",   "--", c_iid,  "i.i.d. / Betting"),
        ("adaptive", "hoeffding", "-",  c_ada, "Adaptive / Hoeffding"),
        ("adaptive", "betting",   "--", c_ada, "Adaptive / Betting"),
    ]

    for sampling, cs_type, ls, color, label in line_specs:
        s = sub[
            (sub["sampling"] == sampling) & (sub["cs_type"] == cs_type)
        ].sort_values("n")
        if s.empty:
            continue
        ax.plot(
            s["n"].to_numpy(),
            (2.0 * s["radius"]).to_numpy(),
            linestyle=ls, linewidth=1.3, color=color, label=label,
        )

    # Reference line ~ sqrt(log(n)/n)
    s_ref = sub[
        (sub["sampling"] == "iid") & (sub["cs_type"] == "hoeffding")
    ].sort_values("n")
    n_vals = s_ref["n"].to_numpy(dtype=np.float64)
    ref = np.sqrt(np.log(n_vals) / n_vals)
    width0 = 2.0 * s_ref["radius"].iloc[0]
    scale = float(width0 / ref[0])
    ax.plot(
        n_vals, scale * ref,
        color="grey", linestyle=":", linewidth=1.2,
        label=r"$\propto \sqrt{\log n\,/\,n}$",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"Sample size $n$")
    ax.set_ylabel(r"CS width ($2 \times$ radius)")
    ax.annotate(
        r"Hard pruning, $k\!=\!256$, $p\!=\!0.5$",
        xy=(0.98, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
    )
    ax.legend(loc="upper right", framealpha=0.9, bbox_to_anchor=(1.0, 0.85))
    _save(fig, "e1_cs_width_vs_n")


# ===================================================================
# Figure 2: e1_certification_speedup  --  Betting vs Hoeffding speedup
# ===================================================================
def plot_e1_certification_speedup() -> None:
    print("[Fig 2] Certification speedup ...")
    df = pd.read_csv(RESULTS_DIR / "e1_certification.csv")
    sub = df[
        (df["keep"] == 256) & (df["mask_p"] == 0.1) & (df["sampling"] == "iid")
    ].copy()
    assert not sub.empty, "No data for keep=256 / mask_p=0.1 / iid"

    methods_all = ["cip-affine", "cip-const", "vbp", "random"]
    F0s = sorted(sub["F_0"].unique())

    # Build speedup table.  For each method x F_0 where BOTH cs_types exist:
    #   speedup = n_hoeffding / n_betting
    # If hoeffding fails but betting succeeds -> infinity
    # If both fail -> skip
    # If neither certifies under any cs_type at all -> omit method
    records = []
    any_certify = set()
    for m in methods_all:
        for f0 in F0s:
            row_h = sub[
                (sub["method"] == m) & (sub["F_0"] == f0)
                & (sub["cs_type"] == "hoeffding")
            ]
            row_b = sub[
                (sub["method"] == m) & (sub["F_0"] == f0)
                & (sub["cs_type"] == "betting")
            ]
            if row_h.empty or row_b.empty:
                continue
            cert_h = bool(row_h["certified"].iloc[0])
            cert_b = bool(row_b["certified"].iloc[0])
            n_h = int(row_h["n_certify"].iloc[0])
            n_b = int(row_b["n_certify"].iloc[0])

            if cert_h or cert_b:
                any_certify.add(m)

            if cert_h and cert_b:
                speedup = n_h / n_b
                records.append(dict(method=m, F_0=f0, speedup=speedup, inf=False))
            elif (not cert_h) and cert_b:
                records.append(dict(method=m, F_0=f0, speedup=np.nan, inf=True))
            # else: both fail -- omit

    # Keep only methods that ever certify
    methods_keep = [m for m in methods_all if m in any_certify]
    rdf = pd.DataFrame(records)
    rdf = rdf[rdf["method"].isin(methods_keep)]
    # Drop F0 values with no valid speedup records (e.g. 0.99 where all fail)
    F0s = [f0 for f0 in F0s if f0 in rdf["F_0"].values]

    if rdf.empty or not F0s:
        print("  [WARN] No speedup data -- skipping figure.")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    n_methods = len(methods_keep)
    n_f0 = len(F0s)
    bar_width = 0.8 / n_f0
    x_base = np.arange(n_methods, dtype=np.float64)
    f0_colors = [PALETTE[2], PALETTE[3], PALETTE[4]]

    max_finite = 1.0
    for rr in records:
        if not rr["inf"] and not np.isnan(rr["speedup"]):
            max_finite = max(max_finite, rr["speedup"])

    star_y = max_finite * 2.5  # placement for infinity markers

    for fi, f0 in enumerate(F0s):
        offsets = x_base + (fi - (n_f0 - 1) / 2) * bar_width
        heights = []
        is_inf = []
        for m in methods_keep:
            row = rdf[(rdf["method"] == m) & (rdf["F_0"] == f0)]
            if row.empty:
                heights.append(0)
                is_inf.append(False)
            elif row["inf"].iloc[0]:
                heights.append(star_y)
                is_inf.append(True)
            else:
                heights.append(row["speedup"].iloc[0])
                is_inf.append(False)

        bars = ax.bar(
            offsets, heights, width=bar_width,
            color=f0_colors[fi], edgecolor="white", linewidth=0.5,
            label=f"threshold $F_0 = {f0}$",
        )
        # Put stars on infinite speedup bars
        for bar, inf_flag, h in zip(bars, is_inf, heights):
            if inf_flag:
                bar.set_alpha(0.25)
                cx = bar.get_x() + bar.get_width() / 2
                ax.plot(cx, star_y, marker="*", markersize=10,
                        color=f0_colors[fi], zorder=5)
                ax.annotate(
                    r"$\infty$", xy=(cx, star_y * 1.15),
                    ha="center", va="bottom", fontsize=8,
                    color=f0_colors[fi], fontweight="bold",
                )

    # Speedup reference lines
    ax.axhline(1, color="grey", linestyle="-", linewidth=1, alpha=0.6)
    ax.axhline(10, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(30, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.annotate("no speedup", xy=(n_methods - 0.5, 1.05), fontsize=7,
                color="grey", ha="right")
    ax.annotate(r"$10\times$", xy=(n_methods - 0.5, 10 * 1.08), fontsize=7,
                color="grey", ha="right")
    ax.annotate(r"$30\times$", xy=(n_methods - 0.5, 30 * 1.08), fontsize=7,
                color="grey", ha="right")

    ax.set_yscale("log")
    ax.set_xticks(x_base)
    ax.set_xticklabels([METHOD_DISPLAY.get(m, m) for m in methods_keep])
    ax.set_ylabel(r"Speedup  ($n_{\mathrm{Hoeffding}} \;/\; n_{\mathrm{Betting}}$)")
    # Title in LaTeX caption, not here
    ax.legend(framealpha=0.9)
    ax.set_ylim(bottom=0.8)
    _save(fig, "e1_certification_speedup")


# ===================================================================
# Figure 3: e2_cs_convergence  --  GPT-2 IOI patching CS
# ===================================================================
def plot_e2_cs_convergence() -> None:
    print("[Fig 3] E2 CS convergence ...")
    df = pd.read_csv(RESULTS_DIR / "e2_patching_cs.csv")
    sub = df[df["sampling"] == "iid"].copy()
    assert not sub.empty

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    sizes = sorted(sub["circuit_size"].unique())
    # Sequential colormap for circuit sizes
    cmap = plt.cm.viridis
    size_colors = [cmap(i / max(1, len(sizes) - 1)) for i in range(len(sizes))]

    for k, size in enumerate(sizes):
        color = size_colors[k]
        s_h = sub[
            (sub["circuit_size"] == size) & (sub["cs_type"] == "hoeffding")
        ].sort_values("n")
        s_b = sub[
            (sub["circuit_size"] == size) & (sub["cs_type"] == "betting")
        ].sort_values("n")

        if s_h.empty:
            continue

        n_arr = s_h["n"].to_numpy()
        mu = s_h["mu_hat"].to_numpy()

        # Hoeffding CS -- light shading
        ax.fill_between(
            n_arr,
            s_h["lcb_mu"].to_numpy(),
            np.minimum(s_h["ucb_mu"].to_numpy(), 1.05),  # clip for display
            color=color, alpha=0.10, linewidth=0,
        )
        # Betting CS -- darker shading
        if not s_b.empty:
            ax.fill_between(
                s_b["n"].to_numpy(),
                s_b["lcb_mu"].to_numpy(),
                np.minimum(s_b["ucb_mu"].to_numpy(), 1.05),
                color=color, alpha=0.25, linewidth=0,
            )
        # Point estimate line on top
        ax.plot(n_arr, mu, color=color, linewidth=1.3, label=f"{size} heads")

    # Threshold lines
    for thr in [0.80, 0.90, 0.95]:
        ax.axhline(
            thr, color="grey", linestyle="--", linewidth=0.8, alpha=0.7,
        )
        ax.annotate(
            f"{thr:.0%}", xy=(sub["n"].max() * 1.02, thr),
            va="center", fontsize=7, color="grey",
        )

    ax.set_xlabel(r"Sample size $n$ (prompt pairs)")
    ax.set_ylabel(r"Normalised patching effect $\hat{\mu}$")
    # Title in LaTeX caption
    ax.legend(title="Circuit size", framealpha=0.9, loc="lower right")
    ax.set_ylim(0.0, 1.08)
    _save(fig, "e2_cs_convergence")


# ===================================================================
# Figure 4: e2_completeness_vs_size  --  CS at n=2000 vs circuit size
# ===================================================================
def plot_e2_completeness_vs_size() -> None:
    print("[Fig 4] E2 completeness vs size ...")
    df = pd.read_csv(RESULTS_DIR / "e2_patching_cs.csv")
    sub = df[(df["n"] == 2000) & (df["sampling"] == "iid")].copy()
    assert not sub.empty

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    sizes = sorted(sub["circuit_size"].unique())

    for cs_type, ls, lw, color, alpha_err, label in [
        ("hoeffding", "-",  2, PALETTE[0], 0.8, "Hoeffding"),
        ("betting",   "--", 2, PALETTE[1], 0.8, "Betting"),
    ]:
        s = sub[sub["cs_type"] == cs_type].set_index("circuit_size")
        mu = [float(s.loc[sz, "mu_hat"]) for sz in sizes]
        lcb = [float(s.loc[sz, "lcb_mu"]) for sz in sizes]
        ucb = [float(s.loc[sz, "ucb_mu"]) for sz in sizes]
        yerr_lo = [m - l for m, l in zip(mu, lcb)]
        yerr_hi = [u - m for m, u in zip(mu, ucb)]
        ax.errorbar(
            sizes, mu,
            yerr=[yerr_lo, yerr_hi],
            fmt="o", linestyle=ls, linewidth=lw, capsize=5,
            color=color, label=label, alpha=alpha_err, markersize=5,
        )

    for thr in [0.80, 0.90, 0.95]:
        ax.axhline(thr, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.annotate(
            f"{thr:.0%}", xy=(sizes[-1] + 0.5, thr),
            va="center", fontsize=7, color="grey",
        )

    ax.set_xlabel("Circuit size (number of attention heads)")
    ax.set_ylabel(r"$\hat{\mu}$ with CS error bars")
    # Title in LaTeX caption
    ax.set_xticks(sizes)
    ax.legend(framealpha=0.9)
    _save(fig, "e2_completeness_vs_size")


# ===================================================================
# Figure 5: e3_fidelity_across_pi  --  Sensitivity to intervention dist
# ===================================================================
def plot_e3_fidelity_across_pi() -> None:
    print("[Fig 5] E3 fidelity across pi ...")
    df = pd.read_csv(RESULTS_DIR / "e3_sensitivity.csv")
    assert not df.empty

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    metrics = ["zero_one", "clipped_kl", "clipped_l2"]
    offsets = {"zero_one": -0.012, "clipped_kl": 0.0, "clipped_l2": 0.012}
    colors_m = [PALETTE[0], PALETTE[2], PALETTE[4]]

    for k, metric in enumerate(metrics):
        s = df[df["metric"] == metric].sort_values("mask_p")
        x = s["mask_p"].to_numpy(dtype=np.float64) + offsets[metric]
        y = s["f_hat"].to_numpy(dtype=np.float64)
        lo = s["lcb_f"].to_numpy(dtype=np.float64)
        hi = np.minimum(s["ucb_f"].to_numpy(dtype=np.float64), 1.0)
        yerr = np.vstack([y - lo, hi - y])
        ax.errorbar(
            x, y, yerr=yerr,
            fmt="o-", capsize=3, linewidth=1.3, markersize=4,
            color=colors_m[k],
            label=METRIC_DISPLAY.get(metric, metric),
        )

    ax.set_xlabel(r"Mask swap probability $p$")
    ax.set_ylabel(r"Fidelity $\hat{F}$ (with CS error bars)")
    # Title in LaTeX caption
    ax.set_ylim(0.0, 1.05)
    ax.legend(framealpha=0.9)
    _save(fig, "e3_fidelity_across_pi")


# ===================================================================
# Supplementary: e1_cs_overlap  --  Paired CS overlap
# ===================================================================
def plot_e1_cs_overlap() -> None:
    print("[Supp] E1 CS overlap ...")
    df_pairs = pd.read_csv(RESULTS_DIR / "e1_paired.csv")
    df_pairs_h = df_pairs[df_pairs["cs_type"] == "hoeffding"].copy()

    # Find the non-significant pair
    row = df_pairs_h[df_pairs_h["significant"] == False].head(1)
    if row.empty:
        row = df_pairs_h.head(1)
    pair = str(row["pair"].iloc[0])
    m1, m2 = pair.split("_vs_", 1)
    keep = int(row["keep"].iloc[0])
    mask_p = float(row["mask_p"].iloc[0])

    # Get individual CS intervals from traces
    df_trace = pd.read_csv(RESULTS_DIR / "e1_cs_traces.csv")
    s = df_trace[
        (df_trace["keep"] == keep)
        & (df_trace["mask_p"] == mask_p)
        & (df_trace["sampling"] == "iid")
        & (df_trace["n"] == 10000)
        & (df_trace["method"].isin([m1, m2]))
        & (df_trace["cs_type"] == "hoeffding")
    ].copy()
    assert s.shape[0] == 2, f"Expected 2 rows, got {s.shape[0]}"
    s = s.set_index("method")

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    colors_pair = [PALETTE[0], PALETTE[2]]
    intervals = []

    for y_pos, m, c in [(1, m1, colors_pair[0]), (2, m2, colors_pair[1])]:
        f_hat = 1.0 - float(s.loc[m, "r_hat"])
        lcb = float(s.loc[m, "lcb_f"])
        ucb = min(float(s.loc[m, "ucb_f"]), 1.0)
        intervals.append((lcb, ucb))
        ax.errorbar(
            x=f_hat, y=y_pos,
            xerr=[[f_hat - lcb], [ucb - f_hat]],
            fmt="o", capsize=4, linewidth=1.5, markersize=6,
            color=c, label=METHOD_DISPLAY.get(m, m),
        )

    # Grey overlap region
    overlap_lo = max(lo for lo, _ in intervals)
    overlap_hi = min(hi for _, hi in intervals)
    if overlap_hi > overlap_lo:
        ax.axvspan(overlap_lo, overlap_hi, color="grey", alpha=0.15,
                   label="Overlap (non-significant)")

    ax.set_yticks([1, 2])
    ax.set_yticklabels([METHOD_DISPLAY.get(m1, m1), METHOD_DISPLAY.get(m2, m2)])
    ax.set_xlabel(r"Fidelity $\hat{F}$ (Hoeffding CS interval)")
    ax.set_title(f"CS overlap: {METHOD_DISPLAY.get(m1, m1)} vs {METHOD_DISPLAY.get(m2, m2)}"
                 f" ($k$={keep}, $p$={mask_p})")
    ax.legend(framealpha=0.9, loc="lower right")
    _save(fig, "e1_cs_overlap")


# ===================================================================
# Supplementary: e2_compute_savings  --  Forward passes to certify
# ===================================================================
def plot_e2_compute_savings() -> None:
    print("[Supp] E2 compute savings ...")
    df = pd.read_csv(RESULTS_DIR / "e2_completeness.csv")
    sub = df[df["threshold"] == 0.80].copy()
    assert not sub.empty

    sizes = sorted(sub["circuit_size"].unique())
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    x = np.arange(len(sizes), dtype=np.float64)
    bar_w = 0.35

    combo = [
        ("iid", "hoeffding", "i.i.d. / Hoeffding",    PALETTE[0]),
        ("iid", "betting",   "i.i.d. / Betting",      PALETTE[1]),
        ("adaptive", "hoeffding", "Adaptive / Hoeffding", PALETTE[2]),
        ("adaptive", "betting",   "Adaptive / Betting",  PALETTE[3]),
    ]

    n_combo = len(combo)
    total_w = 0.8
    w = total_w / n_combo

    for ci, (sampling, cs_type, label, color) in enumerate(combo):
        s = sub[(sub["sampling"] == sampling) & (sub["cs_type"] == cs_type)]
        if s.empty:
            continue
        s = s.set_index("circuit_size")
        vals = []
        for sz in sizes:
            if sz in s.index:
                vals.append(int(s.loc[sz, "n_certify"]))
            else:
                vals.append(0)
        offset = (ci - (n_combo - 1) / 2) * w
        bars = ax.bar(x + offset, vals, width=w * 0.92,
                      color=color, edgecolor="white", linewidth=0.5,
                      label=label)
        # Mark uncertified bars
        for bar, sz in zip(bars, sizes):
            if sz in s.index:
                cert = bool(s.loc[sz, "certified"])
                if not cert:
                    bar.set_hatch("//")
                    bar.set_edgecolor("0.4")
                    bar.set_linewidth(0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Circuit size (number of attention heads)")
    ax.set_ylabel(r"Forward passes to certify ($n_{\mathrm{certify}}$)")
    ax.set_title(r"Forward passes to certify completeness $\geq 0.80$")
    ax.legend(framealpha=0.9, ncol=2, fontsize=9)
    _save(fig, "e2_compute_savings")


# ===================================================================
# Supplementary: e3_cs_overlap_metrics  --  CS overlap between metrics
# ===================================================================
def plot_e3_cs_overlap_metrics() -> None:
    print("[Supp] E3 CS overlap metrics ...")
    df = pd.read_csv(RESULTS_DIR / "e3_sensitivity.csv")
    sub = df[df["mask_p"] == 0.2].copy()
    assert not sub.empty

    metrics = ["zero_one", "clipped_kl", "clipped_l2"]
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    colors_m = [PALETTE[0], PALETTE[2], PALETTE[4]]
    intervals = []

    for y_pos, metric, color in zip(range(1, 4), metrics, colors_m):
        row = sub[sub["metric"] == metric].iloc[0]
        f_hat = float(row["f_hat"])
        lcb = float(row["lcb_f"])
        ucb = min(float(row["ucb_f"]), 1.0)
        intervals.append((lcb, ucb))

        ax.errorbar(
            x=f_hat, y=y_pos,
            xerr=[[f_hat - lcb], [ucb - f_hat]],
            fmt="o", capsize=4, linewidth=1.5, markersize=6,
            color=color, label=METRIC_DISPLAY.get(metric, metric),
        )

    # Highlight overlap region
    overlap_lo = max(lo for lo, _ in intervals)
    overlap_hi = min(hi for _, hi in intervals)
    if overlap_hi > overlap_lo:
        ax.axvspan(overlap_lo, overlap_hi, color="grey", alpha=0.15,
                   label="Overlap region")

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels([METRIC_DISPLAY.get(m, m) for m in metrics])
    ax.set_xlabel(r"Fidelity $\hat{F}$ (CS interval)")
    ax.set_title("CS overlap between metrics at $p=0.2$")
    ax.legend(framealpha=0.9, loc="lower right")
    _save(fig, "e3_cs_overlap_metrics")


# ===================================================================
# Supplementary: e4_coverage  --  Coverage validation
# ===================================================================
def plot_e4_coverage() -> None:
    print("[Supp] E4 coverage ...")
    df = pd.read_csv(RESULTS_DIR / "e4_coverage.csv")
    assert not df.empty

    protocols = ["iid", "adaptive", "aggressive_peeking"]
    protocol_display = {
        "iid": "i.i.d.",
        "adaptive": "Adaptive",
        "aggressive_peeking": "Aggressive\npeeking",
    }
    deltas = sorted(df["delta"].unique())

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    df_idx = df.set_index(["delta", "protocol"]).sort_index()
    x_positions = []
    x_labels = []
    idx = 0
    group_positions = {}  # delta -> list of positions
    for di, d in enumerate(deltas):
        group_positions[d] = []
        for pi, p in enumerate(protocols):
            x_positions.append(idx)
            x_labels.append(f"{protocol_display.get(p, p)}")
            group_positions[d].append(idx)
            idx += 1
        idx += 0.8  # gap between delta groups

    x_arr = np.array(x_positions, dtype=np.float64)
    bar_w = 0.35

    cov_h, std_h, cov_b, std_b = [], [], [], []
    for d in deltas:
        for p in protocols:
            cov_h.append(float(df_idx.loc[(d, p), "empirical_coverage_hoeffding"]))
            std_h.append(float(df_idx.loc[(d, p), "coverage_std_hoeffding"]))
            cov_b.append(float(df_idx.loc[(d, p), "empirical_coverage_betting"]))
            std_b.append(float(df_idx.loc[(d, p), "coverage_std_betting"]))

    cov_h, std_h = np.array(cov_h), np.array(std_h)
    cov_b, std_b = np.array(cov_b), np.array(std_b)

    ax.bar(
        x_arr - 0.5 * bar_w, cov_h, width=bar_w,
        color=PALETTE[0], edgecolor="white", linewidth=0.5,
        label="Hoeffding",
    )
    ax.bar(
        x_arr + 0.5 * bar_w, cov_b, width=bar_w,
        color=PALETTE[1], edgecolor="white", linewidth=0.5,
        label="Betting",
    )
    ax.errorbar(
        x_arr - 0.5 * bar_w, cov_h, yerr=2.0 * std_h,
        fmt="none", ecolor="black", capsize=3, linewidth=1,
    )
    ax.errorbar(
        x_arr + 0.5 * bar_w, cov_b, yerr=2.0 * std_b,
        fmt="none", ecolor="black", capsize=3, linewidth=1,
    )

    # Nominal coverage lines
    for nom in [0.90, 0.95, 0.99]:
        ax.axhline(nom, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.annotate(
            f"{nom:.0%}", xy=(x_arr[-1] + 0.8, nom),
            va="center", fontsize=7, color="grey",
        )

    # Delta group labels at bottom
    for d in deltas:
        positions = group_positions[d]
        mid = np.mean(positions)
        ax.annotate(
            f"$\\delta={d}$",
            xy=(mid, 0), xycoords=("data", "axes fraction"),
            xytext=(0, -42), textcoords="offset points",
            ha="center", va="top", fontsize=11, fontweight="bold",
        )

    ax.set_xticks(x_arr)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.88, 1.012)
    ax.set_title("Empirical coverage of confidence sequences (500 reps)")
    ax.legend(framealpha=0.9, ncol=2)
    fig.subplots_adjust(bottom=0.22)
    _save(fig, "e4_coverage")


# ===================================================================
# Main
# ===================================================================
def make_all_plots() -> None:
    _set_style()
    print(f"Saving figures to {FIGURES_DIR}/")
    print()

    # Main figures
    plot_e1_cs_width_vs_n()
    plot_e1_certification_speedup()
    plot_e2_cs_convergence()
    plot_e2_completeness_vs_size()
    plot_e3_fidelity_across_pi()

    # Supplementary figures
    plot_e1_cs_overlap()
    plot_e2_compute_savings()
    plot_e3_cs_overlap_metrics()
    plot_e4_coverage()

    print()
    print("Done -- all figures saved.")


if __name__ == "__main__":
    make_all_plots()
