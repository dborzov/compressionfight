"""Reverse water-filling over 8 transform coordinates (the WATERLINE lab's variances), θ = 2,
against equal allocation at the same total rate.
Run: uv run --with matplotlib --with numpy python waterfill.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()

BG, FG, DIM = "#f3eadb", "#2a211b", "#5a4a3d"
ORANGE, PURPLE, CYAN, GREEN, YELLOW = "#b05a4c", "#6b4a7a", "#246b66", "#6f8f4e", "#b8892e"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.edgecolor": DIM, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                     "text.color": FG, "font.size": 15, "axes.titlesize": 18, "axes.spines.top": False,
                     "axes.spines.right": False, "grid.color": "#b89f78", "grid.linestyle": ":"})

var = np.array([64, 32, 16, 8, 4, 2, 1, 0.5]); theta = 2.0
R = np.maximum(0, 0.5 * np.log2(var / theta)); D = np.minimum(var, theta)
R_eq = R.sum() / 8; D_eq = var * 2.0 ** (-2 * R_eq)
idx = np.arange(8); labels = [f"σ²={v:g}" for v in var]

fig, axes = plt.subplots(1, 2, figsize=(16, 7.6), dpi=100, gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
ax.bar(idx, var, color="#e0d2b6", edgecolor=DIM, width=0.72, label="coordinate variance σ²")
ax.bar(idx, D, color=CYAN, alpha=0.75, width=0.72, label="distortion left: min(σ², θ)")
ax.axhline(theta, color=YELLOW, lw=2.5, ls="--")
ax.text(7.45, theta * 0.62, "water level θ = 2", color=YELLOW, ha="right", fontsize=14)
for i, (r, v) in enumerate(zip(R, var)):
    ax.text(i, max(v, theta) * 1.12, f"{r:g} bit" if r else "0 bit", ha="center",
            color=ORANGE if r else DIM, fontsize=13, fontweight="bold" if r else "normal")
ax.text(6.5, theta * 1.9, "drowned", ha="center", color=DIM, fontsize=12.5)
ax.set_yscale("log"); ax.set_ylim(0.3, 160)
ax.set_xticks(idx); ax.set_xticklabels(labels, fontsize=12.5)
ax.set_ylabel("variance / distortion (log scale)")
ax.set_title(f"Reverse water-filling: {R.sum():g} bits total, ΣD = {D.sum():g}")
ax.legend(loc="upper right", frameon=False, fontsize=13)

ax = axes[1]
w = 0.38
ax.bar(idx - w / 2, D, width=w, color=CYAN, label=f"water-filling  ΣD = {D.sum():g}")
ax.bar(idx + w / 2, D_eq, width=w, color=ORANGE, label=f"equal {R_eq:.3g} bits each  ΣD = {D_eq.sum():.1f}")
ax.set_xticks(idx); ax.set_xticklabels(labels, fontsize=12.5)
ax.set_ylabel("distortion per coordinate")
ax.set_title(f"Same {R.sum():g} bits, two allocations ({10*np.log10(D_eq.sum()/D.sum()):.1f} dB apart)")
ax.legend(loc="upper right", frameon=False, fontsize=13)
ax.text(0.02, 0.9, "Equal bits waste\nprecision on σ² = 0.5\nand starve σ² = 64", transform=ax.transAxes, color=DIM, fontsize=13)
out = os.path.join(os.path.dirname(__file__), "..", "..", "img", "rate-and-distortion", "waterfill.png")
fig.tight_layout(); fig.savefig(out); print("wrote", os.path.abspath(out))
