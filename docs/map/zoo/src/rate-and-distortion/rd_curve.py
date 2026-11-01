"""R(D) curve of a Gaussian source with the operating points of a uniform quantizer at Δ = 1, 2, 4.
Run: uv run --with matplotlib --with numpy python rd_curve.py
Writes ../../img/rate-and-distortion/rd_curve.png (1600 px wide, dark theme)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()

BG, FG, DIM = "#f3eadb", "#2a211b", "#5a4a3d"
ORANGE, PURPLE, CYAN, GREEN = "#b05a4c", "#6b4a7a", "#246b66", "#6f8f4e"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.edgecolor": DIM, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                     "text.color": FG, "font.size": 15, "axes.titlesize": 19, "axes.spines.top": False,
                     "axes.spines.right": False, "grid.color": "#b89f78", "grid.linestyle": ":"})

x = np.array([3.7, -1.2, 0.4, 8.9, -5.5, 2.1])
var = x.var()                                    # 19.57: a Gaussian with the toy's spread
h = 0.5 * np.log2(2 * np.pi * np.e * var)        # differential entropy, bits

fig, ax = plt.subplots(figsize=(16, 8.2), dpi=100)
D = np.logspace(np.log10(0.02), np.log10(var), 400)
ax.plot(D, 0.5 * np.log2(var / D), color=PURPLE, lw=3, label=r"Shannon $R(D)=\frac{1}{2}\log_2(\sigma^2/D)$, the floor")
steps = np.array([1, 2, 4])
Dq, Rq = steps**2 / 12, h - np.log2(steps)       # high-resolution uniform quantizer + ideal entropy coder
ax.plot(D, h - 0.5 * np.log2(12 * D), color=ORANGE, lw=2.2, ls="--",
        label="uniform step Δ + entropy coder: $R = h(X) - \\log_2\\Delta$,  $D=\\Delta^2/12$")
ax.scatter(Dq, Rq, s=220, color=ORANGE, zorder=5, edgecolor=FG, lw=1.5)
for s, d, r in zip(steps, Dq, Rq):
    ax.annotate(f"Δ = {s}\nD = {d:.3f}\nR = {r:.2f} bits", (d, r), xytext=(14, 10), textcoords="offset points",
                fontsize=13.5, color=FG)
# the constant gap
d0 = Dq[1]
ax.annotate("", xy=(d0, 0.5 * np.log2(var / d0)), xytext=(d0, h - np.log2(2)),
            arrowprops=dict(arrowstyle="<->", color=CYAN, lw=2))
ax.text(d0 * 0.93, 0.5 * np.log2(var / d0) - 0.12, "0.25 bit gap (= 1.53 dB)\nat every Δ",
        color=CYAN, fontsize=13.5, va="top", ha="right")
# 6 dB per bit: a doubling of Δ
ax.annotate("", xy=(Dq[2], Rq[2]), xytext=(Dq[1], Rq[1]),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=2, connectionstyle="arc3,rad=-0.3"))
ax.text(Dq[2] * 0.55, Rq[2] - 0.5, "double Δ:  −1 bit,  ×4 error (+6.02 dB)", color=GREEN, fontsize=13.5)
ax.set_xscale("log")
ax.set_xlabel("distortion D (mean squared error), log scale")
ax.set_ylabel("rate R (bits per sample)")
ax.set_title(f"Rate against distortion for a Gaussian source with σ² = {var:.2f} (the spread of the six toy values)")
ax.set_ylim(0, 5.2); ax.grid(True)
ax.legend(loc="upper right", frameon=False, fontsize=13.5)
ax.text(0.012, 0.03, "Every lossy codec is a point on this plane. The curve is the floor; the dashed line is what rounding to a grid buys you.",
        transform=ax.transAxes, color=DIM, fontsize=13)
out = os.path.join(os.path.dirname(__file__), "..", "..", "img", "rate-and-distortion", "rd_curve.png")
fig.tight_layout(); fig.savefig(out); print("wrote", os.path.abspath(out))
