"""μ-law companding curve (μ = 255) with the 8-bit grids it implies, and the relative error of
8-bit uniform vs 8-bit μ-law across amplitudes.  Run: uv run --with matplotlib --with numpy python mulaw.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, DIM = "#0b0714", "#ece6f7", "#8a81a3"
ORANGE, PURPLE, CYAN, GREEN, YELLOW = "#ff6a3d", "#d05cff", "#38d6ff", "#4dff7c", "#ffe23d"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.edgecolor": DIM, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                     "text.color": FG, "font.size": 15, "axes.titlesize": 18, "axes.spines.top": False,
                     "axes.spines.right": False, "grid.color": "#2a2140", "grid.linestyle": ":"})
mu = 255.0
F = lambda x: np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
Finv = lambda y: np.sign(y) * np.expm1(np.abs(y) * np.log1p(mu)) / mu

fig, axes = plt.subplots(1, 2, figsize=(16, 7.8), dpi=100)
ax = axes[0]
x = np.linspace(0, 1, 1000)
ax.plot(x, F(x), color=ORANGE, lw=3, label="μ-law  F(x) = ln(1+255|x|) / ln 256")
ax.plot(x, x, color=DIM, lw=1.5, ls="--", label="identity (uniform)")
codes = np.arange(0, 128, 8) / 127                   # every 8th μ-law code, for legibility
for c in codes:
    ax.plot([Finv(c), Finv(c)], [0, c], color=CYAN, lw=0.8, alpha=0.7)
    ax.plot([0, Finv(c)], [c, c], color=CYAN, lw=0.8, alpha=0.7)
ax.scatter(Finv(codes), codes, color=CYAN, s=22, zorder=5, label="every 8th of the 128 magnitude codes")
ax.scatter([0.01], [F(0.01)], color=YELLOW, s=160, zorder=6, edgecolor=FG)
ax.annotate(f"x = 0.01 lands at F = {F(0.01):.3f}:\na 1 % signal gets 23 % of the codes", (0.01, F(0.01)),
            xytext=(0.2, 0.12), fontsize=13.5, color=YELLOW, arrowprops=dict(arrowstyle="->", color=YELLOW))
ax.set_xlabel("input amplitude |x|"); ax.set_ylabel("compressed value F(x), quantized uniformly")
ax.set_title("Bend the ruler: equal steps in F, fine steps near zero in x", fontsize=16)
ax.legend(loc="lower right", frameon=False, fontsize=12.5); ax.grid(True)

ax = axes[1]
amps = np.logspace(-3, 0, 2000)
d = 1 / 128
unif = np.abs(np.round(amps / d) * d - amps) / amps
mlaw = np.abs(Finv(np.round(F(amps) * 127) / 127) - amps) / amps
ax.plot(amps, 100 * unif, color=PURPLE, lw=0.9, alpha=0.55, label="8-bit uniform, Δ = 1/128 (actual)")
ax.plot(amps, 100 * mlaw, color=ORANGE, lw=0.9, alpha=0.55, label="8-bit μ-law, sign + 7 bits (actual)")
ax.plot(amps, 100 * (d / 2) / amps, color=PURPLE, lw=2.5, ls="--", label="uniform worst case  Δ/2 / x")
ax.axhline(100 * (256 ** (1 / 254) - 1), color=ORANGE, lw=2.5, ls="--", label="μ-law worst case ≈ 2.2 %")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.01, 300)
ax.axvline(0.01, color=YELLOW, lw=1, ls=":")
ax.text(0.0105, 150, "x = 0.01:\n21.9 % vs 0.10 %", color=YELLOW, fontsize=13)
ax.set_xlabel("input amplitude |x| (log)"); ax.set_ylabel("relative error |x̂ − x| / |x|  (%)")
ax.set_title("Relative error: uniform explodes on quiet samples, μ-law stays flat", fontsize=16)
ax.legend(loc="lower left", frameon=False, fontsize=12.5); ax.grid(True)
out = os.path.join(os.path.dirname(__file__), "..", "..", "img", "scalar-quantization", "mulaw.png")
fig.tight_layout(); fig.savefig(out); print("wrote", os.path.abspath(out))
