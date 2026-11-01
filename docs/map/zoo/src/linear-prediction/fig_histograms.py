"""Figure: histogram of a resonant signal vs its residuals under three predictors, with order-0 entropies."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()
import numpy as np, os

BG, INK, MUTED, GRID = "#f3eadb", "#2a211b", "#5a4a3d", "#b89f78"
C = ["#246b66", "#b8892e", "#b05a4c", "#6b4a7a"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
                     "font.family": "DejaVu Sans", "font.size": 13})

rng = np.random.default_rng(3)
N = 65536
# A resonance: an all-pole (AR(2)) filter with poles at radius 0.98, angle 0.3 rad, driven by small white noise.
r, w = 0.98, 0.3
a1, a2 = 2 * r * np.cos(w), -r * r
e = rng.normal(0, 40, N)
x = np.zeros(N)
for n in range(2, N):
    x[n] = a1 * x[n - 1] + a2 * x[n - 2] + e[n]
x = np.round(x).astype(int)
print("signal std", x.std().round(1), "range", x.min(), x.max())

def H(v):
    _, c = np.unique(v, return_counts=True); p = c / c.sum(); return -(p * np.log2(p)).sum()

r0 = x[2:]
r1 = x[2:] - x[1:-1]
r2 = x[2:] - (2 * x[1:-1] - x[:-2])
A = np.column_stack([x[1:-1], x[:-2]]).astype(float)
coef, *_ = np.linalg.lstsq(A, x[2:].astype(float), rcond=None)
rl = x[2:] - np.round(A @ coef).astype(int)
print("fitted LPC", coef.round(4), "true", round(a1, 4), round(a2, 4))
panels = [("raw samples x[n]", r0), ("order-1: x[n] - x[n-1]", r1), ("fixed order-2: x[n] - (2x[n-1] - x[n-2])", r2),
          (f"fitted LPC a=({coef[0]:.3f}, {coef[1]:.3f})", rl)]
fig, axes = plt.subplots(1, 4, figsize=(16, 5.2), dpi=100, sharey=True)
lim = np.percentile(np.abs(r0), 99.5)
for ax, (title, v), col in zip(axes, panels, C):
    ax.hist(v, bins=121, range=(-lim, lim), color=col, lw=0)
    ax.set_title(title, fontsize=12.5, loc="left")
    ax.text(0.97, 0.92, f"H₀ = {H(v):.2f} bits\nstd = {v.std():.0f}", transform=ax.transAxes, ha="right", va="top", fontsize=13, color=INK)
    ax.grid(True, axis="y", lw=0.5); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-lim, lim)
    print(title, "H0", round(H(v), 3), "std", round(v.std(), 1))
axes[0].set_ylabel("count (65,534 samples)")
fig.suptitle("Same number of symbols, a different histogram: prediction moves the mass to zero", x=0.01, ha="left", fontsize=15)
out = os.path.join(os.path.dirname(__file__), "..", "..", "img", "linear-prediction", "histograms.png")
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
from PIL import Image
w = Image.open(out).width                      # re-save so the trimmed image is exactly 1600 px wide
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3, dpi=100 * 1600 / w)
print("wrote", out)
