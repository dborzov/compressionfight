"""Figure: the damped-cosine toy, its fixed order-2 residual and its fitted LPC residual."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

BG, INK, MUTED, GRID = "#0b0714", "#e8e4f0", "#9b93aa", "#2a2238"
C = ["#3b8fd6", "#c4812a", "#2fa06e", "#b85fb0"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
                     "font.family": "DejaVu Sans", "font.size": 13})

n = np.arange(16)
x = np.round(100 * 0.9**n * np.cos(0.6 * n)).astype(int)
fixed = x[2:] - (2 * x[1:-1] - x[:-2])
A = np.column_stack([x[1:-1], x[:-2]]).astype(float)
a, *_ = np.linalg.lstsq(A, x[2:].astype(float), rcond=None)
fitted = x[2:] - A @ a
print("x", x.tolist()); print("fixed", fixed.tolist()); print("fitted", fitted.round(2).tolist(), "a", a.round(4))
print("energies", (x[2:]**2).sum(), (fixed**2).sum(), (fitted**2).sum().round(2))

fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), dpi=100, sharey=True)
series = [("the signal  x[n] = round(100·0.9ⁿ cos 0.6n)", n, x, C[0]),
          ("fixed predictor residual  x[n] − (2x[n−1] − x[n−2])", n[2:], fixed, C[1]),
          (f"fitted predictor residual  a = ({a[0]:.3f}, {a[1]:.3f})", n[2:], fitted, C[2])]
for ax, (title, t, v, col) in zip(axes, series):
    ax.axhline(0, color=GRID, lw=1)
    ax.vlines(t, 0, v, color=col, lw=2.2); ax.plot(t, v, "o", color=col, ms=7)
    ax.set_title(title, fontsize=12.5, loc="left")
    ax.text(0.97, 0.06, f"Σ v² = {np.sum(v.astype(float)**2):.0f}", transform=ax.transAxes, ha="right", fontsize=13, color=INK)
    ax.set_xlabel("n"); ax.set_xticks(range(0, 16, 3)); ax.spines[["top", "right"]].set_visible(False); ax.grid(True, axis="y", lw=0.5)
axes[0].set_ylim(-65, 105)
fig.suptitle("A resonance is an all-pole filter: two coefficients fitted to the block flatten it to rounding noise", x=0.01, ha="left", fontsize=15)
out = os.path.join(os.path.dirname(__file__), "..", "..", "img", "linear-prediction", "resonance.png")
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
from PIL import Image
w = Image.open(out).width                      # re-save so the trimmed image is exactly 1600 px wide
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3, dpi=100 * 1600 / w)
print("wrote", out)
