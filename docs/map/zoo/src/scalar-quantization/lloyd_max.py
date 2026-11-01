"""Lloyd-Max levels for a unit Gaussian at 3 bits (8 levels) against the best uniform 8-level grid.
Run: uv run --with matplotlib --with numpy python lloyd_max.py"""
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

g = np.linspace(-6, 6, 240_001); pdf = np.exp(-g**2 / 2) / np.sqrt(2 * np.pi); w = pdf / pdf.sum()
def mse(lv):
    i = np.abs(g[:, None] - lv[None, :]).argmin(1); return np.sum(w * (g - lv[i]) ** 2)
steps = np.linspace(0.3, 1.2, 901)
s_best = steps[np.argmin([mse((np.arange(8) - 3.5) * s) for s in steps])]
uni = (np.arange(8) - 3.5) * s_best
lv = uni.copy()
for _ in range(60):
    i = np.abs(g[:, None] - lv[None, :]).argmin(1)
    lv = np.array([np.sum(w[i == k] * g[i == k]) / np.sum(w[i == k]) for k in range(8)])
bounds = (lv[1:] + lv[:-1]) / 2

fig, axes = plt.subplots(2, 1, figsize=(16, 8.4), dpi=100, sharex=True)
for ax, levels, color, name, step_txt in (
        (axes[0], uni, ORANGE, "best uniform grid", f"step {s_best:.3f}"),
        (axes[1], lv, CYAN, "Lloyd-Max, 60 iterations", "steps grow outward")):
    ax.fill_between(g, 0, pdf, color="#3a2d55", alpha=0.9)
    ax.plot(g, pdf, color=FG, lw=1.5)
    b = (levels[1:] + levels[:-1]) / 2
    for bb in b: ax.axvline(bb, color=DIM, lw=1, ls=":")
    ax.vlines(levels, 0, 0.44, color=color, lw=3)
    for l in levels: ax.text(l, 0.455, f"{l:+.3f}", ha="center", color=color, fontsize=11.5, rotation=90, va="bottom")
    ax.set_ylim(0, 0.62); ax.set_yticks([])
    ax.text(-3.55, 0.3, f"{name}\n{step_txt}\nMSE = {mse(levels):.5f}", color=color, fontsize=14.5, va="center")
axes[1].set_xlabel("value x   (unit Gaussian source; dotted lines are decision boundaries, bars are reconstruction levels)")
axes[0].set_title(f"Eight levels on a bell curve: Lloyd-Max buys {10*np.log10(mse(uni)/mse(lv)):.2f} dB over the best uniform grid "
                  "by crowding levels where the density is")
axes[1].set_xlim(-3.6, 3.6)
out = os.path.join(os.path.dirname(__file__), "..", "..", "img", "scalar-quantization", "lloyd_max.png")
fig.tight_layout(); fig.savefig(out); print("wrote", os.path.abspath(out))
