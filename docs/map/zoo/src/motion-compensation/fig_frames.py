"""Figure: the two toy frames, the plain difference, the SAD search surface and the motion-compensated residual."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()
import numpy as np, os

BG, INK, MUTED, GRID = "#f3eadb", "#2a211b", "#5a4a3d", "#b89f78"
C = ["#246b66", "#b8892e", "#b05a4c", "#6b4a7a"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans", "font.size": 13})

f1 = np.zeros((8, 8), int); f1[2:5, 2:5] = 200
f2 = np.zeros((8, 8), int); f2[2:5, 3:6] = 200
def shift(ref, dx, dy):
    out = np.zeros_like(ref); H, W = ref.shape
    ys, xs = np.mgrid[0:H, 0:W]; sy, sx = ys - dy, xs - dx
    ok = (sy >= 0) & (sy < H) & (sx >= 0) & (sx < W)
    out[ok] = ref[sy[ok], sx[ok]]; return out
sad = np.array([[np.abs(f2 - shift(f1, dx, dy)).sum() for dx in range(-2, 3)] for dy in range(-2, 3)])
resid = f2 - shift(f1, 1, 0)

fig, axes = plt.subplots(1, 5, figsize=(16, 4.3), dpi=100)
def grid(ax, M, title, vmin, vmax, cmap):
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    for (i, j), v in np.ndenumerate(M):
        if v != 0: ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5, color=INK if cmap != "Greys" else BG)
    ax.set_title(title, fontsize=12, loc="left"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor(GRID)
grid(axes[0], f1, "frame 1\n(reference)", 0, 255, "Greys")
grid(axes[1], f2, "frame 2\n", 0, 255, "Greys")
grid(axes[2], f2 - f1, "frame 2 − frame 1\n(6 nonzero cells)", -200, 200, "PuOr")
im = axes[3].imshow(sad, cmap="YlGnBu")
for (i, j), v in np.ndenumerate(sad):
    axes[3].text(j, i, str(v), ha="center", va="center", fontsize=8.5, color=INK if v > 1500 else BG)
axes[3].set_xticks(range(5)); axes[3].set_xticklabels(range(-2, 3)); axes[3].set_yticks(range(5)); axes[3].set_yticklabels(range(-2, 3))
axes[3].set_xlabel("dx"); axes[3].set_ylabel("dy", rotation=0, labelpad=10); axes[3].set_title("SAD over the\n±2 search", fontsize=12, loc="left")
for s in axes[3].spines.values(): s.set_edgecolor(GRID)
grid(axes[4], resid, "residual after copying\nwith (+1, 0): all zero", -200, 200, "PuOr")
fig.subplots_adjust(wspace=0.45)
axes[1].annotate("", xy=(4.5, 3), xytext=(3, 3), arrowprops=dict(arrowstyle="-|>", color=C[1], lw=2.5))
axes[1].text(3.7, 1.3, "v = (+1, 0)", color=C[1], fontsize=12, ha="center")
out = os.path.join(os.path.dirname(__file__), "..", "..", "img", "motion-compensation", "frames.png")
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
from PIL import Image
w = Image.open(out).width                      # re-save so the trimmed image is exactly 1600 px wide
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3, dpi=100 * 1600 / w)
print(sad); print("wrote", out)
