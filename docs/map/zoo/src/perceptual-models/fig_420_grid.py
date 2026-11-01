"""Figure: 4:4:4 / 4:2:2 / 4:2:0 sampling grids on an 8x4 pixel patch, with numbers per 4 pixels.
Output: img/perceptual-models/chroma-420.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()
from matplotlib.patches import Rectangle

BG, INK, INK2, MUTED, GRID = "#f3eadb", "#2a211b", "#5a4a3d", "#5a4a3d", "#b89f78"
C = ["#246b66", "#b8892e", "#b05a4c", "#6b4a7a", "#6f8f4e"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "font.size": 13, "axes.titlesize": 15})

fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), dpi=100)
W, H = 8, 4
schemes = [("4:4:4", 1, 1, "4 Y + 4 Cb + 4 Cr = 12 numbers per 2×2\n(100%)"),
           ("4:2:2", 2, 1, "4 Y + 2 Cb + 2 Cr = 8 numbers per 2×2\n(67%)"),
           ("4:2:0", 2, 2, "4 Y + 1 Cb + 1 Cr = 6 numbers per 2×2\n(50%): half the data, nobody notices")]
for ax, (name, sx, sy, note) in zip(axes, schemes):
    ax.set_xlim(-0.3, W + 0.3); ax.set_ylim(-0.3, H + 0.3); ax.set_aspect("equal"); ax.axis("off")
    for x in range(W):
        for y in range(H):
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#e0d2b6", edgecolor=GRID, linewidth=1))
            ax.scatter(x + 0.5, y + 0.5, s=50, color=INK, zorder=3)        # luma sample at every pixel
    for x in range(0, W, sx):
        for y in range(0, H, sy):
            cx, cy = x + sx / 2, y + sy / 2
            ax.scatter(cx, cy, s=360, marker="s", facecolor="none", edgecolor=C[1], linewidth=2.2, zorder=4)
    ny = W * H; nc = (W // sx) * (H // sy)
    ax.set_title(f"{name}:  {ny} Y,  {nc} Cb,  {nc} Cr")
    ax.text(W / 2, -0.25, note, ha="center", va="top", color=INK2, fontsize=12)
h1 = axes[0].scatter([], [], s=50, color=INK, label="Y (brightness) sample: one per pixel")
h2 = axes[0].scatter([], [], s=120, marker="s", facecolor="none", edgecolor=C[1], linewidth=2, label="Cb and Cr (colour) sample")
fig.legend(handles=[h1, h2], loc="lower center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Chroma subsampling: the same 32 pixels, three ways of sampling colour", y=0.985, fontsize=15)
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
fig.savefig("/home/dima/Work/chooseyourcodec/docs/map/zoo/img/perceptual-models/chroma-420.png")
print("ok")
