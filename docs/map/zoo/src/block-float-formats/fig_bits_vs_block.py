"""Figure: bits per weight vs block size for b = 2..8 with a 16-bit scale, plus the
points where real formats sit.  Output: img/block-float-formats/bits-vs-block.png"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()

BG, INK, INK2, MUTED, GRID = "#f3eadb", "#2a211b", "#5a4a3d", "#5a4a3d", "#b89f78"
C = ["#246b66", "#b8892e", "#b05a4c", "#6b4a7a", "#6f8f4e", "#9c6b3f", "#a7403a"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 13, "axes.titlesize": 15})

n = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])
fig, ax = plt.subplots(figsize=(16, 7.6), dpi=100)
for k, b in enumerate(range(2, 9)):
    y = (16 + n * b) / n
    ax.plot(n, y, color=C[k], linewidth=2, marker="o", markersize=5)
    ax.text(1024 * 1.12, b + 16 / 1024, f"b = {b}", color=C[k], va="center", fontsize=12)
    ax.axhline(b, color=C[k], linewidth=0.8, linestyle=":", alpha=0.6)

pts = [("Q4_0: fp16 scale / 32", 32, 4.5, C[2]), ("NVFP4: fp8 scale / 16", 16, 4.5, C[2]),
       ("Q8_0: fp16 scale / 32", 32, 8.5, C[6]), ("MXFP4: 8-bit exponent / 32", 32, 4.25, C[2]),
       ("Q4_1: scale + min / 32", 32, 5.0, C[2]), ("KIVI int2: scale + zero / 32", 32, 3.0, C[0])]
for label, bx, by, col in pts:
    ax.scatter(bx, by, s=170, marker="D", color=col, edgecolor=INK, linewidth=1.2, zorder=5)
for label, bx, by, col, dx, dy in [(pts[0][0], 32, 4.5, C[2], 14, 8), (pts[1][0], 16, 4.5, C[2], -14, 14),
                                   (pts[2][0], 32, 8.5, C[6], 14, 6), (pts[3][0], 32, 4.25, C[2], 14, -14),
                                   (pts[4][0], 32, 5.0, C[2], 14, 8), (pts[5][0], 32, 3.0, C[0], 14, -14)]:
    ax.annotate(label, (bx, by), xytext=(dx, dy), textcoords="offset points", color=INK, fontsize=11,
                ha="left" if dx > 0 else "right")
ax.set_xscale("log", base=2)
ax.set_xticks(n); ax.set_xticklabels([str(v) for v in n])
ax.set_xlim(3.4, 1500); ax.set_ylim(1.5, 13)
ax.set_xlabel("block size n (values sharing one scale)")
ax.set_ylabel("bits per weight = (16 + n·b) / n")
ax.set_title("Every block format is a point on these curves: the scale is amortised over the block, and the nominal width b is a floor")
ax.grid(True, alpha=0.35)
fig.tight_layout()
fig.savefig("/home/dima/Work/chooseyourcodec/docs/map/zoo/img/block-float-formats/bits-vs-block.png")
print("ok")
