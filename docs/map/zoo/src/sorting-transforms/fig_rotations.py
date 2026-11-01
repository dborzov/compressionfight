"""Figure: the sorted rotation table of banana$ and abracadabra$, with the first and last columns highlighted."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()
from matplotlib.patches import Rectangle
import os

BG, INK, MUTED, GRID = "#f3eadb", "#2a211b", "#5a4a3d", "#b89f78"
C = ["#246b66", "#b8892e", "#b05a4c", "#6b4a7a"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK, "font.family": "DejaVu Sans Mono", "font.size": 13})

def table(ax, s, title):
    rots = sorted(s[i:] + s[:i] for i in range(len(s)))
    idx = rots.index(s); n = len(s)
    ax.set_xlim(-1.2, n + 0.6); ax.set_ylim(-n - 1.8, 1.6); ax.axis("off")
    ax.add_patch(Rectangle((-0.5, -n - 0.4), 1, n + 0.8, fc=C[0], alpha=0.18, ec="none"))
    ax.add_patch(Rectangle((n - 1.5, -n - 0.4), 1, n + 0.8, fc=C[1], alpha=0.25, ec="none"))
    ax.text(0, 0.9, "F", ha="center", color=C[0], fontsize=15); ax.text(n - 1, 0.9, "L", ha="center", color=C[1], fontsize=15)
    for r, rot in enumerate(rots):
        for j, ch in enumerate(rot):
            col = INK if j in (0, n - 1) else MUTED
            ax.text(j, -r - 0.5, ch, ha="center", va="center", fontsize=17, color=col)
        ax.text(-1.0, -r - 0.5, str(r), ha="center", va="center", fontsize=11, color=MUTED)
        if r == idx:
            ax.add_patch(Rectangle((-0.5, -r - 1), n, 1, fill=False, ec=C[2], lw=2))
    ax.text(-0.5, -n - 0.9, f"row {idx} is the original: the primary index", color=C[2], fontsize=12, va="top")
    L = "".join(r[-1] for r in rots)
    ax.set_title(f"{title}\nsorted rotations → last column L = {L}", loc="left", fontsize=13, color=INK, pad=10)

fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), dpi=100, gridspec_kw={"width_ratios": [1, 1.5]})
table(axes[0], "banana$", "banana$")
table(axes[1], "abracadabra$", "abracadabra$")
out = os.path.join(os.path.dirname(__file__), "..", "..", "img", "sorting-transforms", "rotations.png")
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
from PIL import Image
w = Image.open(out).width                      # re-save so the trimmed image is exactly 1600 px wide
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3, dpi=100 * 1600 / w)
print("wrote", out)
