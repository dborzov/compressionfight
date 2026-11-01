"""Figure: LZ77 matches highlighted on the toy text."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()
from matplotlib.patches import FancyArrowPatch, Rectangle
import os

BG, INK, MUTED, GRID = "#f3eadb", "#2a211b", "#5a4a3d", "#b89f78"
C = ["#246b66", "#b8892e", "#b05a4c", "#6b4a7a"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK,
                     "font.family": "DejaVu Sans Mono", "font.size": 13})

s = "the_cat_sat_on_the_mat_the_cat_sat"
# tokens from the toy parser: (distance, length) at start positions
matches = [(9, 4, 3), (15, 15, 4), (20, 11, 3), (23, 23, 11)]  # (start, distance, length)

fig, ax = plt.subplots(figsize=(16, 5.2), dpi=100)
ax.set_xlim(-0.5, len(s) + 0.5); ax.set_ylim(-2.3, 3.2); ax.axis("off")
for i, ch in enumerate(s):
    ax.text(i + 0.5, 0, ch, ha="center", va="center", fontsize=20, color=INK)
    ax.text(i + 0.5, -0.55, str(i), ha="center", va="center", fontsize=9, color=MUTED)
for k, (st, d, L) in enumerate(matches):
    col = C[k % 4]
    # the copy (target) and its source
    ax.add_patch(Rectangle((st, -0.3), L, 0.6, fill=False, ec=col, lw=2.2))
    ax.add_patch(Rectangle((st - d, -0.3), L, 0.6, fill=True, fc=col, alpha=0.18, ec="none"))
    arrow = FancyArrowPatch((st + L / 2, 0.34), (st - d + L / 2, 0.34),
                            connectionstyle=f"arc3,rad={0.25 + 0.06 * k}", arrowstyle="-|>",
                            mutation_scale=14, color=col, lw=1.8)
    ax.add_patch(arrow)
    ax.text(st + L / 2, -1.0 - 0.38 * (k % 2), f"({d},{L})", ha="center", va="center", fontsize=13, color=col)
ax.text(len(s) / 2, 2.9, "13 literals + 4 back-references cover 34 bytes; the shaded boxes are where each copy comes from",
        ha="center", va="center", fontsize=13, color=MUTED)
ax.text(len(s) / 2, -2.0, "token = (distance back, length): 'go back d bytes and copy L'", ha="center", fontsize=12, color=MUTED)
out = os.path.join(os.path.dirname(__file__), "..", "..", "img", "lz-family", "matches.png")
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
from PIL import Image
w = Image.open(out).width                      # re-save so the trimmed image is exactly 1600 px wide
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3, dpi=100 * 1600 / w)
print("wrote", out)
