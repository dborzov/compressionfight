"""The JPEG luma quantization table (Annex K, quality 50) as a heat matrix, beside the Wikipedia
example block's DCT coefficients and the quantized result (44 zeros of 64).
Run: uv run --with matplotlib --with numpy python jpeg_table.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()
from matplotlib.colors import LinearSegmentedColormap

BG, FG, DIM = "#f3eadb", "#2a211b", "#5a4a3d"
ORANGE, PURPLE, CYAN, GREEN, YELLOW = "#b05a4c", "#6b4a7a", "#246b66", "#6f8f4e", "#b8892e"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.edgecolor": DIM, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                     "text.color": FG, "font.size": 15, "axes.titlesize": 16.5})
block = np.array([[52,55,61,66,70,61,64,73],[63,59,55,90,109,85,69,72],[62,59,68,113,144,104,66,73],
                  [63,58,71,122,154,106,70,69],[67,61,68,104,126,88,68,70],[79,65,60,70,77,68,58,75],
                  [85,71,64,59,55,61,65,83],[87,79,69,68,65,76,78,94]], float)
Q = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56],
              [14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
              [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])
n = np.arange(8); C = np.cos((2 * n[None, :] + 1) * n[:, None] * np.pi / 16) * 0.5; C[0] /= np.sqrt(2)
coef = C @ (block - 128) @ C.T
q = np.round(coef / Q).astype(int)

heat = LinearSegmentedColormap.from_list("heat", ["#f3eadb", "#9cc4b8", "#246b66"])
fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), dpi=100, gridspec_kw={"wspace": 0.08})
plt.rcParams["axes.titlesize"] = 14.5
def grid(ax, M, title, cmap, vals, fmt, zero_dim=False):
    ax.imshow(M, cmap=cmap); ax.set_title(title, fontsize=14.5)
    for r in range(8):
        for c in range(8):
            v = vals[r, c]; z = zero_dim and v == 0
            ax.text(c, r, fmt(v), ha="center", va="center", fontsize=11.5 if not z else 10,
                    color=(DIM if z else ("#f3eadb" if (cmap is heat and M[r, c] > 0.55 * M.max()) else FG)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor(DIM)
grid(axes[0], Q, "quantization table Δ(u, v) at quality 50:\ncoarse where the eye is blind (bottom right)", heat, Q, lambda v: f"{v}")
grid(axes[1], np.log10(np.abs(coef) + 1), "DCT coefficients of the toy block:\nenergy piles up top left", heat, coef, lambda v: f"{v:.0f}")
zero_map = (q == 0).astype(float)
grid(axes[2], 1 - zero_map, f"q = round(coef / Δ): {np.sum(q == 0)} of 64 are zero,\nwhat the run-length coder eats",
     LinearSegmentedColormap.from_list("z", ["#ebdfc8", "#ebdfc8"]), q, lambda v: f"{v}", zero_dim=True)
fig.text(0.5, 0.035, "Horizontal frequency grows to the right, vertical frequency downward; the top-left cell is the DC term.",
         ha="center", color=DIM, fontsize=12.5)
out = os.path.join(os.path.dirname(__file__), "..", "..", "img", "scalar-quantization", "jpeg_table.png")
fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.09, wspace=0.08)
fig.savefig(out); print("wrote", os.path.abspath(out))
