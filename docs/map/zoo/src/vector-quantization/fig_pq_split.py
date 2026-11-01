"""Figure: product quantization split diagram for x = [0.9, 0.1, -0.3, 0.8].
Output: img/vector-quantization/pq-split.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG, INK, INK2, MUTED, GRID = "#f3eadb", "#2a211b", "#5a4a3d", "#5a4a3d", "#b89f78"
C = ["#246b66", "#b8892e", "#b05a4c", "#6b4a7a", "#6f8f4e"]
DIM = "#e0d2b6"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "font.size": 13})

x = [0.9, 0.1, -0.3, 0.8]
cb1 = [[0.8, 0.2], [-0.7, 0.1], [0.1, 0.9], [0.0, -0.8]]
cb2 = [[0.5, 0.5], [-0.4, 0.7], [0.6, -0.6], [-0.7, -0.5]]
i, j = 0, 1

fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

def box(x0, y0, w, h, text, fc, ec=None, fs=13, weight="normal", tc=INK):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=fc, edgecolor=ec or fc, linewidth=1.5))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight=weight)

def arrow(p, q, color=INK2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, color=color, linewidth=1.6))

# row A: the input vector
ax.text(8.0, 8.75, "x = 4 floats = 128 bits", ha="center", color=INK2, fontsize=13)
for k, v in enumerate(x):
    box(5.8 + k * 1.1, 7.75, 1.0, 0.8, f"{v:+.1f}", C[0] if k < 2 else C[1], fs=15, weight="bold")
ax.text(6.85, 7.42, "half 1", ha="center", color=C[0]); ax.text(9.05, 7.42, "half 2", ha="center", color=C[1])

# row B: two codebooks as rows of four entries
def codebook(x0, entries, col, title, pick):
    ax.text(x0 + 3.5, 6.95, title, ha="center", color=col, fontsize=13, fontweight="bold")
    centers = []
    for k, e in enumerate(entries):
        fc = col if k == pick else DIM
        box(x0 + k * 1.78, 5.95, 1.66, 0.75, f"{k:02b} ({e[0]:+.1f}, {e[1]:+.1f})", fc, ec=col, fs=11,
            weight="bold" if k == pick else "normal")
        centers.append(x0 + k * 1.78 + 0.83)
    return centers
c1 = codebook(0.6, cb1, C[0], "codebook 1 (dims 0, 1): four 2-D centroids", i)
c2 = codebook(8.4, cb2, C[1], "codebook 2 (dims 2, 3): four 2-D centroids", j)
arrow((6.85, 7.3), (4.3, 7.15), C[0]); arrow((9.05, 7.3), (11.7, 7.15), C[1])
ax.text(3.2, 5.6, "nearest to (+0.9, +0.1)", ha="center", color=C[0], fontsize=11)
ax.text(12.0, 5.6, "nearest to (−0.3, +0.8)", ha="center", color=C[1], fontsize=11)

# row C: the code
box(7.0, 3.9, 1.0, 0.8, f"{i:02b}", C[0], fs=17, weight="bold")
box(8.1, 3.9, 1.0, 0.8, f"{j:02b}", C[1], fs=17, weight="bold")
arrow((c1[i], 5.9), (7.45, 4.75), C[0]); arrow((c2[j], 5.9), (8.65, 4.75), C[1])
ax.text(8.05, 3.5, "code (i, j) = (0, 1): 4 bits.  128 bits → 4 bits, 32:1", ha="center", color=INK, fontsize=13)
ax.text(8.05, 3.05, "decoded x̂ = (+0.8, +0.2, −0.4, +0.7),  error (−0.1, +0.1, −0.1, −0.1)", ha="center", color=INK2, fontsize=12)

# row D: reachable points
ax.text(3.2, 2.45, "the 16 vectors the decoder can produce", ha="center", color=INK, fontsize=13, fontweight="bold")
for r in range(4):
    for c in range(4):
        fc = C[2] if (r, c) == (i, j) else DIM
        box(1.6 + c * 0.82, 1.6 - r * 0.5 + 0.0, 0.74, 0.42, f"{r}{c}", fc, ec=C[2], fs=11)
ax.text(10.8, 2.45, "stored: 4 + 4 = 8 centroids.  reachable: 4 × 4 = 16 points.", ha="center", color=INK, fontsize=13, fontweight="bold")
ax.text(10.8, 1.75, "Faiss PQ96x8 on 768-D: 96 codebooks × 256 entries = 24,576 stored centroids,\n"
        "256⁹⁶ ≈ 10²³¹ reachable vectors, 96 bytes per vector (32:1 against float32).", ha="center", color=INK2, fontsize=12)
ax.text(10.8, 0.75, "Each codebook is side information: the decoder owns 8 (or 24,576) vectors before the first index arrives.",
        ha="center", color=MUTED, fontsize=12)

fig.savefig("/home/dima/Work/chooseyourcodec/docs/map/zoo/img/vector-quantization/pq-split.png")
print("ok")
