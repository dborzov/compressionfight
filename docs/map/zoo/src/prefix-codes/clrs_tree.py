"""Figure: the Huffman tree for the CLRS frequencies A45 B13 C12 D16 E9 F5, built bottom-up.
Run: uv run --with matplotlib --with numpy python clrs_tree.py
"""
import os, heapq, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()

BG, FG, MUTED = "#f3eadb", "#2a211b", "#5a4a3d"
PURPLE, GREEN, YELLOW, RED, BLUE = "#6b4a7a", "#6f8f4e", "#b8892e", "#b05a4c", "#246b66"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": MUTED,
                     "grid.color": "#b89f78", "font.size": 16, "font.family": "DejaVu Sans"})
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "img", "prefix-codes", "clrs_tree.png")

counts = {"A": 45, "B": 13, "C": 12, "D": 16, "E": 9, "F": 5}

# Build the tree: smaller weight goes left (0). Ties broken by creation order.
cnt = itertools.count()
heap = [(w, next(cnt), {"sym": s, "w": w}) for s, w in counts.items()]
heapq.heapify(heap)
while len(heap) > 1:
    a = heapq.heappop(heap)[2]
    b = heapq.heappop(heap)[2]
    node = {"w": a["w"] + b["w"], "left": a, "right": b}
    heapq.heappush(heap, (node["w"], next(cnt), node))
root = heap[0][2]

codes = {}


def walk(n, code=""):
    if "sym" in n:
        codes[n["sym"]] = code or "0"
        return
    walk(n["left"], code + "0")
    walk(n["right"], code + "1")


walk(root)
for s in "ABCDEF":
    print(s, counts[s], codes[s], len(codes[s]))
mean = sum(counts[s] * len(codes[s]) for s in counts) / 100
p = np.array(list(counts.values())) / 100
print("mean", mean, "H", float(-(p * np.log2(p)).sum()))

# Layout: leaves get x positions in in-order; internal nodes centred over children; depth gives y.
pos = {}
xcounter = itertools.count()


def layout(n, depth=0):
    if "sym" in n:
        x = next(xcounter) * 1.6
    else:
        xl = layout(n["left"], depth + 1)
        xr = layout(n["right"], depth + 1)
        x = (xl + xr) / 2
    pos[id(n)] = (x, -depth)
    return x


layout(root)

fig, ax = plt.subplots(figsize=(16, 8), dpi=100)


def draw(n):
    x, y = pos[id(n)]
    if "sym" in n:
        ax.add_patch(plt.Rectangle((x - 0.55, y - 0.3), 1.1, 0.6, color=PURPLE, zorder=3))
        ax.text(x, y, f'{n["sym"]}:{n["w"]}', ha="center", va="center", fontsize=17, color=BG, weight="bold", zorder=4)
        ax.text(x, y - 0.55, codes[n["sym"]], ha="center", va="top", fontsize=16, color=YELLOW, family="monospace", zorder=4)
        return
    for child, bit in ((n["left"], "0"), (n["right"], "1")):
        cx, cy = pos[id(child)]
        ax.plot([x, cx], [y, cy], color=MUTED, lw=2, zorder=1)
        ax.text((x + cx) / 2 + (-0.18 if bit == "0" else 0.18), (y + cy) / 2 + 0.08, bit, color=GREEN, fontsize=15, ha="center", zorder=5)
        draw(child)
    ax.add_patch(plt.Circle((x, y), 0.36, color="#e0d2b6", ec=MUTED, lw=2, zorder=3))
    ax.text(x, y, str(n["w"]), ha="center", va="center", fontsize=15, color=FG, zorder=4)


draw(root)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(f"Huffman tree for A45 B13 C12 D16 E9 F5: mean length {mean:.2f} bits, entropy {float(-(p*np.log2(p)).sum()):.2f} bits",
             fontsize=19, pad=10)
ax.text(0.5, 0.02, "merge the two smallest weights, repeat; leaf depth = code length; the Kraft sum  1/2 + 3/8 + 2/16 = 1",
        transform=ax.transAxes, ha="center", fontsize=15, color=MUTED)
ax.set_xlim(-1.2, 9.2)
ax.set_ylim(-5.1, 0.6)
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
