"""Figure: bits per symbol for the three weather stations under four charging schemes.
Run: uv run --with matplotlib --with numpy python three_stations.py
"""
import os, heapq
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
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "img", "modeling-and-coding", "three_stations.png")


def H(p):
    p = np.asarray(p, float)
    return float(-(p * np.log2(p)).sum())


def huffman_lengths(probs):
    heap = [(p, i, (i,)) for i, p in enumerate(probs)]
    heapq.heapify(heap)
    lens = [0] * len(probs)
    n = len(probs)
    while len(heap) > 1:
        a, b = heapq.heappop(heap), heapq.heappop(heap)
        for i in a[2] + b[2]:
            lens[i] += 1
        heapq.heappush(heap, (a[0] + b[0], n, a[2] + b[2]))
        n += 1
    return np.array(lens)


stations = {
    "Station 1\n(0.5, 0.25, 0.125, 0.125)": [0.5, 0.25, 0.125, 0.125],
    "Station 2\n(0.9, 0.05, 0.03, 0.02)": [0.9, 0.05, 0.03, 0.02],
    "Station 3\n(0.25, 0.5, 0.125, 0.125)": [0.25, 0.5, 0.125, 0.125],
}
q1 = np.array([0.5, 0.25, 0.125, 0.125])
rows = []
for name, p in stations.items():
    p = np.array(p)
    fixed = 2.0
    ent = H(p)
    huff = float(p @ huffman_lengths(p))
    cross = float(-(p * np.log2(q1)).sum())
    rows.append((name, fixed, ent, huff, cross))
    print(f"{name.splitlines()[0]}: fixed={fixed:.3f}  H={ent:.3f}  best prefix code={huff:.3f}  under station-1 model={cross:.3f}")

labels = ["fixed 2-bit code", "entropy H(P) (the floor)", "best prefix code (Huffman)", "charged with station 1's model: H(P,Q)"]
colors = [MUTED, GREEN, PURPLE, RED]
fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
x = np.arange(len(rows))
w = 0.2
for k in range(4):
    vals = [r[k + 1] for r in rows]
    bars = ax.bar(x + (k - 1.5) * w, vals, w, color=colors[k], label=labels[k], edgecolor=BG)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=13, color=FG)
ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in rows], fontsize=15)
ax.set_ylabel("bits per hourly report")
ax.set_ylim(0, 2.6)
ax.set_title("Three stations, one coder: the floor belongs to the probabilities", fontsize=20, pad=16)
ax.legend(frameon=False, fontsize=14, loc="upper left", ncol=2)
ax.grid(True, axis="y", alpha=0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
