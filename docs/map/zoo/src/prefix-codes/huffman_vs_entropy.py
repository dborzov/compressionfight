"""Figure: Huffman mean code length vs entropy as P(A) sweeps 0.1 -> 0.95 on the CLRS alphabet
(the other five symbols share 1 - P(A) in their CLRS proportions, as in the game's Woodcutter lab).
Run: uv run --with matplotlib --with numpy python huffman_vs_entropy.py
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
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "img", "prefix-codes", "huffman_vs_entropy.png")


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


others = np.array([13, 12, 16, 9, 5]) / 55.0
pA = np.linspace(0.1, 0.95, 171)
L, Hs = [], []
for a in pA:
    p = np.concatenate([[a], (1 - a) * others])
    L.append(float(p @ huffman_lengths(p)))
    Hs.append(float(-(p * np.log2(p)).sum()))
L, Hs = np.array(L), np.array(Hs)
for a in (0.45, 0.5, 0.7, 0.9, 0.95):
    i = np.argmin(abs(pA - a))
    print(f"P(A)={pA[i]:.2f}  Huffman={L[i]:.3f}  H={Hs[i]:.3f}  gap={L[i]-Hs[i]:.3f}")

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=100, gridspec_kw={"width_ratios": [1.6, 1]})
ax.plot(pA, L, color=PURPLE, lw=3, label="Huffman mean length L")
ax.plot(pA, Hs, color=GREEN, lw=3, label="entropy H (the floor)")
ax.plot(pA, Hs + 1, color=MUTED, lw=1.5, ls="--", label="H + 1 (Shannon's guarantee)")
ax.plot(pA, Hs + pA + 0.086, color=YELLOW, lw=1.5, ls=":", label=r"H + $p_{max}$ + 0.086 (Gallager)")
ax.fill_between(pA, Hs, L, color=RED, alpha=0.25, label="the whole-bit tax")
ax.axvline(0.5, color=MUTED, lw=1, alpha=0.6)
ax.text(0.515, 0.78, "P(A) > 0.5: A still costs a full bit", color=RED, fontsize=14)
ax.axhline(1.0, color=RED, lw=1, ls=":", alpha=0.6)
ax.set_xlabel("P(A), the most common symbol")
ax.set_ylabel("bits per symbol")
ax.set_ylim(0, 3.7)
ax.set_title("CLRS alphabet as A takes over", fontsize=18, pad=12)
ax.legend(frameon=False, fontsize=13, loc="lower left")
ax.grid(True, alpha=0.5)

ax2.plot(pA, L / Hs, color=RED, lw=3)
ax2.set_xlabel("P(A)")
ax2.set_ylabel("L / H  (bytes you pay per byte you needed)")
ax2.set_title("Relative overhead", fontsize=18, pad=12)
ax2.grid(True, alpha=0.5)
i = np.argmin(abs(pA - 0.9))
ax2.annotate(f"P(A)=0.9: {L[i]/Hs[i]:.2f}×", (pA[i], L[i] / Hs[i]), xytext=(-150, 10), textcoords="offset points", color=FG, fontsize=14,
             arrowprops=dict(arrowstyle="->", color=MUTED))
for a_ in (ax, ax2):
    for s in ("top", "right"):
        a_.spines[s].set_visible(False)
fig.suptitle("Huffman is optimal among whole-bit codes, and whole bits are the problem", fontsize=20, y=0.995)
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
