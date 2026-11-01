"""Figure: bits paid per symbol as a function of its probability, Huffman (whole bits, two-symbol alphabet)
vs arithmetic coding (-log2 p); right panel: average bits/symbol on a binary source as P(A) varies.
Run: uv run --with matplotlib --with numpy python bits_vs_p.py
"""
import os
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
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "img", "arithmetic-and-ans", "bits_vs_p.png")

p = np.linspace(0.5, 0.999, 500)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

# left: the cost of the likely symbol itself
ax.plot(p, -np.log2(p), color=GREEN, lw=3, label=r"arithmetic / ANS: $-\log_2 p$")
ax.plot(p, np.ones_like(p), color=PURPLE, lw=3, label="any prefix code: 1 bit minimum")
ax.fill_between(p, -np.log2(p), 1, color=RED, alpha=0.25, label="paid for nothing")
for q, off in ((0.9, (-150, -40)), (0.98, (-200, 45)), (0.99, (-120, 110))):
    c = -np.log2(q)
    ax.plot(q, c, "o", color=GREEN, ms=10)
    ax.annotate(f"p={q}: {c:.3f} bits", (q, c), xytext=off, textcoords="offset points",
                color=FG, fontsize=13, arrowprops=dict(arrowstyle="->", color=MUTED))
ax.set_xlabel("probability p of the symbol being coded")
ax.set_ylabel("bits charged for that symbol")
ax.set_ylim(0, 1.15)
ax.set_title("One likely symbol", fontsize=18, pad=12)
ax.legend(frameon=False, fontsize=13, loc="lower left")
ax.grid(True, alpha=0.5)

# right: average on a binary source
pa = np.linspace(0.5, 0.999, 500)
h = -(pa * np.log2(pa) + (1 - pa) * np.log2(1 - pa))
ax2.plot(pa, h, color=GREEN, lw=3, label="arithmetic / ANS: H(p)")
ax2.plot(pa, np.ones_like(pa), color=PURPLE, lw=3, label="Huffman on {A, B}: 1 bit/symbol")
ax2.fill_between(pa, h, 1, color=RED, alpha=0.25)
for q in (0.9, 0.99):
    i = np.argmin(abs(pa - q))
    ax2.plot(q, h[i], "o", color=GREEN, ms=10)
    ax2.annotate(f"P(A)={q}: H={h[i]:.3f}, Huffman pays {1/h[i]:.1f}× more", (q, h[i]), xytext=(0.52, 0.62) if q < 0.95 else (0.52, 0.30),
                 textcoords="data", color=FG, fontsize=13, arrowprops=dict(arrowstyle="->", color=MUTED))
    print(f"P(A)={q}: H={h[i]:.4f}, ratio {1/h[i]:.2f}")
ax2.set_xlabel("P(A) on a two-symbol source")
ax2.set_ylabel("average bits per symbol")
ax2.set_ylim(0, 1.15)
ax2.set_title("Averaged over a skewed binary source", fontsize=18, pad=12)
ax2.legend(frameon=False, fontsize=13, loc="lower left")
ax2.grid(True, alpha=0.5)
for a_ in (ax, ax2):
    for s in ("top", "right"):
        a_.spines[s].set_visible(False)
fig.suptitle("Fractional bits: what the whole-bit floor costs when one symbol dominates", fontsize=20, y=0.995)
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
