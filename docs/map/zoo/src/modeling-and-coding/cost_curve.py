"""Figure: the price of a symbol, -log2 p, as a function of p, with the two weather stations' symbols marked.
Run: uv run --with matplotlib --with numpy python cost_curve.py
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

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "img", "modeling-and-coding", "cost_curve.png")

p = np.linspace(0.005, 1, 400)
fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
ax.plot(p, -np.log2(p), color=PURPLE, lw=3, label=r"price of a symbol: $-\log_2 p$")

station1 = {"SUN": 0.5, "RAIN": 0.25, "SNOW": 0.125, "HAIL": 0.125}
station2 = {"SUN": 0.9, "RAIN": 0.05, "SNOW": 0.03, "HAIL": 0.02}
for name, pr in station1.items():
    c = -np.log2(pr)
    ax.vlines(pr, 0, c, color=GREEN, lw=2, alpha=0.8)
    ax.plot(pr, c, "o", color=GREEN, ms=11)
    if name == "HAIL":
        continue  # same probability as SNOW: one shared label
    label = "SNOW, HAIL 0.125 → 3 bits" if name == "SNOW" else f"{name} {pr:g} → {c:.0f} bit{'s' if c > 1 else ''}"
    ax.annotate(label, (pr, c), xytext=(10, 8), textcoords="offset points", color=GREEN, fontsize=14)
for name, pr in station2.items():
    c = -np.log2(pr)
    ax.vlines(pr, 0, c, color=YELLOW, lw=2, alpha=0.8, ls="--")
    ax.plot(pr, c, "s", color=YELLOW, ms=10)
    dy = {"SUN": 14, "RAIN": 8, "SNOW": 8, "HAIL": 8}[name]
    dx = {"SUN": -120, "RAIN": 10, "SNOW": 10, "HAIL": 10}[name]
    ax.annotate(f"{name} {pr:g} → {c:.2f} bits", (pr, c), xytext=(dx, dy), textcoords="offset points", color=YELLOW, fontsize=14)

ax.axhline(2, color=MUTED, lw=1.5, ls=":")
ax.text(0.62, 2.1, "fixed 2-bit code pays this for every symbol", color=MUTED, fontsize=14)
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 7)
ax.set_xlabel("probability p the model gave the symbol that actually occurred")
ax.set_ylabel("bits charged")
ax.set_title("The price of a symbol is how surprising it was", color=FG, fontsize=20, pad=16)
ax.grid(True, alpha=0.5)
ax.plot([], [], "o", color=GREEN, label="station 1: p = (0.5, 0.25, 0.125, 0.125)")
ax.plot([], [], "s", color=YELLOW, label="station 2: p = (0.9, 0.05, 0.03, 0.02)")
ax.legend(frameon=False, loc="upper right", fontsize=15)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
