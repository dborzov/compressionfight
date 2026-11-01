"""Figure: bits per character vs context order on a real text sample (docs/map/primer.md, English prose
about compression), for two estimators: an order-k model alone with Laplace smoothing (no back-off), and a
PPM-style model with escapes to shorter contexts (PPMC escape estimate, exclusion at order -1). Reference lines:
zlib level 9 and xz (lzma preset 9) on the same bytes. Everything is computed here, adaptively, one pass.
Run: uv run --with matplotlib --with numpy python bits_vs_order.py
"""
import os, math, zlib, lzma
from collections import defaultdict
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
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "img", "context-models", "bits_vs_order.png")
SRC = os.path.join(HERE, "..", "..", "..", "primer.md")

data = open(SRC, "rb").read()
N = len(data)
ALPHA = 256
print("sample:", SRC, N, "bytes")


def laplace_order_k(data, k):
    """Adaptive order-k model, Laplace smoothing over 256 symbols, no back-off."""
    tables = defaultdict(lambda: [0] * ALPHA)
    totals = defaultdict(int)
    bits = 0.0
    for i in range(k, len(data)):
        ctx = data[i - k:i]
        s = data[i]
        p = (tables[ctx][s] + 1) / (totals[ctx] + ALPHA)
        bits -= math.log2(p)
        tables[ctx][s] += 1
        totals[ctx] += 1
    return bits / (len(data) - k)


def ppm(data, maxo):
    """PPM with PPMC escapes: P(s|ctx) = n_s/(n+u), P(esc) = u/(n+u); full update of every order."""
    tables = [defaultdict(dict) for _ in range(maxo + 1)]
    bits = 0.0
    for i in range(len(data)):
        s = data[i]
        excluded = set()
        for o in range(min(maxo, i), -1, -1):
            ctx = data[i - o:i]
            t = tables[o].get(ctx)
            if not t:
                continue
            # exclusion: symbols already ruled out at higher orders do not count here
            n = sum(c for sym, c in t.items() if sym not in excluded)
            u = sum(1 for sym in t if sym not in excluded)
            if n == 0:
                continue
            if s in t:
                bits -= math.log2(t[s] / (n + u))
                break
            bits -= math.log2(u / (n + u))
            excluded.update(t.keys())
        else:
            bits += math.log2(ALPHA - len(excluded))
        for o in range(min(maxo, i) + 1):
            ctx = data[i - o:i]
            d = tables[o].setdefault(ctx, {})
            d[s] = d.get(s, 0) + 1
    return bits / len(data)


orders = list(range(0, 7))
lap = [laplace_order_k(data, k) for k in orders]
ppmc = [ppm(data, k) for k in orders]
for k, a, b in zip(orders, lap, ppmc):
    print(f"order {k}: Laplace-only {a:.3f} bits/char   PPM with escapes {b:.3f} bits/char")
gz = 8 * len(zlib.compress(data, 9)) / N
xz = 8 * len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)) / N
print(f"zlib -9: {gz:.3f} bits/char   xz -9e: {xz:.3f} bits/char   raw: 8.000")

fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
ax.plot(orders, lap, "o-", color=RED, lw=3, ms=10, label="order-k alone, Laplace smoothing (no back-off)")
ax.plot(orders, ppmc, "s-", color=GREEN, lw=3, ms=10, label="PPM: longest context first, escape to shorter ones")
ax.axhline(gz, color=MUTED, lw=1.5, ls="--")
ax.text(6.05, gz + 0.06, f"zlib -9: {gz:.2f}", color=MUTED, fontsize=13, ha="right")
ax.axhline(xz, color=BLUE, lw=1.5, ls="--")
ax.text(6.05, xz + 0.06, f"xz -9e: {xz:.2f}", color=BLUE, fontsize=13, ha="right")
for k, v in zip(orders, ppmc):
    ax.text(k, v - 0.22, f"{v:.2f}", ha="center", color=GREEN, fontsize=13)
for k, v in zip(orders, lap):
    ax.text(k, v + 0.12, f"{v:.2f}", ha="center", color=RED, fontsize=13)
ax.set_xlabel("context order k (how many previous characters the model conditions on)")
ax.set_ylabel("bits per character (adaptive, one pass)")
ax.set_xticks(orders)
ax.set_ylim(0, max(lap) + 0.6)
ax.set_title(f"Bits per character vs context order on a {N//1000} KB English sample (docs/map/primer.md)", fontsize=19, pad=14)
ax.legend(frameon=False, fontsize=14, loc="upper left")
ax.grid(True, alpha=0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
