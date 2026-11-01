"""Figure: the PPM escape ladder for one symbol of the toy text `the_theme_of_the_thesis`: coding the first
's' (position 20, after 'the'), which no context has ever seen. PPMC estimates, no exclusion (for readability),
uniform fallback over the 27-symbol alphabet. All numbers are computed below.
Run: uv run --with matplotlib --with numpy python escape_ladder.py
"""
import os, math
from collections import Counter
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "img", "context-models", "escape_ladder.png")

text = "the_theme_of_the_thesis"
A = 27
pos = 20
hist, sym = text[:pos], text[pos]
assert sym == "s" and hist.endswith("the")


def counts_after(hist, ctx):
    c = Counter()
    for i in range(len(ctx), len(hist)):
        if hist[i - len(ctx):i] == ctx:
            c[hist[i]] += 1
    return c


rungs = []
total = 0.0
for o in (3, 2, 1, 0):
    ctx = hist[len(hist) - o:] if o else ""
    cnt = counts_after(hist, ctx) if o else Counter(hist)
    n, u = sum(cnt.values()), len(cnt)
    seen = " ".join(f"{k if k != '_' else '␣'}:{v}" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]))
    if sym in cnt:
        p = cnt[sym] / (n + u)
        rungs.append((o, ctx, f"hit  {cnt[sym]}/{n + u}", -math.log2(p), seen))
        total += -math.log2(p)
        break
    p = u / (n + u)
    rungs.append((o, ctx, f"escape  u/(n+u) = {u}/{n + u}", -math.log2(p), seen))
    total += -math.log2(p)
else:
    rungs.append((-1, "", f"uniform  1/{A}", math.log2(A), "every symbol never seen"))
    total += math.log2(A)
for r in rungs:
    print(r)
print("total bits for 's':", total)

fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
n = len(rungs)
for i, (o, ctx, what, bits, seen) in enumerate(rungs):
    y = -i * 1.4
    col = GREEN if what.startswith("hit") else (YELLOW if o >= 0 else RED)
    label = f"order {o}" + (f'  after "{ctx.replace("_", "␣")}"' if o > 0 else ("  (no context)" if o == 0 else "  (fallback)"))
    ax.add_patch(plt.Rectangle((0, y - 0.45), 5.0, 0.9, color="#e0d2b6", ec=MUTED, lw=1.5))
    ax.text(0.15, y, label, va="center", fontsize=16, color=FG)
    ax.text(5.3, y + 0.18, what, va="center", fontsize=15, color=col)
    ax.text(5.3, y - 0.22, f"seen here: {seen}", va="center", fontsize=13, color=MUTED)
    ax.add_patch(plt.Rectangle((12.4, y - 0.35), bits * 0.5, 0.7, color=col, alpha=0.9))
    ax.text(12.4 + bits * 0.5 + 0.15, y, f"{bits:.2f} bits", va="center", fontsize=15, color=col)
    if i < n - 1:
        ax.annotate("", (2.5, y - 1.4 + 0.5), (2.5, y - 0.5), arrowprops=dict(arrowstyle="->", color=YELLOW, lw=2))
        ax.text(2.7, y - 0.72, "'s' unseen here → escape", fontsize=13, color=YELLOW, va="center")
ax.text(5.3, -n * 1.4 + 0.25, f"total paid for this 's':  {total:.2f} bits", fontsize=16, color=FG, ha="left")
ax.text(5.3, -n * 1.4 - 0.2, "(a raw byte is 8 bits; a symbol its order-3 context has seen before usually costs under 1)",
        fontsize=13, color=MUTED, ha="left")
ax.set_xlim(-0.2, 16.2)
ax.set_ylim(-n * 1.4 - 0.5, 0.9)
ax.axis("off")
fig.suptitle('Escape ladder: coding the first "s" of "thesis" (after "the") in  the␣theme␣of␣the␣thesis', fontsize=18, y=0.98)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT)
print("wrote", OUT)
