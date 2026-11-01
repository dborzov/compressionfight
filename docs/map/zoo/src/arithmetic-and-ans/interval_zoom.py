"""Figure: arithmetic coding of ABAC with P(A)=.5, P(B)=P(C)=.25 as a sequence of interval zooms.
Run: uv run --with matplotlib --with numpy python interval_zoom.py
"""
import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, MUTED = "#0b0714", "#e8e2f0", "#8b83a0"
PURPLE, GREEN, YELLOW, RED, BLUE = "#b38cff", "#5ee8a0", "#ffd166", "#ff6b8a", "#6ec3ff"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": MUTED,
                     "grid.color": "#2a2140", "font.size": 16, "font.family": "DejaVu Sans"})
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "img", "arithmetic-and-ans", "interval_zoom.png")

model = [("A", 0.5, PURPLE), ("B", 0.25, GREEN), ("C", 0.25, YELLOW)]
msg = "ABAC"
lo, hi = 0.0, 1.0
steps = [(lo, hi, None)]
for s in msg:
    w = hi - lo
    c = 0.0
    for sym, p, _ in model:
        if sym == s:
            lo, hi = lo + w * c, lo + w * (c + p)
            break
        c += p
    steps.append((lo, hi, s))
for st in steps:
    print(st)
print("width", hi - lo, "bits", -math.log2(hi - lo))

fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
n = len(steps)
for row, (lo, hi, s) in enumerate(steps):
    y = -row * 1.5
    w = hi - lo
    # draw the current interval stretched to the full width; sub-intervals of the model inside it
    c = 0.0
    for sym, p, col in model:
        ax.add_patch(plt.Rectangle((c * 10, y - 0.4), p * 10, 0.8, color=col, alpha=0.9 if (row < n - 1 and msg[row] == sym) else 0.35, ec=BG))
        ax.text((c + p / 2) * 10, y, f"{sym}  p={p:g}", ha="center", va="center", fontsize=15, color=BG, weight="bold")
        c += p
    ax.text(-0.3, y, f"[{lo:.6g}, {hi:.6g})", ha="right", va="center", fontsize=15, family="monospace", color=FG)
    ax.text(10.3, y, f"width {w:g} = 1/{round(1/w)}" if row else "width 1", ha="left", va="center", fontsize=15, color=MUTED)
    if row < n - 1:
        # zoom lines from the chosen sub-interval to the next full-width row
        c = 0.0
        for sym, p, col in model:
            if sym == msg[row]:
                ax.plot([c * 10, 0], [y - 0.4, y - 1.1], color=col, lw=1.5, ls="--", alpha=0.8)
                ax.plot([(c + p) * 10, 10], [y - 0.4, y - 1.1], color=col, lw=1.5, ls="--", alpha=0.8)
                ax.text(5, y - 0.78, f'code "{msg[row]}": keep its slice, pay {-math.log2(p):g} bit{"s" if p < 0.5 else ""}, zoom in',
                        ha="center", va="center", fontsize=14, color=col)
            c += p
ax.set_xlim(-4.2, 13.5)
ax.set_ylim(-(n - 1) * 1.5 - 1.1, 0.9)
ax.axis("off")
ax.set_title("ABAC as one number: the final interval has width 1/64, so any point inside it needs 6 bits = 1 + 2 + 1 + 2",
             fontsize=18, pad=10)
ax.text(5, -(n - 1) * 1.5 - 0.9, "0.296875 = 0.010011₂ lies inside [0.296875, 0.3125): the message is the six bits 010011",
        ha="center", fontsize=15, color=FG, family="monospace")
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
