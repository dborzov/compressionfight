"""Stacked latency ledger for a two-way call against the 150 ms mouth-to-ear budget, with what an
MP3-style frame would do to it.  Run: uv run --with matplotlib --with numpy python latency_budget.py"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()

BG, FG, DIM = "#f3eadb", "#2a211b", "#5a4a3d"
ORANGE, PURPLE, CYAN, GREEN, YELLOW, GOLD = "#b05a4c", "#6b4a7a", "#246b66", "#6f8f4e", "#b8892e", "#9c6b3f"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.edgecolor": DIM, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                     "text.color": FG, "font.size": 14.5, "axes.titlesize": 17, "axes.spines.top": False,
                     "axes.spines.right": False, "grid.color": "#b89f78", "grid.linestyle": ":"})
fs = 44_100
# (label, colour, VoIP-style 20 ms codec, MP3-style 1152-sample codec with lookahead and overlap)
rows = [("capture buffer: one frame", CYAN, 20, 1152 / fs * 1e3),
        ("encoder algorithmic: lookahead + window overlap", ORANGE, 6.5, 1152 / fs * 1e3 + 1152 / fs * 1e3),
        ("encode compute", "#b05a4c", 3, 3),
        ("network propagation (one way)", PURPLE, 40, 40),
        ("jitter buffer", "#6b4a7a", 40, 40),
        ("decode compute", "#6f8f4e", 2, 2),
        ("playout buffer: one frame", GREEN, 20, 1152 / fs * 1e3)]
names = ["20 ms frames, 6.5 ms lookahead\n(an Opus-class design)", "1152-sample frames + one frame of overlap\n+ one frame of lookahead (an MP3-class design)"]
fig, ax = plt.subplots(figsize=(16, 7.4), dpi=100)
left = np.zeros(2)
for label, color, a, b in rows:
    vals = np.array([a, b])
    ax.barh([0, 1], vals, left=left, color=color, edgecolor=BG, height=0.55, label=label)
    for i, v in enumerate(vals):
        if v >= 6: ax.text(left[i] + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=12.5, color="#f3eadb", fontweight="bold")
    left += vals
for i, t in enumerate(left):
    ax.text(t + 3, i, f"{t:.0f} ms", va="center", fontsize=15, color=FG, fontweight="bold")
ax.axvline(150, color=YELLOW, lw=3, ls="--")
ax.text(153, 0.5, "150 ms: the mouth-to-ear limit\nbefore a conversation starts to\nfeel like a walkie-talkie (ITU-T G.114)", color=YELLOW, fontsize=12.5, va="center")
ax.set_yticks([0, 1]); ax.set_yticklabels(names, fontsize=12.5); ax.set_ylim(-0.5, 1.5)
ax.set_xlim(0, 245); ax.set_xlabel("milliseconds, one direction")
fig.suptitle("The latency ledger of a call: network and jitter buffer are fixed costs; the codec's frame length is paid three times", fontsize=15.5, y=0.97)
fig.legend(loc="lower center", frameon=False, fontsize=11.5, ncol=4)
ax.grid(True, axis="x")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "img", "anatomy-of-a-job", "latency_budget.png")
fig.subplots_adjust(left=0.27, right=0.97, top=0.9, bottom=0.24); fig.savefig(out); print("wrote", os.path.abspath(out))
