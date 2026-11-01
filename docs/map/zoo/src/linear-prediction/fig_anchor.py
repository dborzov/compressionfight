"""Figure: reconstruction error over a run of 200 'tokens' for open-loop chain, closed-loop chain and anchor prediction."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

BG, INK, MUTED, GRID = "#0b0714", "#e8e4f0", "#9b93aa", "#2a2238"
C = ["#3b8fd6", "#c4812a", "#2fa06e", "#b85fb0"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
                     "font.family": "DejaVu Sans", "font.size": 13})

rng = np.random.default_rng(11)
N, D, G = 200, 4, 10
x = 100 + np.cumsum(rng.integers(-3, 5, N))      # a slowly wandering value, like one KV channel across tokens
q = lambda v: D * np.round(v / D)

# open-loop chain: deltas between ORIGINAL neighbours, quantized; decoder sums them
d = np.diff(x); open_chain = np.concatenate([[x[0]], x[0] + np.cumsum(q(d))])
# closed-loop chain: predict from the previous RECONSTRUCTION
closed = [x[0]]
for v in x[1:]:
    closed.append(closed[-1] + q(v - closed[-1]))
closed = np.array(closed)
# anchor: every G-th value stored exactly, the others as quantized deltas from the anchor
anchor = x.astype(float).copy()
for g in range(0, N, G):
    anchor[g + 1:g + G] = x[g] + q(x[g + 1:g + G] - x[g])

def H(v):
    _, c = np.unique(v, return_counts=True); p = c / c.sum(); return -(p * np.log2(p)).sum()
sym_chain = q(d) / D; sym_anchor = np.concatenate([q(x[g + 1:g + G] - x[g]) / D for g in range(0, N, G)])
print("max |err| open", np.abs(open_chain - x).max(), "closed", np.abs(closed - x).max(), "anchor", np.abs(anchor - x).max())
print("H(symbols) chain", round(H(sym_chain), 3), "anchor", round(H(sym_anchor), 3))

fig, axes = plt.subplots(3, 1, figsize=(16, 8.2), dpi=100, sharex=True)
rows = [("open-loop chain: quantized deltas of the original neighbours, summed by the decoder", open_chain, C[1]),
        ("closed-loop chain: each delta measured from the previous reconstruction (decoder must run in order)", closed, C[0]),
        (f"anchor every {G} tokens: deltas from the anchor, all decodable in parallel", anchor, C[2])]
for ax, (title, rec, col) in zip(axes, rows):
    err = rec - x
    ax.axhspan(-D / 2, D / 2, color=GRID, alpha=0.6, lw=0)
    ax.step(np.arange(N), err, where="mid", color=col, lw=1.8)
    ax.set_title(title, fontsize=12.5, loc="left")
    lim = max(8, np.abs(err).max() + 3)
    ax.set_ylim(-lim, lim); ax.set_ylabel("error"); ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.99, 0.85, f"max |error| = {np.abs(err).max():.0f}", transform=ax.transAxes, ha="right", fontsize=13, color=INK)
axes[1].text(1, D / 2 + 0.8, "±Δ/2 band (Δ = 4)", color=MUTED, fontsize=11)
axes[-1].set_xlabel("token index")
fig.suptitle("Where you predict from decides whether quantization error walks away", x=0.01, ha="left", fontsize=15)
out = os.path.join(os.path.dirname(__file__), "..", "..", "img", "linear-prediction", "anchor.png")
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
from PIL import Image
w = Image.open(out).width                      # re-save so the trimmed image is exactly 1600 px wide
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3, dpi=100 * 1600 / w)
print("wrote", out)
