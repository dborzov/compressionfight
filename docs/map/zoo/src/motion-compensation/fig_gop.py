"""Figure: estimated bits per frame across two GOPs of a synthetic moving-texture sequence (toy codec)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()
import numpy as np, os

BG, INK, MUTED, GRID = "#f3eadb", "#2a211b", "#5a4a3d", "#b89f78"
C = ["#246b66", "#b8892e", "#b05a4c", "#6b4a7a"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
                     "font.family": "DejaVu Sans", "font.size": 13})

rng = np.random.default_rng(5)
# A 64x64 textured background that pans 2 px/frame, plus sensor noise; 16 frames, blocks of 8x8, search ±4.
tex = rng.integers(0, 256, (96, 128))
tex = np.clip(np.round(0.5 * tex + 0.5 * np.roll(tex, 1, 0) + 0.25 * np.roll(tex, 1, 1) - 60), 0, 255)  # smooth it a bit
frames = []
for t in range(16):
    f = tex[8:72, 8 + 2 * t: 72 + 2 * t]
    frames.append(np.clip(f + rng.normal(0, 2, f.shape), 0, 255).round().astype(int))

Q = 8
def H(v):
    _, c = np.unique(v, return_counts=True); p = c / c.sum(); return -(p * np.log2(p)).sum() * len(v)
def intra_bits(f):
    # a crude intra predictor: left neighbour, quantized residual, order-0 entropy
    r = np.round((f - np.hstack([np.zeros((64, 1)), f[:, :-1]])) / Q).astype(int)
    return H(r.ravel())
def inter_bits(f, ref, B=8, S=4):
    bits = 0.0; residuals = []
    for by in range(0, 64, B):
        for bx in range(0, 64, B):
            blk = f[by:by + B, bx:bx + B]; best = None
            for dy in range(-S, S + 1):
                for dx in range(-S, S + 1):
                    sy, sx = by - dy, bx - dx
                    if sy >= 0 and sy + B <= 64 and sx >= 0 and sx + B <= 64:
                        cand = ref[sy:sy + B, sx:sx + B]; sad = np.abs(blk - cand).sum()
                        if best is None or sad < best[0]: best = (sad, cand)
            residuals.append(np.round((blk - best[1]) / Q).astype(int).ravel())
            bits += 2 * np.log2(2 * S + 1)                 # the vector, uncoded
    return bits + H(np.concatenate(residuals))

G = 8
kinds, bits = [], []
for t, f in enumerate(frames):
    if t % G == 0:
        kinds.append("I"); bits.append(intra_bits(f))
    else:
        kinds.append("P"); bits.append(inter_bits(f, frames[t - 1]))
for t, (k, b) in enumerate(zip(kinds, bits)): print(t, k, round(b))
print("I mean", np.mean([b for k, b in zip(kinds, bits) if k == "I"]).round(), "P mean", np.mean([b for k, b in zip(kinds, bits) if k == "P"]).round())

fig, ax = plt.subplots(figsize=(16, 6.4), dpi=100)
cols = [C[1] if k == "I" else C[0] for k in kinds]
ax.bar(range(16), bits, color=cols, width=0.72)
for t, (k, b) in enumerate(zip(kinds, bits)):
    ax.text(t, b + 400, f"{k}\n{b / 1000:.1f}k", ha="center", color=INK, fontsize=11)
ax.set_xticks(range(16)); ax.set_xlabel("frame"); ax.set_ylabel("estimated bits (order-0 entropy of quantized residual + vectors)")
ax.grid(True, axis="y", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, max(bits) * 1.25)
ax.text(0.5, max(bits) * 1.15, "I = intra-coded (seek point)", color=C[1], fontsize=12)
ax.text(5.5, max(bits) * 1.15, "P = predicted from the previous frame", color=C[0], fontsize=12)
ax.set_title("A panning 64×64 texture, GOP of 8: the keyframe costs what the whole picture costs; a P-frame costs the motion", loc="left", fontsize=14, pad=14)
out = os.path.join(os.path.dirname(__file__), "..", "..", "img", "motion-compensation", "gop.png")
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
from PIL import Image
w = Image.open(out).width                      # re-save so the trimmed image is exactly 1600 px wide
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3, dpi=100 * 1600 / w)
print("wrote", out)
