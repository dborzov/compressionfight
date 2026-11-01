"""Figures for docs/map/zoo/channel-and-layout.md.
Run: uv run --with matplotlib --with numpy python docs/map/zoo/src/channel-and-layout/figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from matplotlib.patches import FancyBboxPatch

BG, FG, DIM = "#0b0714", "#ece6f5", "#8a7fa3"
ACC1, ACC2, ACC3, ACC4 = "#ffb454", "#5ad1ff", "#ff6b9d", "#9df27a"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": DIM, "ytick.color": DIM,
    "axes.edgecolor": DIM, "grid.color": "#2a2240", "font.size": 13,
    "axes.titlesize": 15, "axes.titleweight": "bold",
})
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "img", "channel-and-layout")
os.makedirs(OUT, exist_ok=True)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(ROOT)))


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=100); plt.close(fig); print("wrote", path)


def H(a):
    _, c = np.unique(a, return_counts=True); p = c / c.sum(); return float(-(p * np.log2(p)).sum())


# ---------- 1. RGB vs YCbCr planes of a picture ----------
img = (mimg.imread(os.path.join(REPO, "img", "fighters", "dictionary.png"))[:, :, :3] * 255.0)
R, G, B = img[..., 0], img[..., 1], img[..., 2]
Y = 0.299 * R + 0.587 * G + 0.114 * B
Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
planes = [("R", R), ("G", G), ("B", B), ("Y", Y), ("Cb", Cb), ("Cr", Cr)]
fig = plt.figure(figsize=(16, 9.4))
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.15, 1.6])
for i, (name, Pl) in enumerate(planes):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    ax.imshow(Pl, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(f"{name}:  σ = {Pl.std():.1f},  H ≈ {H(np.round(Pl)):.2f} bits", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
ax = fig.add_subplot(gs[:, 4])
names = [p[0] for p in planes]; stds = [p[1].std() for p in planes]
cols = [ACC3, ACC4, ACC2, ACC1, ACC2, ACC3]
ax.barh(names[::-1], stds[::-1], color=cols[::-1])
for n_, s_ in zip(names[::-1], stds[::-1]):
    ax.text(s_ + 1, n_, f"{s_:.1f}", va="center", color=FG)
ax.set_xlabel("standard deviation of the plane (0–255 scale)")
ax.set_title("Three similar planes become\none big and two small")
ax.grid(True, axis="x", alpha=0.4)
fig.suptitle("RGB → YCbCr on a 256×256 drawn portrait: brightness was being stored three times", fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "planes.png")
print("stds", [(n, round(float(p.std()), 1)) for n, p in planes])
print("corr RG RB GB", np.corrcoef(R.ravel(), G.ravel())[0, 1], np.corrcoef(R.ravel(), B.ravel())[0, 1], np.corrcoef(G.ravel(), B.ravel())[0, 1])
print("corr Y-Cb Y-Cr Cb-Cr", np.corrcoef(Y.ravel(), Cb.ravel())[0, 1], np.corrcoef(Y.ravel(), Cr.ravel())[0, 1], np.corrcoef(Cb.ravel(), Cr.ravel())[0, 1])

# ---------- 2. byte planes of a float32 array ----------
n = 4096
t = np.linspace(0, 1, n, dtype=np.float32)
sig = (2.0 + 0.5 * np.sin(2 * np.pi * 3 * t) + 0.002 * np.random.default_rng(0).standard_normal(n)).astype(np.float32)
by = sig.view(np.uint8).reshape(-1, 4)  # little-endian: column 3 = sign+exponent
fig = plt.figure(figsize=(16, 8.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
ax = fig.add_subplot(gs[0, :])
ax.plot(sig, color=ACC1, lw=1.5)
ax.set_title("4096 float32 samples: a slow sine around 2.0 with a little noise"); ax.set_xlabel("index"); ax.set_ylabel("value")
ax.grid(True, alpha=0.4)
ax = fig.add_subplot(gs[1, 0])
show = by[:160, ::-1].T  # rows: b3 b2 b1 b0
ax.imshow(show, cmap="magma", aspect="auto", interpolation="nearest")
ax.set_yticks(range(4)); ax.set_yticklabels(["byte 3 (sign + exp)", "byte 2 (exp LSB + mantissa)", "byte 1 (mantissa)", "byte 0 (mantissa)"])
ax.set_xlabel("first 160 samples →"); ax.set_title("The four byte planes, as one row each")
ax = fig.add_subplot(gs[1, 1])
ents = [H(by[:, j]) for j in (3, 2, 1, 0)]
labels = ["byte 3", "byte 2", "byte 1", "byte 0", "interleaved\n(all bytes)"]
vals = ents + [H(sig.view(np.uint8))]
ax.bar(labels, vals, color=[ACC4, ACC2, ACC3, ACC3, DIM])
for i, v in enumerate(vals):
    ax.text(i, v + 0.15, f"{v:.2f}", ha="center", color=FG)
ax.set_ylim(0, 9); ax.set_ylabel("zeroth-order entropy, bits per byte")
ax.set_title(f"Per plane Σ = {sum(ents):.1f} bits/float; interleaved 4 × {vals[-1]:.2f} = {4 * vals[-1]:.1f}", fontsize=13)
ax.grid(True, axis="y", alpha=0.4)
fig.suptitle("Byte shuffle: the exponent plane is almost constant, the low mantissa planes are noise", fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "byteplanes.png")
print("entropies b3..b0", ents, "interleaved", vals[-1])

# ---------- 3. row vs column layout diagram ----------
rows = [("2024-03-01", "GET", "200", "/api/users", "12"),
        ("2024-03-01", "GET", "200", "/api/users", "15"),
        ("2024-03-01", "POST", "201", "/api/orders", "48"),
        ("2024-03-01", "GET", "404", "/api/users", "3"),
        ("2024-03-01", "GET", "200", "/api/items", "11"),
        ("2024-03-01", "GET", "200", "/api/users", "14")]
colnames = ["date", "method", "status", "path", "ms"]
colcol = [ACC4, ACC2, ACC1, ACC3, "#c3a6ff"]
fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
for ax in axes:
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.4); ax.axis("off")


def cell(ax, x, y, w, text, color, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, 0.8, boxstyle="round,pad=0.02", fc=color, ec=BG, alpha=0.85))
    ax.text(x + w / 2, y + 0.4, text, ha="center", va="center", fontsize=fs, color="#111")


ax = axes[0]
ax.set_title("Row-wise (CSV, a struct per record): unlike neighbours")
widths = [2.3, 1.1, 1.0, 2.3, 0.8]
for r, row in enumerate(rows):
    x = 0.4
    for c, (val, w) in enumerate(zip(row, widths)):
        cell(ax, x, 6.2 - r * 0.95, w, val, colcol[c]); x += w + 0.08
ax.annotate("the byte stream walks →→→ along each row:\n a date, then a word, then a number, then a path…\nLZ sees mush, the entropy coder sees a flat histogram",
            xy=(0.4, 0.15), fontsize=11.5, color=DIM)
ax = axes[1]
ax.set_title("Column-wise (Parquet, struct-of-arrays): like with like")
for c, (name, w) in enumerate(zip(colnames, widths)):
    y = 6.2 - c * 1.15
    ax.text(0.2, y + 0.4, name, va="center", fontsize=12, color=FG, ha="left")
    x = 1.3
    for r, row in enumerate(rows):
        cell(ax, x, y, 1.2, row[c], colcol[c], fs=9.5); x += 1.28
    distinct = len(set(r_[c] for r_ in rows))
    ax.text(9.95, y + 0.4, f"{distinct} distinct", va="center", ha="right", fontsize=11, color=DIM)
ax.annotate("date: 1 distinct → RLE.  method/status/path: 2–3 distinct → dictionary + bit-pack.\n ms: 6 distinct small ints → delta or bit-pack.  Then zstd per column.",
            xy=(0.2, 0.15), fontsize=11.5, color=DIM)
fig.suptitle("Same 30 values. The layout decides which statistics the downstream coders get to see.", fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "layout.png")
