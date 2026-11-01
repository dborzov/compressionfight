"""Figure: run structure of a real text sample before and after the BWT, plus the order-0 entropy after MTF,
and bits/char vs block size. The sample is the Map's own documentation (docs/map/*.md)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()
import numpy as np, os, glob

BG, INK, MUTED, GRID = "#f3eadb", "#2a211b", "#5a4a3d", "#b89f78"
C = ["#246b66", "#b8892e", "#b05a4c", "#6b4a7a"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
                     "font.family": "DejaVu Sans", "font.size": 13})

here = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(here, "..", "..", "..", "*.md")))
text = b"".join(open(f, "rb").read() for f in files)
print("files", [os.path.basename(f) for f in files], "bytes", len(text))

def suffix_array(s):
    """Prefix-doubling suffix array of a cyclic string (so this gives the BWT's rotation order)."""
    n = len(s); rank = np.frombuffer(s, dtype=np.uint8).astype(np.int64); k = 1
    sa = np.argsort(rank, kind="stable")
    while k < n:
        key = rank * (n + 1) + rank[(np.arange(n) + k) % n]
        sa = np.argsort(key, kind="stable")
        newrank = np.zeros(n, dtype=np.int64)
        newrank[sa[1:]] = np.cumsum(key[sa[1:]] != key[sa[:-1]])
        rank = newrank; k *= 2
        if rank.max() == n - 1: break
    return sa

def bwt(block):
    sa = suffix_array(block); b = np.frombuffer(block, dtype=np.uint8)
    return b[(sa - 1) % len(block)]

def mtf(v):
    tbl = list(range(256)); out = np.empty(len(v), dtype=np.int64)
    for i, c in enumerate(v):
        j = tbl.index(c); out[i] = j; tbl.insert(0, tbl.pop(j))
    return out

def runs(v):
    v = np.asarray(v); return np.diff(np.flatnonzero(np.concatenate([[True], v[1:] != v[:-1], [True]])))

def H(v):
    _, c = np.unique(v, return_counts=True); p = c / c.sum(); return -(p * np.log2(p)).sum()

raw = np.frombuffer(text, dtype=np.uint8)
L = bwt(text)
r_raw, r_bwt = runs(raw), runs(L)
m = mtf(L)
print("runs raw", len(r_raw), "runs bwt", len(r_bwt), "longest raw", r_raw.max(), "longest bwt", r_bwt.max())
print("H0 raw", round(H(raw), 3), "H0 bwt", round(H(L), 3), "H0 mtf", round(H(m), 3), "zeros in mtf", round((m == 0).mean(), 3))

# block-size sweep: bits/char = H0 of MTF output over blocks of size B (a stand-in for bzip2's back end)
sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
bpc = []
for B in sizes:
    syms = np.concatenate([mtf(bwt(text[i:i + B])) for i in range(0, len(text) - B + 1, B)])
    bpc.append(H(syms)); print("block", B, "bits/char after BWT+MTF", round(bpc[-1], 3))

fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
ax = axes[0]
bins = np.arange(1, 41)
ax.hist(np.clip(r_raw, 1, 40), bins=bins, color=C[0], histtype="step", lw=2.2, label=f"raw text: {len(r_raw):,} runs")
ax.hist(np.clip(r_bwt, 1, 40), bins=bins, color=C[1], histtype="step", lw=2.2, label=f"after BWT: {len(r_bwt):,} runs")
ax.set_yscale("log"); ax.set_xlabel("run length (40 = 40 or more)"); ax.set_ylabel("number of runs")
ax.legend(frameon=False); ax.grid(True, axis="y", lw=0.5); ax.spines[["top", "right"]].set_visible(False)
ax.set_title(f"Run lengths in {len(text) // 1024} KB of Markdown, before and after the BWT", loc="left", fontsize=13)
ax = axes[1]
ax.plot(sizes, bpc, color=C[2], lw=2, marker="o", ms=7, mfc=BG, mew=2)
for s_, b in zip(sizes, bpc):
    ax.annotate(f"{b:.2f}", (s_, b), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=11, color=MUTED)
ax.set_xscale("log", base=2); ax.set_xticks(sizes); ax.set_xticklabels([f"{s // 1024} KB" for s in sizes])
ax.axhline(H(raw), color=C[0], lw=1.2, ls="--"); ax.text(sizes[0], H(raw) + 0.05, f"order-0 entropy of the raw text: {H(raw):.2f}", color=C[0], fontsize=11)
ax.set_xlabel("BWT block size"); ax.set_ylabel("bits/char: order-0 entropy after BWT + MTF")
ax.grid(True, axis="y", lw=0.5); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Bigger blocks, more context, fewer bits", loc="left", fontsize=13)
out = os.path.join(here, "..", "..", "img", "sorting-transforms", "runs.png")
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
from PIL import Image
w = Image.open(out).width                      # re-save so the trimmed image is exactly 1600 px wide
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3, dpi=100 * 1600 / w)
print("wrote", out)
