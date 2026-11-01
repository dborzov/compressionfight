"""Figure: compression ratio vs LZ window size on a synthetic log (real LZMA with dict_size swept)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()
import numpy as np, lzma, os

BG, INK, MUTED, GRID = "#f3eadb", "#2a211b", "#5a4a3d", "#b89f78"
C = ["#246b66", "#b8892e", "#b05a4c", "#6b4a7a"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
                     "font.family": "DejaVu Sans", "font.size": 13})

rng = np.random.default_rng(7)
# A synthetic service log. Each line is a template filled with a few random fields.
# 3000 templates, each ~90 bytes, drawn uniformly: a given template recurs about every
# 3000 lines = ~300 KB. Short-range repeats (timestamps, field names) sit within a few hundred bytes.
templates = [f"svc-{rng.integers(0, 40):02d} handler_{rng.integers(0, 5000):04d} {rng.choice(['ok', 'retry', 'timeout', 'denied'])} "
             f"path=/api/v{rng.integers(1, 4)}/{''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz'), 8))}/{{id}} "
             f"took={{ms}}ms user={{u}}" for _ in range(3000)]
lines = []
t = 1700000000
for i in range(40000):
    t += int(rng.integers(0, 3))
    tpl = templates[rng.integers(0, len(templates))]
    lines.append(f"{t} INFO " + tpl.format(id=rng.integers(0, 10**6), ms=rng.integers(1, 999), u=rng.integers(0, 10**5)))
data = "\n".join(lines).encode()
print("log bytes", len(data))

sizes = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]
ratios = []
for ds in sizes:
    f = [{"id": lzma.FILTER_LZMA2, "dict_size": ds, "preset": 6}]
    c = lzma.compress(data, format=lzma.FORMAT_RAW, filters=f)
    ratios.append(len(data) / len(c))
    print(f"dict {ds:>8} -> {len(c):>8} bytes, ratio {ratios[-1]:.2f}")

fig, ax = plt.subplots(figsize=(16, 7), dpi=100)
ax.plot(sizes, ratios, color=C[0], lw=2, marker="o", ms=7, mfc=BG, mew=2)
ax.set_xscale("log", base=2)
ax.set_xticks(sizes); ax.set_xticklabels([f"{s // 1024} KB" if s < 2**20 else f"{s // 2**20} MB" for s in sizes])
ax.set_xlabel("LZ window (LZMA dict_size)"); ax.set_ylabel("compression ratio (input bytes / output bytes)")
ax.grid(True, axis="y", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
ax.axvline(32768, color=C[1], lw=1.2, ls="--"); ax.text(32768 * 1.08, ratios[0] + 0.1, "DEFLATE's 32 KB", color=C[1], fontsize=12)
ax.axvline(300_000, color=C[2], lw=1.2, ls="--"); ax.text(300_000 * 1.08, ratios[0] + 0.1, "typical template recurrence (~300 KB)", color=C[2], fontsize=12)
for s_, r in zip(sizes, ratios):
    ax.annotate(f"{r:.2f}", (s_, r), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=11, color=MUTED)
ax.set_title("A 3.7 MB synthetic log: the same codec, the window dial turned", loc="left", fontsize=15, pad=14)
out = os.path.join(os.path.dirname(__file__), "..", "..", "img", "lz-family", "window.png")
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
from PIL import Image
w = Image.open(out).width                      # re-save so the trimmed image is exactly 1600 px wide
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3, dpi=100 * 1600 / w)
print("wrote", out)
