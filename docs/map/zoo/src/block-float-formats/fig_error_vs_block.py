"""Figure: int4 absmax quantization error vs block size on a vector with 1% outliers,
compared with the same vector without outliers.  Output: img/block-float-formats/error-vs-block.png"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, INK, INK2, MUTED, GRID = "#0b0714", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a"
C = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 13, "axes.titlesize": 15})

rng = np.random.default_rng(2)
W = rng.normal(0, 1, 4096)
W_out = W.copy(); idx = rng.choice(4096, 40, replace=False); W_out[idx] *= 20   # 1% outliers, 20× larger

def q_int4(blocks):
    s = np.abs(blocks).max(axis=1, keepdims=True) / 7
    return np.round(blocks / s) * s

sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 4096]
res = {}
for name, v in [("1% outliers at 20×", W_out), ("no outliers", W)]:
    res[name] = [((q_int4(v.reshape(-1, n)) - v.reshape(-1, n)) ** 2).mean() for n in sizes]

fig, ax = plt.subplots(figsize=(16, 7.6), dpi=100)
for (name, ys), col in zip(res.items(), [C[1], C[0]]):
    ax.plot(sizes, ys, color=col, linewidth=2.2, marker="o", markersize=7, label=name)
    ax.text(sizes[-1] * 1.15, ys[-1], name, color=col, va="center", fontsize=12)
for n, e in zip(sizes, res["1% outliers at 20×"]):
    ax.annotate(f"{(16 + 4 * n) / n:.2f} bpw", (n, e), xytext=(0, 12), textcoords="offset points",
                ha="center", color=INK2, fontsize=11)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes])
ax.set_xlim(6, 9000)
ax.set_xlabel("block size n (int4, one absmax scale per block)")
ax.set_ylabel("mean squared error (log scale)")
ax.set_title("4,096 Gaussian weights, 40 of them 20× too big: the block size decides how many neighbours each outlier ruins")
ax.legend(loc="upper left", frameon=False)
ax.grid(True, alpha=0.35, which="both")
fig.tight_layout()
fig.savefig("/home/dima/Work/chooseyourcodec/docs/map/zoo/img/block-float-formats/error-vs-block.png")
print({k: [round(x, 4) for x in v] for k, v in res.items()})
