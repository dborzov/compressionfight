"""Figure: scalar sign grid vs joint k-means centroids on 16 correlated pairs.
Output: img/vector-quantization/grid-vs-joint.png"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()

BG, INK, INK2, MUTED, GRID = "#f3eadb", "#2a211b", "#5a4a3d", "#5a4a3d", "#b89f78"
C = ["#246b66", "#b8892e", "#b05a4c", "#6b4a7a", "#6f8f4e"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 13, "axes.titlesize": 15})

a = np.array([-2.0,-1.5,-1.2,-0.9,-0.6,-0.4,-0.2,-0.1, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.1])
b = np.array([-1.8,-1.6,-1.0,-1.1,-0.5,-0.2,-0.4, 0.1,-0.1, 0.4, 0.3, 0.9, 0.8, 1.4, 1.5, 1.9])
P = np.stack([a, b], 1)

# scalar: sign grid, reconstruct at ±mean|.|
ma, mb = np.abs(a).mean(), np.abs(b).mean()
rec = np.stack([np.where(a >= 0, ma, -ma), np.where(b >= 0, mb, -mb)], 1)
mse_s = ((P - rec) ** 2).sum(1).mean()
# vector: k-means K=4
Cb = P[[0, 5, 10, 15]].astype(float)
for _ in range(20):
    asg = ((P[:, None] - Cb[None]) ** 2).sum(-1).argmin(1)
    Cb = np.array([P[asg == k].mean(0) for k in range(4)])
d = ((P[:, None] - Cb[None]) ** 2).sum(-1); asg = d.argmin(1); mse_v = d.min(1).mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.6), dpi=100, sharex=True, sharey=True)
for ax in (ax1, ax2):
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.3, 2.3); ax.set_aspect("equal"); ax.grid(True, alpha=0.35)
    ax.set_xlabel("a"); ax.set_ylabel("b")

# left panel
ax1.axhline(0, color=INK2, linewidth=1.2); ax1.axvline(0, color=INK2, linewidth=1.2)
for rx in (-ma, ma):
    for ry in (-mb, mb):
        ax1.scatter(rx, ry, s=300, marker="X", color=C[1] if (rx * ry > 0) else C[3], edgecolor=BG, linewidth=1.5, zorder=4)
for p, r in zip(P, rec):
    ax1.plot([p[0], r[0]], [p[1], r[1]], color=MUTED, linewidth=1, zorder=2)
ax1.scatter(a, b, s=80, color=C[0], edgecolor=BG, zorder=3)
ax1.text(1.45, -1.9, "two cells hold\n1 point each", color=C[3], ha="center", fontsize=12)
ax1.text(-1.45, 1.9, "two cells hold\n1 point each", color=C[3], ha="center", fontsize=12)
ax1.set_title(f"Scalar: 1 bit for a, 1 bit for b  →  MSE {mse_s:.3f} per pair")

# right panel: Voronoi of the 4 centroids
xx, yy = np.meshgrid(np.linspace(-2.6, 2.6, 400), np.linspace(-2.3, 2.3, 360))
G = np.stack([xx.ravel(), yy.ravel()], 1)
lab = ((G[:, None] - Cb[None]) ** 2).sum(-1).argmin(1).reshape(xx.shape)
ax2.contourf(xx, yy, lab, levels=np.arange(-0.5, 4.5, 1), colors=[C[1], C[2], C[3], C[4]], alpha=0.14)
ax2.contour(xx, yy, lab, levels=np.arange(0.5, 3.5, 1), colors=[INK2], linewidths=1.0, linestyles="--")
for p, k in zip(P, asg):
    ax2.plot([p[0], Cb[k][0]], [p[1], Cb[k][1]], color=MUTED, linewidth=1, zorder=2)
ax2.scatter(a, b, s=80, color=C[0], edgecolor=BG, zorder=3)
for k, c in enumerate(Cb):
    ax2.scatter(*c, s=300, marker="X", color=[C[1], C[2], C[3], C[4]][k], edgecolor=BG, linewidth=1.5, zorder=4)
ax2.set_title(f"Vector: one 2-bit index per pair (k-means, K=4)  →  MSE {mse_v:.3f} per pair")
fig.suptitle("Same 2 bits per pair. The grid spends two of its four cells on nearly nothing; the codebook puts all four on the diagonal.",
             fontsize=13, color=INK2, y=0.98)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("/home/dima/Work/chooseyourcodec/docs/map/zoo/img/vector-quantization/grid-vs-joint.png")
print("ok", mse_s, mse_v)
