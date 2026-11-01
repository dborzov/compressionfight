"""Figure: the piggy/chicken island map with k-means centroids and Voronoi cells,
and the 4-entry codebook with its cells.  Output: img/vector-quantization/piggy-map.png"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, INK, INK2, MUTED, GRID = "#0b0714", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a"
C = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 13, "axes.titlesize": 15})

pigs = np.array([[1.0, 2.0], [1.4, 2.6], [1.8, 2.3]])
chick = np.array([[6.1, 0.9], [6.5, 1.5], [5.8, 1.3]])
X = np.vstack([pigs, chick])
Cb = np.array([pigs.mean(0), chick.mean(0)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.2), dpi=100, gridspec_kw={"width_ratios": [1.45, 1]})

# left: island map, Voronoi of the two centroids via a dense grid
xx, yy = np.meshgrid(np.linspace(0, 8, 400), np.linspace(0, 4, 200))
G = np.stack([xx.ravel(), yy.ravel()], 1)
lab = ((G[:, None] - Cb[None]) ** 2).sum(-1).argmin(1).reshape(xx.shape)
ax1.contourf(xx, yy, lab, levels=[-0.5, 0.5, 1.5], colors=[C[0], C[1]], alpha=0.12)
ax1.contour(xx, yy, lab, levels=[0.5], colors=[INK2], linewidths=1.2, linestyles="--")
ax1.scatter(*pigs.T, s=140, color=C[0], edgecolor=BG, linewidth=1.5, zorder=3, label="piggy houses")
ax1.scatter(*chick.T, s=140, color=C[1], edgecolor=BG, linewidth=1.5, zorder=3, label="chicken houses")
ax1.scatter(*Cb.T, s=260, marker="X", color=INK, edgecolor=BG, linewidth=1.5, zorder=4, label="centroids (codebook)")
for k, c in enumerate(Cb):
    ax1.annotate(f"index {k}: ({c[0]:.2f}, {c[1]:.2f})", c, xytext=(44, -30), textcoords="offset points",
                 ha="center", color=INK, fontsize=12)
for p, c in list(zip(pigs, [Cb[0]] * 3)) + list(zip(chick, [Cb[1]] * 3)):
    ax1.plot([p[0], c[0]], [p[1], c[1]], color=MUTED, linewidth=1, zorder=2)
ax1.set_xlim(0, 8); ax1.set_ylim(0, 3.6); ax1.set_aspect("equal")
ax1.set_xlabel("km east"); ax1.set_ylabel("km north")
ax1.set_title("Six houses, two codewords: 1 bit per house, total squared error 0.93")
ax1.legend(loc="upper right", frameon=False, fontsize=11)
ax1.grid(True, alpha=0.4)

# right: the 4-entry codebook and its Voronoi cells
CB = np.array([[-0.8, -0.6], [-0.2, 0.7], [0.5, 0.1], [0.9, -0.7]])
x = np.array([0.4, 0.2])
xx, yy = np.meshgrid(np.linspace(-1.3, 1.3, 400), np.linspace(-1.1, 1.1, 340))
G = np.stack([xx.ravel(), yy.ravel()], 1)
lab = ((G[:, None] - CB[None]) ** 2).sum(-1).argmin(1).reshape(xx.shape)
ax2.contourf(xx, yy, lab, levels=np.arange(-0.5, 4.5, 1), colors=C[:4], alpha=0.14)
ax2.contour(xx, yy, lab, levels=np.arange(0.5, 3.5, 1), colors=[INK2], linewidths=1.0, linestyles="--")
for k, c in enumerate(CB):
    ax2.scatter(*c, s=260, marker="X", color=C[k], edgecolor=BG, linewidth=1.5, zorder=4)
    ax2.annotate(f"index {k} = {k:02b}", c, xytext=(0, 14 if k != 2 else -26), textcoords="offset points", ha="center", color=INK, fontsize=12)
ax2.scatter(*x, s=120, color=INK, zorder=5)
ax2.annotate("x = (0.4, 0.2)\nnearest: index 2\nsent as '10'", x, xytext=(-10, 40), textcoords="offset points",
             color=INK, fontsize=12, arrowprops=dict(arrowstyle="-", color=INK2))
ax2.set_xlim(-1.3, 1.3); ax2.set_ylim(-1.1, 1.1); ax2.set_aspect("equal")
ax2.set_title("A 4-entry codebook: every point in a cell decodes to its X")
ax2.grid(True, alpha=0.4)

fig.tight_layout()
fig.savefig("/home/dima/Work/chooseyourcodec/docs/map/zoo/img/vector-quantization/piggy-map.png")
print("ok")
