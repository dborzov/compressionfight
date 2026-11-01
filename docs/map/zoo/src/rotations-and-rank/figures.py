"""Figures for docs/map/zoo/rotations-and-rank.md.
Run: uv run --with matplotlib --with numpy python docs/map/zoo/src/rotations-and-rank/figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, DIM = "#0b0714", "#ece6f5", "#8a7fa3"
ACC1, ACC2, ACC3, ACC4 = "#ffb454", "#5ad1ff", "#ff6b9d", "#9df27a"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": DIM, "ytick.color": DIM,
    "axes.edgecolor": DIM, "grid.color": "#2a2240", "font.size": 13,
    "axes.titlesize": 15, "axes.titleweight": "bold",
})
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "img", "rotations-and-rank")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=100); plt.close(fig); print("wrote", path)


# ---------- 1. before/after scatter with 2-bit grids ----------
rng = np.random.default_rng(7)
x = rng.standard_normal(40) * 2.0
y = 0.8 * x + rng.standard_normal(40) * 0.5
P = np.stack([x, y], 1); P -= P.mean(0)
C = np.cov(P.T, bias=True); lam, V = np.linalg.eigh(C); V = V[:, ::-1]; lam = lam[::-1]
if V[0, 0] < 0: V[:, 0] *= -1
if V[1, 1] < 0: V[:, 1] *= -1
R = P @ V


def quant(A, bits):
    out = np.zeros_like(A); edges = []
    for j, b in enumerate(bits):
        if b == 0:
            edges.append(np.array([0.0])); continue
        L = 2 ** b; s = 2 * np.abs(A[:, j]).max() / L
        q = np.clip(np.floor(A[:, j] / s), -L // 2, L // 2 - 1)
        out[:, j] = (q + 0.5) * s
        edges.append(np.arange(-L // 2, L // 2 + 1) * s)
    return out, edges


def mse(A, B): return ((A - B) ** 2).sum(1).mean()


fig, axes = plt.subplots(1, 3, figsize=(16, 6.4))
lim = 6.2
for ax in axes:
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
# panel 0: original axes, 2+2
Q, ed = quant(P, [2, 2])
ax = axes[0]
for e in ed[0]: ax.axvline(e, color=DIM, lw=0.8)
for e in ed[1]: ax.axhline(e, color=DIM, lw=0.8)
ax.scatter(P[:, 0], P[:, 1], s=36, color=ACC1, zorder=3, label="data")
ax.scatter(Q[:, 0], Q[:, 1], s=90, marker="s", facecolors="none", edgecolors=ACC2, zorder=4, label="quantized")
for p, q in zip(P, Q): ax.plot([p[0], q[0]], [p[1], q[1]], color=ACC3, lw=0.8, alpha=0.7)
ax.set_title(f"Original axes, 2 + 2 bits\nMSE = {mse(P, Q):.3f}")
ax.legend(frameon=False, loc="upper left")
# panel 1: rotated axes, 2+2 (grid drawn in original coordinates)
Qr, ed = quant(R, [2, 2]); Qb = Qr @ V.T
ax = axes[1]
for e in ed[0]:
    p0 = e * V[:, 0]; d = V[:, 1] * 20
    ax.plot([p0[0] - d[0], p0[0] + d[0]], [p0[1] - d[1], p0[1] + d[1]], color=DIM, lw=0.8)
for e in ed[1]:
    p0 = e * V[:, 1]; d = V[:, 0] * 20
    ax.plot([p0[0] - d[0], p0[0] + d[0]], [p0[1] - d[1], p0[1] + d[1]], color=DIM, lw=0.8)
ax.scatter(P[:, 0], P[:, 1], s=36, color=ACC1, zorder=3)
ax.scatter(Qb[:, 0], Qb[:, 1], s=90, marker="s", facecolors="none", edgecolors=ACC2, zorder=4)
for p, q in zip(P, Qb): ax.plot([p[0], q[0]], [p[1], q[1]], color=ACC3, lw=0.8, alpha=0.7)
ax.set_title(f"Rotated 38.7°, still 2 + 2 bits\nMSE = {mse(R, Qr):.3f}  (no gain)")
# panel 2: rotated, 4+0
Qr, ed = quant(R, [4, 0]); Qb = Qr @ V.T
ax = axes[2]
for e in ed[0]:
    p0 = e * V[:, 0]; d = V[:, 1] * 20
    ax.plot([p0[0] - d[0], p0[0] + d[0]], [p0[1] - d[1], p0[1] + d[1]], color=DIM, lw=0.8)
d = V[:, 0] * 20
ax.plot([-d[0], d[0]], [-d[1], d[1]], color=ACC4, lw=1.5)
ax.scatter(P[:, 0], P[:, 1], s=36, color=ACC1, zorder=3)
ax.scatter(Qb[:, 0], Qb[:, 1], s=90, marker="s", facecolors="none", edgecolors=ACC2, zorder=4)
for p, q in zip(P, Qb): ax.plot([p[0], q[0]], [p[1], q[1]], color=ACC3, lw=0.8, alpha=0.7)
ax.set_title(f"Rotated, 4 + 0 bits (minor axis dropped)\nMSE = {mse(R, Qr):.3f}  (3.2× smaller)")
fig.suptitle("Same 40 points, same 4 bits per point. The rotation only pays once the bit allocation can follow the variance.",
             fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "scatter.png")

# ---------- 2. RoPE wandering ----------
k = np.array([8.0, 0.3])
m = np.arange(0, 32)
th = m * 0.5
rot = np.stack([np.cos(th) * k[0] - np.sin(th) * k[1], np.sin(th) * k[0] + np.cos(th) * k[1]], 1)
fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
ax.plot(m, rot[:, 0], "o-", color=ACC1, lw=2, label="channel 2i after RoPE")
ax.plot(m, rot[:, 1], "s-", color=ACC2, lw=2, label="channel 2i+1 after RoPE")
ax.axhline(8.0, color=ACC1, ls="--", lw=1.2, alpha=0.8); ax.axhline(0.3, color=ACC2, ls="--", lw=1.2, alpha=0.8)
ax.text(31.5, 8.35, "pre-RoPE: 8.0 at every position", color=ACC1, ha="right", fontsize=12)
ax.text(31.5, -0.55, "pre-RoPE: 0.3 at every position", color=ACC2, ha="right", fontsize=12)
ax.set_xlabel("token position m"); ax.set_ylabel("stored value"); ax.set_ylim(-9.5, 9.5)
ax.set_title("The same key content (8.0, 0.3), stored after RoPE with θ = 0.5 rad per position")
ax.legend(frameon=False, loc="lower left"); ax.grid(True, alpha=0.4)
ax = axes[1]
ax.set_aspect("equal"); ax.set_xlim(-9.5, 9.5); ax.set_ylim(-9.5, 9.5)
circ = plt.Circle((0, 0), np.hypot(*k), fill=False, color=DIM, ls=":")
ax.add_patch(circ)
ax.scatter(rot[:, 0], rot[:, 1], c=m, cmap="cool", s=50, zorder=3)
ax.scatter([8.0], [0.3], s=220, marker="*", color=ACC1, zorder=4, label="pre-RoPE key")
ax.axvline(0, color=DIM, lw=0.6); ax.axhline(0, color=DIM, lw=0.6)
ax.set_xlabel("channel 2i"); ax.set_ylabel("channel 2i+1")
ax.set_title("Per-channel range [−8, 8] on both,\ninstead of a constant 8.0 and 0.3", fontsize=13)
ax.legend(frameon=False, loc="upper left"); ax.grid(True, alpha=0.3)
fig.tight_layout(rect=(0, 0, 1, 0.98))
save(fig, "rope.png")

# ---------- 3. singular value decay: synthetic K pre-RoPE, K post-RoPE, V ----------
L, d = 1024, 128
rng = np.random.default_rng(11)
# keys: rank-16 structure + small noise (synthetic stand-in for "pre-RoPE keys of one prompt")
A = rng.standard_normal((L, 16)) * np.linspace(3, 0.6, 16)
B = rng.standard_normal((16, d))
K = A @ B + 0.35 * rng.standard_normal((L, d))
# values: no shared subspace, mild spectrum
V_ = rng.standard_normal((L, d)) * np.linspace(1.4, 0.6, d)
# post-RoPE keys: rotate channel pairs by position-dependent angles
freqs = 10000 ** (-np.arange(0, d, 2) / d)
ang = np.arange(L)[:, None] * freqs[None, :]
K2 = K.copy()
c, s = np.cos(ang), np.sin(ang)
K2[:, 0::2] = c * K[:, 0::2] - s * K[:, 1::2]
K2[:, 1::2] = s * K[:, 0::2] + c * K[:, 1::2]
sv = {name: np.linalg.svd(M, compute_uv=False) for name, M in [("keys, pre-RoPE", K), ("keys, post-RoPE", K2), ("values", V_)]}
fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
ax = axes[0]
for (name, sval), col in zip(sv.items(), [ACC1, ACC3, ACC2]):
    ax.plot(np.arange(1, d + 1), sval / sval[0], lw=2.5, color=col, label=name)
ax.set_yscale("log"); ax.set_xlabel("index i"); ax.set_ylabel("σᵢ / σ₁")
ax.set_title("Singular values of a 1024×128 matrix (synthetic)"); ax.legend(frameon=False); ax.grid(True, alpha=0.4)
ax = axes[1]
for (name, sval), col in zip(sv.items(), [ACC1, ACC3, ACC2]):
    ax.plot(np.arange(1, d + 1), np.cumsum(sval ** 2) / (sval ** 2).sum(), lw=2.5, color=col, label=name)
ax.axhline(0.99, color=DIM, ls="--"); ax.text(2, 0.955, "99 % of the energy", color=DIM)
ax.set_xlabel("rank r kept"); ax.set_ylabel("energy captured by rank r")
ax.set_title("Energy vs rank: a subspace exists for pre-RoPE keys only"); ax.legend(frameon=False, loc="lower right"); ax.grid(True, alpha=0.4)
fig.suptitle("Low rank is energy compaction along the token axis, and RoPE scrambles it (synthetic data built to mirror ShadowKV's observation)",
             fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "svd.png")
for name, sval in sv.items():
    e = np.cumsum(sval ** 2) / (sval ** 2).sum()
    print(name, "rank for 99%:", int(np.searchsorted(e, 0.99)) + 1, "rank for 90%:", int(np.searchsorted(e, 0.90)) + 1)

# ---------- 4. Hadamard flattening of an outlier channel ----------
H4 = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2.0
rng = np.random.default_rng(7)
Kc = rng.standard_normal((2000, 4)) * np.array([8.0, 0.5, 0.5, 0.5])
KH = Kc @ H4.T


def q_shared(A, b=4):
    Lv = 2 ** b; s = 2 * np.abs(A).max(1, keepdims=True) / Lv
    q = np.clip(np.floor(A / s), -Lv // 2, Lv // 2 - 1); return (q + 0.5) * s


e0 = ((Kc - q_shared(Kc)) ** 2).sum(1).mean(); e1 = ((KH - q_shared(KH)) ** 2).sum(1).mean()
fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))
for ax, M, title, col in [(axes[0], Kc, f"Raw channels, one int4 scale per vector: MSE {e0:.3f}", ACC3),
                          (axes[1], KH, f"After a 4×4 Hadamard, same quantizer: MSE {e1:.3f}", ACC4)]:
    ax.boxplot([M[:, j] for j in range(4)], widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=col, color=col, alpha=0.6), medianprops=dict(color=FG),
               whiskerprops=dict(color=DIM), capprops=dict(color=DIM), flierprops=dict(marker=".", color=DIM, alpha=0.3))
    ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["ch 0", "ch 1", "ch 2", "ch 3"])
    ax.set_ylim(-28, 28); ax.set_title(title); ax.grid(True, axis="y", alpha=0.4)
    ax.set_ylabel("value")
fig.suptitle("Hadamard does the opposite of PCA: it spreads one loud channel over all four so a shared scale fits everyone", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.92))
save(fig, "hadamard.png")
print("hadamard mse", e0, e1, e0 / e1)
