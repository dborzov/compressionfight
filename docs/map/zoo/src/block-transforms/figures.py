"""Figures for docs/map/zoo/block-transforms.md.
Run: uv run --with matplotlib --with numpy python docs/map/zoo/src/block-transforms/figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mimg

BG, FG, DIM = "#0b0714", "#ece6f5", "#8a7fa3"
ACC1, ACC2, ACC3 = "#ffb454", "#5ad1ff", "#ff6b9d"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": DIM, "ytick.color": DIM,
    "axes.edgecolor": DIM, "grid.color": "#2a2240", "font.size": 13,
    "axes.titlesize": 15, "axes.titleweight": "bold",
})
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "img", "block-transforms")
os.makedirs(OUT, exist_ok=True)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(ROOT)))


def dct_matrix(N):
    n = np.arange(N)
    C = np.cos(np.pi * (2 * n[None, :] + 1) * n[:, None] / (2 * N)) * np.sqrt(2 / N)
    C[0] /= np.sqrt(2)
    return C


C8 = dct_matrix(8)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print("wrote", path)


# ---------- 1. 8x8 DCT basis images ----------
fig, axes = plt.subplots(8, 8, figsize=(16, 16.6))
for u in range(8):
    for v in range(8):
        basis = np.outer(C8[u], C8[v])
        ax = axes[u, v]
        ax.imshow(basis, cmap="magma", vmin=-0.25, vmax=0.25, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#2a2240")
fig.suptitle("The 64 basis images of the 8×8 DCT-II. Row u = vertical frequency, column v = horizontal frequency.\n"
             "Top-left is the flat DC image; every pixel block is a weighted sum of exactly these 64 pictures.",
             fontsize=15, y=0.995)
fig.text(0.5, 0.004, "Any 8×8 block = Σ X[u,v] · (this image). Smooth blocks need only the top-left few.",
         ha="center", color=DIM, fontsize=13)
fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.025, wspace=0.08, hspace=0.08)
save(fig, "basis.png")

# ---------- 2. coefficient magnitudes: smooth photo block vs noise block ----------
img = mimg.imread(os.path.join(REPO, "img", "fighters", "dct.png"))[:, :, :3]
gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]) * 255.0
# pick the 8x8 block with the median variance among non-flat blocks: a "typical" smooth block
blocks = gray.reshape(32, 8, 32, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
v = blocks.var(axis=(1, 2))
cand = np.argsort(v)
blk = blocks[cand[int(0.6 * len(cand))]]  # moderately textured block
rng = np.random.default_rng(3)
noise = rng.uniform(0, 255, (8, 8))
Xb = C8 @ blk @ C8.T
Xn = C8 @ noise @ C8.T
fig, axes = plt.subplots(2, 3, figsize=(16, 9.6), gridspec_kw={"width_ratios": [1, 1, 1.6]})
for row, (B, X, label) in enumerate([(blk, Xb, "8×8 block from a picture"), (noise, Xn, "8×8 block of uniform noise")]):
    axes[row, 0].imshow(B, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[row, 0].set_title(label); axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
    im = axes[row, 1].imshow(np.log10(np.abs(X) + 1), cmap="magma", vmin=0, vmax=3.3, interpolation="nearest")
    axes[row, 1].set_title("log₁₀(|DCT coefficient| + 1)"); axes[row, 1].set_xticks(range(8)); axes[row, 1].set_yticks(range(8))
    for (u, vv), val in np.ndenumerate(X):
        axes[row, 1].text(vv, u, f"{val:.0f}", ha="center", va="center", fontsize=8,
                          color="#000" if abs(val) > 60 else FG)
    mags = np.sort(np.abs(X).ravel())[::-1]
    energy = np.cumsum(mags ** 2) / (mags ** 2).sum()
    ax = axes[row, 2]
    ax.bar(np.arange(64), mags, color=ACC1 if row == 0 else ACC3, width=0.9)
    ax.set_yscale("log"); ax.set_ylim(0.3, 3000)
    ax.set_xlabel("coefficients sorted by magnitude"); ax.set_ylabel("|coefficient|")
    k90 = int(np.searchsorted(energy, 0.99)) + 1
    ax.axvline(k90 - 0.5, color=ACC2, ls="--")
    ax.text(k90 + 1, 1200, f"99% of the energy\nin {k90} of 64 coefficients", color=ACC2, fontsize=12)
    ax.grid(True, axis="y", alpha=0.4)
    ax.set_title("Where the energy sits (same total both rows, Parseval)")
fig.suptitle("Energy compaction is a property of the data, not of the transform", fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "compaction.png")
print("photo block: 99% energy in", int(np.searchsorted(np.cumsum(np.sort(np.abs(Xb).ravel())[::-1] ** 2) / (Xb ** 2).sum(), 0.99)) + 1,
      "; noise block:", int(np.searchsorted(np.cumsum(np.sort(np.abs(Xn).ravel())[::-1] ** 2) / (Xn ** 2).sum(), 0.99)) + 1)
print("energy check photo", (blk ** 2).sum(), (Xb ** 2).sum())

# ---------- 3. the toy row: 8 samples, 8 coefficients, 3 kept ----------
row = np.array([52, 55, 61, 66, 70, 61, 64, 73], float)
X = C8 @ row
keep = np.zeros(8); idx = np.argsort(-np.abs(X))[:3]; keep[idx] = X[idx]
rec = C8.T @ keep
fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
ax = axes[0]
ax.plot(row, "o-", color=ACC1, lw=2, ms=9, label="original row")
ax.plot(rec, "s--", color=ACC2, lw=2, ms=8, label="rebuilt from 3 of 8 coefficients")
for i, (a, b) in enumerate(zip(row, rec)):
    ax.plot([i, i], [a, b], color=ACC3, lw=1.5, alpha=0.8)
ax.set_xlabel("sample n"); ax.set_ylabel("value"); ax.set_title("Pixel domain: MSE = 11.66", fontsize=13)
ax.legend(frameon=False); ax.grid(True, alpha=0.4)
ax = axes[1]
ax.bar(np.arange(8) - 0.2, X, width=0.4, color=ACC1, label="all 8 DCT coefficients")
ax.bar(np.arange(8) + 0.2, keep, width=0.4, color=ACC2, label="the 3 we kept")
for k, val in enumerate(X):
    ax.text(k - 0.2, val + (6 if val >= 0 else -14), f"{val:.1f}", ha="center", fontsize=10, color=FG)
ax.set_xlabel("coefficient k"); ax.set_ylabel("X[k]"); ax.set_title("Transform domain: the same 11.66, all of it in 5 zeroed coefficients", fontsize=13)
ax.legend(frameon=False); ax.grid(True, axis="y", alpha=0.4)
fig.suptitle("[52, 55, 61, 66, 70, 61, 64, 73] through the DCT-II", fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "row.png")

# ---------- 4. Princen-Bradley sine windows ----------
N = 64  # frame hop; window length 2N
n = np.arange(2 * N)
w = np.sin(np.pi * (n + 0.5) / (2 * N))
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
ax = axes[0]
cols = [ACC1, ACC2, ACC3, "#9df27a", ACC1]
total = np.zeros(6 * N)
for j in range(5):
    seg = slice(j * N, j * N + 2 * N)
    ax.plot(np.arange(j * N, j * N + 2 * N), w, color=cols[j], lw=2.5)
    total[seg] += w ** 2
ax.set_ylabel("window w[n]"); ax.set_title("MDCT frames: each window is 2N long, frames start every N samples (50% overlap)")
ax.grid(True, alpha=0.4)
for j in range(6):
    ax.axvline(j * N, color=DIM, lw=0.8, ls=":")
ax = axes[1]
ax.plot(total, color=FG, lw=2.5)
ax.plot(w ** 2, color=ACC1, lw=1.5, ls="--", label="w²[n] of frame 0")
ax.plot(np.arange(N, 3 * N), w ** 2, color=ACC2, lw=1.5, ls="--", label="w²[n] of frame 1")
ax.set_ylim(-0.05, 1.3); ax.set_xlim(0, 6 * N)
ax.set_xlabel("sample n"); ax.set_ylabel("Σ w²")
ax.set_title("Princen–Bradley condition: w²[n] + w²[n+N] = 1, so the overlapped halves add back to exactly the signal")
ax.legend(frameon=False, loc="upper right"); ax.grid(True, alpha=0.4)
fig.tight_layout()
save(fig, "windows.png")
