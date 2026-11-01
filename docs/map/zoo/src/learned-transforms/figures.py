"""Figures for docs/map/zoo/learned-transforms.md.
Run: uv run --with matplotlib --with numpy python docs/map/zoo/src/learned-transforms/figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG, FG, DIM = "#0b0714", "#ece6f5", "#8a7fa3"
ACC1, ACC2, ACC3, ACC4, ACC5 = "#ffb454", "#5ad1ff", "#ff6b9d", "#9df27a", "#c3a6ff"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": DIM, "ytick.color": DIM,
    "axes.edgecolor": DIM, "grid.color": "#2a2240", "font.size": 13,
    "axes.titlesize": 15, "axes.titleweight": "bold",
})
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "img", "learned-transforms")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=100); plt.close(fig); print("wrote", path)


# ---------- 1. pipeline diagram ----------
fig, ax = plt.subplots(figsize=(16, 7.6))
ax.set_xlim(0, 16); ax.set_ylim(0, 7.6); ax.axis("off")


def box(x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec=FG, lw=1.2, alpha=0.92))
    ax.text(x + w / 2, y + h - 0.45, title, ha="center", va="center", fontsize=13.5, fontweight="bold", color="#111")
    ax.text(x + w / 2, y + h / 2 - 0.35, sub, ha="center", va="center", fontsize=10.5, color="#111", linespacing=1.4)


def arrow(x0, y0, x1, y1, label=None, color=FG, dy=-0.55):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, color=color, lw=1.8))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", va="top", fontsize=11, color=color, linespacing=1.3)


yb = 3.3; hb = 2.2
box(0.2, yb, 2.2, hb, "waveform frame", "320 samples × 16 bit\n= 5,120 bits", "#d9d2e9")
arrow(2.4, yb + hb / 2, 3.1, yb + hb / 2)
box(3.1, yb, 2.8, hb, "encoder (NEURON)", "causal conv stack,\nstride 320 → 1 frame\n8-dim latent z", ACC2)
arrow(5.9, yb + hb / 2, 6.9, yb + hb / 2, "z ∈ ℝ⁸")
box(6.9, yb, 2.8, hb, "quantizer (RVQ)", "residual VQ, 8 stages\n× 1,024 entries\n8 × 10 = 80 bits", ACC1)
arrow(9.7, yb + hb / 2, 10.7, yb + hb / 2, "80 bits\n64 : 1")
box(10.7, yb, 2.8, hb, "decoder (BRAINJAR)", "conv stack, ≈ 20 MB\nof weights = the prior\nover 'what speech is'", ACC5)
arrow(13.5, yb + hb / 2, 14.2, yb + hb / 2)
box(14.2, yb, 1.6, hb, "x̂", "plausible\n320 samples", "#d9d2e9")
# losses
ax.add_patch(FancyBboxPatch((3.0, 0.5), 7.2, 1.7, boxstyle="round,pad=0.05", fc="#1c1430", ec=ACC3, lw=1.2))
ax.text(6.6, 1.85, "training loss, back-propagated through all three boxes at once", ha="center", fontsize=12.5, color=ACC3, fontweight="bold")
ax.text(6.6, 1.05, "reconstruction  (multi-scale spectral / waveform)   +   adversarial  (FORGER: discriminators say 'real or decoded?')\n"
                   "+   feature-matching   +   VQ commitment  (keep z near the codebooks)", ha="center", fontsize=10.5, color=FG, linespacing=1.5)
arrow(6.6, 2.2, 4.5, yb - 0.05, color=ACC3)
arrow(6.6, 2.2, 8.3, yb - 0.05, color=ACC3)
arrow(6.6, 2.2, 12.1, yb - 0.05, color=ACC3)
# codebooks shared
ax.add_patch(FancyBboxPatch((6.6, 6.0), 7.2, 1.1, boxstyle="round,pad=0.05", fc="#1c1430", ec=ACC4, lw=1.2))
ax.text(10.2, 6.55, "shipped once to both ends (KEYMASTER + BRAINJAR): 8 codebooks × 1,024 × 8 floats, and the decoder weights",
        ha="center", fontsize=11, color=ACC4)
arrow(8.3, 6.0, 8.3, yb + hb + 0.05, color=ACC4)
arrow(12.1, 6.0, 12.1, yb + hb + 0.05, color=ACC4)
ax.text(8, 7.35, "A neural audio codec is one object: the latents are shaped for the codebooks, the codebooks for the decoder, the decoder for the ear",
        ha="center", fontsize=14, fontweight="bold")
save(fig, "pipeline.png")

# ---------- 2. qualitative bitrate vs quality sketch ----------
fig, ax = plt.subplots(figsize=(16, 7))
kbps = np.array([1.5, 3, 6, 12, 24, 48, 96, 128, 192, 320])
ax.set_xscale("log")
ax.set_xticks(kbps); ax.set_xticklabels([str(k) for k in kbps])
ax.set_xlim(1.2, 400); ax.set_ylim(0, 5.3); ax.set_yticks([])
ax.set_xlabel("kbit/s (log scale)"); ax.set_ylabel("perceived speech quality (qualitative)  →")
# curves (hand-drawn shapes, not measurements)
x = np.logspace(np.log10(1.5), np.log10(320), 200)
mp3 = 4.9 / (1 + (40 / x) ** 1.8)
opus = 4.8 / (1 + (9 / x) ** 1.7)
neural = 4.4 / (1 + (2.2 / x) ** 1.6)
ax.plot(x, mp3, color=ACC3, lw=3, label="MDCT + psychoacoustics + Huffman (MP3-like)")
ax.plot(x, opus, color=ACC1, lw=3, label="LPC speech model + short MDCT (Opus-like)")
ax.plot(x, neural, color=ACC2, lw=3, label="learned encoder → RVQ → generative decoder (SoundStream/EnCodec-like)")
ax.axvspan(3, 6, color=ACC2, alpha=0.12); ax.text(4.2, 0.35, "neural codecs:\n3–6 kbps", ha="center", color=ACC2, fontsize=12)
ax.axvspan(110, 150, color=ACC3, alpha=0.12); ax.text(128, 4.95, "MP3:\n128 kbps", ha="center", va="top", color=ACC3, fontsize=12)
ax.annotate("the shared prior moves the\nwhole curve left: the wire\ncarries 'which speech',\nthe weights carry 'what speech is'",
            xy=(3.5, 3.4), xytext=(14, 1.6), fontsize=12, color=FG,
            arrowprops=dict(arrowstyle="->", color=FG))
ax.legend(frameon=False, loc="lower right", fontsize=12); ax.grid(True, alpha=0.3, which="both")
ax.set_title("QUALITATIVE SKETCH, not measured data: three families of speech codec on the rate axis", color=ACC1)
fig.tight_layout()
save(fig, "rd_sketch.png")

# ---------- 3. a linear autoencoder learns PCA ----------
rng = np.random.default_rng(1)
X = rng.standard_normal((500, 2)) @ np.array([[2.0, 1.6], [0, 0.5]])
X -= X.mean(0)
w = np.array([0.1, -0.3]); hist = [w.copy()]
for it in range(3000):
    z = X @ w; E = X - np.outer(z, w)
    g = -2 * (X.T @ (E @ w) + E.T @ z) / len(X)
    w -= 0.02 * g
    if it in (5, 20, 50, 100, 300): hist.append(w.copy())
hist.append(w.copy())
lam, V = np.linalg.eigh(np.cov(X.T, bias=True)); u = V[:, -1] * np.sign(V[0, -1])
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], s=10, color=DIM, alpha=0.7, label="data")
for i, h in enumerate(hist):
    d = h / np.linalg.norm(h) * 6
    ax.plot([-d[0], d[0]], [-d[1], d[1]], color=ACC2, lw=1.2 + 0.4 * i, alpha=0.3 + 0.1 * i)
d = u * 6
ax.plot([-d[0], d[0]], [-d[1], d[1]], color=ACC1, lw=2.5, ls="--", label="PCA first eigenvector")
ax.plot([], [], color=ACC2, lw=2, label="autoencoder weight w during training (faint → solid)")
ax.set_aspect("equal"); ax.set_xlim(-7, 7); ax.set_ylim(-7, 7); ax.grid(True, alpha=0.3)
ax.set_title("2 → 1 → 2 linear autoencoder, tied weights, gradient descent"); ax.legend(frameon=False, loc="upper left", fontsize=11)
ax = axes[1]
# loss curve
w = np.array([0.1, -0.3]); losses = []
for it in range(3000):
    z = X @ w; E = X - np.outer(z, w); losses.append((E ** 2).sum(1).mean())
    g = -2 * (X.T @ (E @ w) + E.T @ z) / len(X); w -= 0.02 * g
ax.plot(losses, color=ACC2, lw=2.5, label="reconstruction MSE during training")
ax.axhline(lam[0], color=ACC1, ls="--", lw=2, label=f"λ₂ = {lam[0]:.3f} (PCA's floor: variance of the dropped axis)")
ax.set_xscale("log"); ax.set_xlabel("gradient step"); ax.set_ylabel("MSE"); ax.grid(True, alpha=0.3)
ax.set_title("It lands exactly on PCA's error"); ax.legend(frameon=False)
fig.suptitle("Make the encoder and decoder linear and train with MSE: you get PCA. Everything beyond PCA comes from nonlinearity and the loss.", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "linear_ae.png")
print("final w", w, "u", u, "final loss", losses[-1], "lambda2", lam[0])
