"""Figure: the luminance contrast sensitivity function (Mannos-Sakrison 1974 fit) on a log
frequency axis, with the JPEG 8x8 DCT frequency range marked.  Output: img/perceptual-models/csf.png"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, INK, INK2, MUTED, GRID = "#0b0714", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a"
C = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 13, "axes.titlesize": 15})

f = np.logspace(np.log10(0.1), np.log10(60), 600)          # cycles per degree
A = 2.6 * (0.0192 + 0.114 * f) * np.exp(-((0.114 * f) ** 1.1))   # Mannos & Sakrison (1974)
A = A / A.max()
fpeak = f[A.argmax()]

fig, ax = plt.subplots(figsize=(16, 7.2), dpi=100)
ax.plot(f, A, color=C[0], linewidth=2.6)
ax.fill_between(f, 0, A, color=C[0], alpha=0.08)
ax.axvline(fpeak, color=INK2, linewidth=1, linestyle=":")
ax.text(fpeak * 1.08, 0.97, f"peak ≈ {fpeak:.1f} cycles/degree", color=INK, fontsize=12)
# where the eye is less sensitive
ax.annotate("coarse gradients:\nsensitivity drops (the eye\nis a band-pass filter)", (0.3, A[np.argmin(np.abs(f - 0.3))]),
            xytext=(0.13, 0.55), color=INK2, fontsize=12, arrowprops=dict(arrowstyle="-", color=INK2))
ax.annotate("fine texture: sensitivity falls\nby orders of magnitude.\nJPEG's table makes the step\nsizes grow in this direction", (30, A[np.argmin(np.abs(f - 30))]),
            xytext=(10, 0.6), color=INK2, fontsize=12, arrowprops=dict(arrowstyle="-", color=INK2))
# a sketch of where JPEG's 8x8 basis functions land at a typical viewing geometry (qualitative)
ax.axvspan(1, 32, color=C[3], alpha=0.06)
ax.text(5.6, 0.06, "approximate range covered by an 8×8 DCT block\nat ordinary screen viewing distances (qualitative)", ha="center", color=C[3], fontsize=11)
ax.set_xscale("log")
ax.set_xticks([0.1, 0.3, 1, 3, 8, 16, 32, 60]); ax.set_xticklabels(["0.1", "0.3", "1", "3", "8", "16", "32", "60"])
ax.set_xlabel("spatial frequency (cycles per degree of visual angle)")
ax.set_ylabel("relative contrast sensitivity (1 / threshold contrast)")
ax.set_title("Contrast sensitivity: how faint a grating can be and still be seen, as a function of how fine it is")
ax.set_ylim(0, 1.08); ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("/home/dima/Work/chooseyourcodec/docs/map/zoo/img/perceptual-models/csf.png")
print("ok", fpeak)
