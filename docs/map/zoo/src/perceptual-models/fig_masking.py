"""Figure: the toy masking curve over 24 bands, the signal levels, and the free bands.
Output: img/perceptual-models/masking-curve.png"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, INK, INK2, MUTED, GRID = "#0b0714", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a"
C = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 13, "axes.titlesize": 15})

bands = np.arange(24)
tone_band, tone_level = 8, 80.0
spread = np.where(bands < tone_band, -25.0 * (tone_band - bands), -10.0 * (bands - tone_band))
mask = tone_level + spread - 15.0
quiet = np.full(24, 20.0)
threshold = np.maximum(mask, quiet)
signal = np.full(24, 30.0); signal[tone_band] = 80; signal[[3, 14, 20]] = [50, 48, 42]
free = signal < threshold
bits_m = np.maximum(signal - threshold, 0) / 6.02
bits_q = np.maximum(signal - 20.0, 0) / 6.02

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), dpi=100, gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)
cols = [C[3] if f else C[0] for f in free]; cols[tone_band] = C[1]
ax1.bar(bands, signal, width=0.72, color=cols, edgecolor=BG, linewidth=1.5, zorder=3)
ax1.plot(bands, threshold, color=INK, linewidth=2.2, marker="o", markersize=5, zorder=4, label="masked threshold")
ax1.plot(bands, quiet, color=MUTED, linewidth=1.4, linestyle="--", zorder=2, label="threshold in quiet (toy: 20 dB)")
ax1.fill_between(bands, 0, threshold, color=INK, alpha=0.06, zorder=1)
ax1.text(tone_band, tone_level + 2, "masker: 80 dB", ha="center", color=C[1], fontsize=12)
ax1.annotate("−25 dB per band\nbelow the masker", (6.3, 48), xytext=(2.0, 62), color=INK2, fontsize=12,
             arrowprops=dict(arrowstyle="-", color=INK2))
ax1.annotate("−10 dB per band\nabove the masker", (11.5, 32), xytext=(13.5, 55), color=INK2, fontsize=12,
             arrowprops=dict(arrowstyle="-", color=INK2))
for b in np.flatnonzero(free):
    ax1.text(b, signal[b] + 2, "free", ha="center", color=C[3], fontsize=11)
ax1.set_ylim(0, 92); ax1.set_ylabel("level (dB)")
ax1.set_title("One loud tone in band 8 lifts the audibility threshold around it: bands 7, 9, 10, 11 are already below it")
ax1.legend(loc="upper right", frameon=False)
ax1.grid(True, alpha=0.3, axis="y")

ax2.bar(bands - 0.19, bits_q, width=0.36, color=MUTED, label="bits with no masking model (noise under 20 dB)")
ax2.bar(bands + 0.19, bits_m, width=0.36, color=C[2], label="bits with masking (noise under the curve)")
ax2.set_xticks(bands); ax2.set_xlabel("band (critical-band-like, toy)")
ax2.set_ylabel("bits per band\n(SNR / 6.02)")
ax2.text(23.5, 8.2, f"total: {bits_q.sum():.1f} bits without masking, {bits_m.sum():.1f} with", ha="right", color=INK, fontsize=12)
ax2.legend(loc="upper left", frameon=False, fontsize=11)
ax2.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("/home/dima/Work/chooseyourcodec/docs/map/zoo/img/perceptual-models/masking-curve.png")
print("ok", free.sum(), bits_m.sum(), bits_q.sum())
