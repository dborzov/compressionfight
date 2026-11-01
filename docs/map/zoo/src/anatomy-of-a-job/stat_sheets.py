"""Radar stat sheets for three audio jobs, read from the game's heist requirement stats
(CYC.HEISTS walkman / phonecall / studio, exported to ../codec-dissections/cyc_data.json).
Run: uv run --with matplotlib --with numpy python stat_sheets.py"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, DIM = "#0b0714", "#ece6f7", "#8a81a3"
ORANGE, PURPLE, CYAN, GREEN, YELLOW = "#ff6a3d", "#d05cff", "#38d6ff", "#4dff7c", "#ffe23d"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.edgecolor": DIM, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                     "text.color": FG, "font.size": 14, "axes.titlesize": 17})
here = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(here, "..", "codec-dissections", "cyc_data.json")))
H = data["heists"]
axes_keys = [("latency", "latency\ntolerance"), ("encBudget", "encoder\ncompute"), ("decBudget", "decoder\ncompute"),
             ("ra", "random\naccess"), ("bandwidth", "bandwidth\npressure"), ("decMem", "decoder\nmemory")]
jobs = [("walkman", "WALKMAN: MP3 (music distribution)", ORANGE), ("phonecall", "THE PHONE CALL: Opus / SoundStream (VoIP)", CYAN),
        ("studio", "STUDIO MASTER: FLAC (lossless archive)", GREEN)]
ang = np.linspace(0, 2 * np.pi, len(axes_keys), endpoint=False)
fig, axs = plt.subplots(1, 3, figsize=(16, 6.6), dpi=100, subplot_kw=dict(polar=True))
for ax, (hid, title, color) in zip(axs, jobs):
    r = H[hid]["req"]; vals = np.array([r[k] for k, _ in axes_keys], float)
    v = np.concatenate([vals, vals[:1]]); a = np.concatenate([ang, ang[:1]])
    ax.plot(a, v, color=color, lw=2.5); ax.fill(a, v, color=color, alpha=0.25)
    ax.set_xticks(ang); ax.set_xticklabels([lab for _, lab in axes_keys], fontsize=12)
    ax.set_yticks([1, 2, 3, 4, 5]); ax.set_yticklabels(["", "", "3", "", "5"], color=DIM, fontsize=10)
    ax.set_ylim(0, 5); ax.grid(color="#2a2140")
    ax.spines["polar"].set_color(DIM)
    lossless = "lossless" if r["lossless"] else "lossy"; prior = "side info OK" if r["priorOK"] else "self-describing"
    ax.set_title(f"{title}\n{lossless} · {prior}", color=color, pad=22, fontsize=14)
fig.suptitle("The same 10-second stereo clip, three jobs. Six of the stat-sheet axes, 0–5, as the game scores them.",
             fontsize=16, y=0.99)
fig.text(0.5, 0.02, "Latency tolerance 0 = milliseconds, 5 = take all night.  Bandwidth pressure 5 = every bit counts.  "
         "Random access 5 = any element on its own.", ha="center", color=DIM, fontsize=12.5)
out = os.path.join(here, "..", "..", "img", "anatomy-of-a-job", "stat_sheets.png")
fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(out); print("wrote", os.path.abspath(out))
