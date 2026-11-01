"""Figure: a simulated 2-down-1-up staircase converging on a just-noticeable difference.
The listener is simulated with a logistic psychometric function; labelled as a simulation.
Output: img/perceptual-models/jnd-staircase.png"""
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

rng = np.random.default_rng(11)
true_jnd, slope = 3.0, 1.2                       # hidden: the level at which detection is 50%
def detects(level):                               # simulated listener, logistic psychometric curve
    return rng.random() < 1 / (1 + np.exp(-(level - true_jnd) / slope * 2))

level, step, levels, hits, correct_run, reversals, last_dir = 10.0, 2.0, [], [], 0, [], 0
for t in range(60):
    h = detects(level); levels.append(level); hits.append(h)
    if h:
        correct_run += 1
        if correct_run == 2:                      # two in a row: make it harder
            correct_run = 0; d = -1
        else:
            continue
    else:
        correct_run = 0; d = +1
    if last_dir and d != last_dir:
        reversals.append(t)
        if len(reversals) == 4: step = 1.0        # finer steps after the first reversals
    last_dir = d
    level = max(0.0, level + d * step)
levels = np.array(levels); hits = np.array(hits)
est = levels[reversals[2:]].mean()               # average level at the later reversals

fig, ax = plt.subplots(figsize=(16, 7), dpi=100)
ax.step(np.arange(60), levels, where="post", color=INK2, linewidth=1.5, zorder=2)
ax.scatter(np.arange(60)[hits], levels[hits], s=70, color=C[2], zorder=3, label="difference detected")
ax.scatter(np.arange(60)[~hits], levels[~hits], s=70, color=C[1], marker="x", linewidth=2.2, zorder=3, label="not detected")
ax.scatter(reversals, levels[reversals], s=260, facecolor="none", edgecolor=C[3], linewidth=1.8, zorder=4, label="reversal")
true707 = true_jnd + slope / 2 * np.log(0.707 / 0.293)   # the level the simulated listener detects 70.7% of the time
ax.axhline(true707, color=C[0], linewidth=1.3, linestyle="--")
ax.axhline(est, color=C[3], linewidth=1.3, linestyle=":")
ax.text(1, 8.3, f"hidden 70.7%-detection level of the simulated listener: {true707:.2f}", color=C[0], fontsize=12)
ax.text(1, 7.6, f"estimate from the later reversals: {est:.2f}  (a 2-down-1-up staircase converges on the 70.7% point)", color=C[3], fontsize=12)
ax.set_xlabel("trial"); ax.set_ylabel("size of the difference presented (arbitrary units)")
ax.set_title("A staircase finds the just-noticeable difference: harder after two hits, easier after a miss (simulated listener)")
ax.set_ylim(0, 11.5); ax.legend(loc="upper right", frameon=False)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("/home/dima/Work/chooseyourcodec/docs/map/zoo/img/perceptual-models/jnd-staircase.png")
print("ok", est, reversals)
