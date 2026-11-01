"""Figures for docs/map/zoo/attention-as-audience.md.

Run:  uv run --with matplotlib --with numpy python docs/map/zoo/src/attention-as-audience/figures.py
Writes 1600px-wide PNGs to docs/map/zoo/img/attention-as-audience/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()
from matplotlib.patches import Rectangle

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "img", "attention-as-audience")
os.makedirs(OUT, exist_ok=True)

BG, FG, GRID, DIM = "#f3eadb", "#2a211b", "#b89f78", "#5a4a3d"
C_PERC, C_QUANT, C_PRIOR, C_TRANS, C_PRED, C_ENT = "#b8892e", "#b05a4c", "#9c6b3f", "#246b66", "#6f8f4e", "#6b4a7a"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": GRID, "grid.linestyle": ":", "font.size": 13,
    "axes.titlesize": 16, "axes.titleweight": "bold", "legend.facecolor": BG, "legend.edgecolor": GRID,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- figure 1: attention mass vs rank
def fig_mass_vs_rank():
    rng = np.random.default_rng(3)
    n = 4096
    logits = rng.normal(0.0, 3.0, size=n)      # heavy-tailed softmax: a few keys dominate
    logits[0] += 6.0                             # an attention sink
    logits[-8:] += 2.0                           # a recent window
    w = np.exp(logits - logits.max()); w /= w.sum()
    ws = np.sort(w)[::-1]
    cum = np.cumsum(ws)
    frac = np.arange(1, n + 1) / n

    # the 32-token toy from the article
    toy = np.zeros(32); toy[0] = 0.35; toy[-4:] = 0.075; toy[1:-4] = 0.35 / 27

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7.2), gridspec_kw={"width_ratios": [1.35, 1]})
    ax.semilogx(frac * 100, cum, color=C_PERC, lw=3, label="synthetic softmax: 4,096 keys, logits ~ N(0, 3²), one sink")
    ax.axhline(0.9, color=DIM, lw=1, ls="--")
    ax.axvline(0.5, color=DIM, lw=1, ls="--")
    i90 = int(np.searchsorted(cum, 0.9)) + 1
    ax.plot([i90 / n * 100], [0.9], "o", color=FG, ms=9)
    ax.annotate(f"toy: top {i90} keys ({i90 / n * 100:.1f}%) carry 90% of the mass",
                xy=(i90 / n * 100, 0.9), xytext=(0.25, 0.72), color=FG,
                arrowprops=dict(arrowstyle="->", color=FG))
    ax.plot([0.5], [0.9], "D", color=C_TRANS, ms=10)
    ax.annotate("reported (SpeCache, Llama-3-8B):\n0.5% of keys carry 90%", xy=(0.5, 0.9), xytext=(0.05, 0.98),
                color=C_TRANS, arrowprops=dict(arrowstyle="->", color=C_TRANS), va="top")
    ax.plot([20], [0.8], "D", color=C_PRED, ms=10)
    ax.annotate("reported (NVIDIA blog): ~20% of\ntokens collect ~80% of attention", xy=(20, 0.8), xytext=(3, 0.3),
                color=C_PRED, arrowprops=dict(arrowstyle="->", color=C_PRED))
    ax.set_xlabel("keys kept, ranked by attention weight (% of context, log scale)")
    ax.set_ylabel("cumulative attention mass")
    ax.set_ylim(0, 1.02)
    ax.set_title("The audience watches a tiny part of the stage")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=11)

    cols = [C_PERC] + [DIM] * 27 + [C_TRANS] * 4
    ax2.bar(np.arange(32), toy, color=cols, width=0.85)
    ax2.set_xlabel("token position (32-token toy)")
    ax2.set_ylabel("attention weight  $w_i$")
    ax2.set_title("32-token toy: sink 0.35, middle 0.35, recent 0.30", fontsize=14)
    ax2.text(0.6, 0.33, "sink", color=C_PERC, fontsize=13, fontweight="bold")
    ax2.text(26.2, 0.085, "recent\nwindow", color=C_TRANS, fontsize=12, ha="center")
    ax2.text(13, 0.03, "27 middle tokens: 0.013 each", color=FG, fontsize=12, ha="center")
    ax2.grid(True, axis="y")
    fig.tight_layout()
    save(fig, "mass_vs_rank.png")


# ---------------------------------------------------------------- figure 2: paged grid before / after compaction
def draw_pages(ax, n_pages, page, alive, title, order=None):
    """alive: boolean per token slot (token i lives at slot i unless order maps it)."""
    n = n_pages * page
    cols = page
    for p in range(n_pages):
        y = n_pages - 1 - p
        page_alive = alive[p * page:(p + 1) * page].any()
        ax.add_patch(Rectangle((-0.15, y - 0.15), cols + 0.3, 0.8, fill=True,
                               facecolor="#ebdfc8" if page_alive else BG,
                               edgecolor=C_QUANT if page_alive else GRID, lw=2.2, ls="-" if page_alive else "--"))
        for s in range(page):
            i = p * page + s
            c = C_PERC if (alive[i] and i == 0 and order is None) else (C_TRANS if alive[i] else "#e0d2b6")
            ax.add_patch(Rectangle((s, y), 0.72, 0.5, facecolor=c, edgecolor=BG))
            if alive[i] and order is not None and n_pages <= 2:
                ax.text(s + 0.36, y + 0.25, str(order[i]), ha="center", va="center", fontsize=9, color=BG, fontweight="bold")
            elif alive[i] and n_pages <= 2:
                ax.text(s + 0.36, y + 0.25, str(i), ha="center", va="center", fontsize=9, color=BG, fontweight="bold")
        ax.text(cols + 0.4, y + 0.25, "allocated" if page_alive else "FREE", va="center", fontsize=11,
                color=C_QUANT if page_alive else C_PRED, fontweight="bold")
    ax.set_xlim(-0.5, cols + 2.6)
    ax.set_ylim(-0.4, n_pages)
    ax.axis("off")
    ax.set_title(title, fontsize=14)


def fig_paged():
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 2.2], hspace=0.35, wspace=0.15)

    # toy: 16 tokens, 2 pages of 8
    page, n_pages = 8, 2
    alive0 = np.ones(16, bool)
    keep = np.array([0, 7, 8, 15])
    alive1 = np.zeros(16, bool); alive1[keep] = True
    alive2 = np.zeros(16, bool); alive2[:4] = True
    order2 = {0: 0, 1: 7, 2: 8, 3: 15}
    ax = fig.add_subplot(gs[0, 0]); draw_pages(ax, n_pages, page, alive0, "16 tokens, 2 pages of 8")
    ax = fig.add_subplot(gs[0, 1]); draw_pages(ax, n_pages, page, alive1, "evict 12 of 16: survivors 0, 7, 8, 15\npages freed: 0")
    ax = fig.add_subplot(gs[0, 2]); draw_pages(ax, n_pages, page, alive2, "compact (3 copies): survivors in page 0\npages freed: 1", order=order2)

    # scaled up: 16 pages of 16 = 256 tokens, importance heavy-hitter pattern, keep 64
    rng = np.random.default_rng(11)
    n, page, n_pages = 256, 16, 16
    imp = rng.pareto(1.2, size=n) + 0.01
    imp[:4] = imp.max() * 10          # sinks always important
    keepers = np.argsort(-imp)[:64]
    aliveA = np.ones(n, bool)
    aliveB = np.zeros(n, bool); aliveB[keepers] = True
    aliveC = np.zeros(n, bool); aliveC[:64] = True
    freedB = n_pages - len(np.unique(keepers // page))
    ax = fig.add_subplot(gs[1, 0]); draw_pages(ax, n_pages, page, aliveA, "256 tokens in 16 pages of 16")
    ax = fig.add_subplot(gs[1, 1]); draw_pages(ax, n_pages, page, aliveB, f"evict 192 of 256 by importance score\npages freed: {freedB} of 16")
    ax = fig.add_subplot(gs[1, 2]); draw_pages(ax, n_pages, page, aliveC, "compact the 64 survivors forward\npages freed: 12 of 16", order={i: int(k) for i, k in enumerate(np.sort(keepers))})
    fig.suptitle("Paged memory frees nothing until a page is empty", fontsize=18, fontweight="bold", y=0.98)
    save(fig, "paged_grid.png")
    return freedB


# ---------------------------------------------------------------- figure 3: exact vs 1-bit proxy scores
def fig_proxy():
    rng = np.random.default_rng(0)
    K = rng.normal(size=(32, 16)); q = rng.normal(size=16)
    s = K @ q
    K1 = np.sign(K) * np.abs(K).mean(axis=1, keepdims=True)
    sp = K1 @ q
    ex = set(np.argsort(-s)[:4]); px4 = set(np.argsort(-sp)[:4]); px8 = set(np.argsort(-sp)[:8])
    w = np.exp(s - s.max()); w /= w.sum()

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.scatter(s, sp, s=60, color=DIM, label="the other keys", zorder=2)
    for i in px8 - px4:
        ax.scatter(s[i], sp[i], s=160, facecolors="none", edgecolors=C_TRANS, lw=2, zorder=3)
    ax.scatter([s[i] for i in px4], [sp[i] for i in px4], s=180, facecolors="none", edgecolors=C_TRANS, lw=2.5,
               label="proxy's top-4 (ring); its top-8 (thin ring)", zorder=3)
    ax.scatter([s[i] for i in ex], [sp[i] for i in ex], s=90, color=C_PERC, label="exact top-4 (what the softmax actually weights)", zorder=4)
    for i in ex | px4:
        ax.annotate(f"k{i}  w={w[i]:.2f}", (s[i], sp[i]), xytext=(8, 6), textcoords="offset points", fontsize=11)
    lim = [min(s.min(), sp.min()) - 1, max(s.max(), sp.max()) + 1]
    ax.plot(lim, lim, color=GRID, lw=1)
    ax.set_xlabel("exact score  $q \\cdot k_i$  (16-dim keys, fp32)")
    ax.set_ylabel("proxy score  $q \\cdot \\hat k_i$  (1 bit per coordinate + one scale per key)")
    ax.set_title(f"Ranking with a 1-bit stand-in: proxy top-4 overlaps exact top-4 on {len(ex & px4)} of 4; "
                 f"exact top-4 ⊂ proxy top-8: {ex <= px8}")
    ax.grid(True); ax.legend(loc="upper left", fontsize=12)
    fig.tight_layout()
    save(fig, "proxy_topk.png")


if __name__ == "__main__":
    fig_mass_vs_rank()
    print("pages freed by eviction in the 256-token grid:", fig_paged())
    fig_proxy()
