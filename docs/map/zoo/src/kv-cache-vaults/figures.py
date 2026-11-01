"""Figures for docs/map/zoo/kv-cache-vaults.md.

Run:  uv run --with matplotlib --with numpy python docs/map/zoo/src/kv-cache-vaults/figures.py
Writes 1600px-wide PNGs to docs/map/zoo/img/kv-cache-vaults/.
Numbers: bytes/token = 2 x layers x kv_heads x head_dim x 2 B (derived from public configs);
MLA = 61 x 576 x 2 B; GLM-5.1 "about 100 KB" and DeepSeek-V4 ~10 KiB are reported / derived from
the research notes in docs/research/kv_systems.md and are labelled as such.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "img", "kv-cache-vaults")
os.makedirs(OUT, exist_ok=True)

BG, FG, GRID, DIM = "#f3eadb", "#2a211b", "#b89f78", "#5a4a3d"
C_PERC, C_QUANT, C_PRIOR, C_TRANS, C_PRED, C_ENT = "#b8892e", "#b05a4c", "#9c6b3f", "#246b66", "#6f8f4e", "#6b4a7a"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": GRID, "grid.linestyle": ":", "font.size": 13,
    "axes.titlesize": 16, "axes.titleweight": "bold", "legend.facecolor": BG, "legend.edgecolor": GRID,
})
KiB = 1024


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print("wrote", path)


def bpt(layers, kv_heads, head_dim, nbytes=2):
    return 2 * layers * kv_heads * head_dim * nbytes


# ---------------------------------------------------------------- figure 1: bytes per token
def fig_bytes_per_token():
    rows = [
        ("Llama-3-70B if it kept all 64 heads (MHA)", bpt(80, 64, 128), DIM, "derived"),
        ("Llama-2-7B  (MHA, 32 KV heads)", bpt(32, 32, 128), C_QUANT, "derived"),
        ("Llama-3-70B (GQA, 8 KV heads)", bpt(80, 8, 128), C_QUANT, "derived; the 40 GiB at 128k"),
        ("Llama-3-8B  (GQA, 8 KV heads)", bpt(32, 8, 128), C_QUANT, "derived"),
        ("GLM-5.1 (DSA, 78 layers)", 100_000, C_PRIOR, "reported: 'about 100 KB'"),
        ("DeepSeek-V3 (MLA, 61 x 576, bf16)", 61 * 576 * 2, C_TRANS, "derived from config"),
        ("DeepSeek-V3.2 (MLA, FP8 latent, 656 B/entry)", 61 * 656, C_TRANS, "656 B per entry reported"),
        ("Llama-3-70B at int2 (KIVI-class, ~3 bits effective)", bpt(80, 8, 128) * 3 / 16, C_PERC, "derived"),
        ("DeepSeek-V4 (CSA/HCA, bf16)", 9.62 * KiB**3 / 1_000_000, C_PRED, "derived: 9.62 GiB per 1M tokens"),
    ]
    rows.sort(key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(16, 8.4))
    y = np.arange(len(rows))
    vals = np.array([r[1] for r in rows]) / KiB
    ax.barh(y, vals, color=[r[2] for r in rows], height=0.7)
    ax.set_xscale("log")
    for yi, (name, v, c, note) in zip(y, rows):
        label = f"{v / KiB:,.1f} KiB" if v >= KiB else f"{v:.0f} B"
        ax.text(v / KiB * 1.08, yi, f"{label}   ({note})", va="center", fontsize=11.5, color=FG)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(5, 30000)
    ax.set_xlabel("KV-cache bytes per token, all layers (log scale)")
    ax.set_title("One tensor, many sizes: bytes per token decided at design time, then by the codec")
    ax.grid(True, axis="x")
    ax2 = ax.secondary_xaxis("top", functions=(lambda k: k * KiB * 131072 / KiB**3, lambda g: g * KiB**3 / (KiB * 131072)))
    ax2.set_xlabel("GiB for a 131,072-token context", color=DIM)
    ax2.tick_params(colors=DIM)
    fig.tight_layout()
    save(fig, "bytes_per_token.png")


# ---------------------------------------------------------------- figure 2: the stat sheets side by side
def fig_stat_sheets():
    # req stats copied from js/data.js CYC.HEISTS (level 5 + kvcache)
    jobs = [
        ("THE ARCHITECT'S\nCUT", "GQA / MLA / CSA", dict(latency=0, encBudget=0, decBudget=0, ra=5, bandwidth=4, decMem=1), False, True, C_TRANS,
         "Lens only,\ntrained with\nthe model"),
        ("THE CONTEXT\nWINDOW", "KIVI / FP8 / QuaRot", dict(latency=0, encBudget=0, decBudget=0, ra=5, bandwidth=3, decMem=1), False, False, C_QUANT,
         "resident in HBM;\nthe Packer's\nchair is empty"),
        ("THE ONE-BIT\nVAULT", "CQ / CommVQ", dict(latency=0, encBudget=1, decBudget=1, ra=5, bandwidth=5, decMem=1), False, True, C_QUANT,
         "resident;\ncalibrated\ncodebooks allowed"),
        ("THE WAREHOUSE\nRUN", "HiSparse / ShadowKV\n/ SpeCache", dict(latency=1, encBudget=2, decBudget=1, ra=5, bandwidth=5, decMem=2), False, True, C_PRIOR,
         "PCIe is the channel;\na lossless copy\none hop away"),
        ("THE COLD\nSTORAGE", "KVTC / NVFP4\ncold pages", dict(latency=3, encBudget=2, decBudget=2, ra=1, bandwidth=5, decMem=3), False, True, C_ENT,
         "decode once;\nthe Packer\nsits down"),
        ("THE\nCOURIER", "CacheGen", dict(latency=2, encBudget=3, decBudget=2, ra=2, bandwidth=5, decMem=3), False, True, C_ENT,
         "a network:\nevery byte\nis latency"),
        ("THE\nPURGE", "SnapKV / StreamingLLM\n/ TriAttention", dict(latency=0, encBudget=1, decBudget=0, ra=5, bandwidth=3, decMem=1), False, False, C_PERC,
         "two-level quantizer;\nfrees pages only\nafter compaction"),
    ]
    keys = ["latency", "encBudget", "decBudget", "ra", "bandwidth", "decMem"]
    names = ["latency\ntolerance", "encoder\nbudget", "decoder\nbudget", "random\naccess", "bandwidth\npressure", "decoder\nmemory"]
    fig, axes = plt.subplots(1, len(jobs), figsize=(16, 8), sharey=True)
    for ax, (title, codec, req, lossless, prior, col, note) in zip(axes, jobs):
        v = [req[k] for k in keys]
        ax.barh(np.arange(6), v, color=col, height=0.62)
        ax.barh(np.arange(6), [5] * 6, color="#ebdfc8", height=0.62, zorder=0)
        for yi, vi in enumerate(v):
            ax.text(5.1, yi, str(vi), va="center", ha="left", fontsize=10, color=DIM)
        ax.set_xlim(0, 5.6); ax.set_yticks(np.arange(6)); ax.set_yticklabels(names, fontsize=10.5)
        ax.invert_yaxis(); ax.set_xticks([])
        for s in ("top", "right", "bottom"): ax.spines[s].set_visible(False)
        ax.set_title(title, fontsize=11.5, color=col, pad=30)
        ax.text(2.6, -1.0, codec, ha="center", va="bottom", fontsize=9, color=FG)
        ax.text(2.6, 6.0, f"lossless: {'yes' if lossless else 'no'}\nside info: {'allowed' if prior else 'no'}", ha="center", va="top", fontsize=9.5, color=DIM)
        ax.text(2.6, 7.3, note, ha="center", va="top", fontsize=9.5, color=FG)
        ax.set_ylim(9.2, -1.9)
    fig.suptitle("Same bytes, seven stat sheets (0 = none, 5 = extreme): the job, not the data, picks the chairs", fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "stat_sheets.png")


# ---------------------------------------------------------------- figure 3: tiers and bandwidths
def fig_tiers():
    cache = bpt(80, 8, 128) * 131072          # 40 GiB
    tiers = [
        ("HBM (H200)", 4.8e12, C_QUANT, "every decode step re-reads the cache from here"),
        ("NVLink-C2C (Grace Hopper CPU <-> GPU)", 0.9e12, C_TRANS, "host DRAM at ~900 GB/s"),
        ("PCIe Gen5 x16, nominal", 64e9, C_PRIOR, "host DRAM, shared by every request on the GPU"),
        ("PCIe, GPU gather kernel (ESS measured)", 37e9, C_PRIOR, "2,048 scattered 656-byte entries per layer"),
        ("NVMe SSD", 6e9, C_ENT, "cold prefix caches"),
        ("datacenter network, 10 Gbit/s", 1.25e9, C_ENT, "prefix caches between nodes"),
        ("PCIe, cudaMemcpyAsync per entry (ESS measured)", 0.79e9, C_PERC, "the scattered-copy trap"),
    ]
    fig, ax = plt.subplots(figsize=(16, 8.2))
    y = np.arange(len(tiers))
    t_full = np.array([cache / bw for _, bw, _, _ in tiers])
    ax.barh(y, t_full * 1e3, color=[c for _, _, c, _ in tiers], height=0.62)
    ax.set_xscale("log")
    for yi, (name, bw, c, note), t in zip(y, tiers, t_full):
        tl = f"{t*1e3:.1f} ms" if t < 1 else (f"{t:.1f} s" if t < 60 else f"{t/60:.1f} min")
        ax.text(t * 1e3 * 1.15, yi, f"{tl}   ({note})", va="center", fontsize=11.5, color=FG)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}\n{bw/1e12:.2f} TB/s" if bw >= 1e12 else f"{n}\n{bw/1e9:.3g} GB/s" for n, bw, _, _ in tiers], fontsize=11)
    ax.invert_yaxis()
    ax.axvline(30, color=C_PRED, lw=2, ls="--"); ax.text(33, 6.75, "30 ms: one decode\nstep's deadline", color=C_PRED, fontsize=11.5, va="center")
    ax.axvline(3098, color=DIM, lw=2, ls=":"); ax.text(3300, 1.0, "3.1 s: recompute a batch of prefixes\n(KVTC's TTFT baseline, 12B model)", color=DIM, fontsize=11, va="center")
    ax.set_xlim(3, 2e6); ax.set_ylim(7.3, -0.6)
    ax.set_xlabel("time to move the whole 40 GiB Llama-3-70B 128k-token cache through the pipe (log scale)")
    ax.set_title("Five pipes, four orders of magnitude: the tier decides the codec")
    ax.grid(True, axis="x")
    fig.tight_layout()
    save(fig, "tiers.png")


if __name__ == "__main__":
    fig_bytes_per_token()
    fig_stat_sheets()
    fig_tiers()
