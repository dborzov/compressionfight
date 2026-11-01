"""Figures for docs/map/zoo/side-information.md.

Run:  uv run --with matplotlib --with numpy python docs/map/zoo/src/side-information/figures.py
Writes 1600px-wide PNGs to docs/map/zoo/img/side-information/.
Only numpy, matplotlib and the standard library (json, zlib) are used.
"""
import os, json, zlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "img", "side-information")
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


# ---------------------------------------------------------------- the toy corpus and the toy LZ77 (same as in the article)
def event(rng):
    return json.dumps({"event": "order.placed", "version": 3,
        "customer_id": int(rng.integers(10**7, 10**8)),
        "sku": "SKU-%05d" % rng.integers(0, 99999),
        "quantity": int(rng.integers(1, 9)),
        "currency": "EUR", "amount": round(float(rng.uniform(5, 500)), 2),
        "warehouse": ["AMS-1", "FRA-2", "MAD-1"][rng.integers(0, 3)],
        "ts": "2016-03-%02dT%02d:%02d:%02dZ" % (rng.integers(1, 29), rng.integers(0, 24), rng.integers(0, 60), rng.integers(0, 60)),
        "trace_id": "%016x" % rng.integers(0, 2**63)})


def lz77(data, window=b"", minlen=4, maxlen=255):
    buf = window + data; i = len(window); out = []
    while i < len(buf):
        best = (0, 0)
        for j in range(max(0, i - 65535), i):
            L = 0
            while L < maxlen and i + L < len(buf) and buf[j + L] == buf[i + L]:
                L += 1
            if L > best[1]:
                best = (i - j, L)
        if best[1] >= minlen:
            out.append(("M",) + best); i += best[1]
        else:
            out.append(("L", buf[i])); i += 1
    return out


def cost(tokens):
    lits = sum(t[0] == "L" for t in tokens); mats = sum(t[0] == "M" for t in tokens)
    return lits + 3 * mats + (len(tokens) + 7) // 8


def deflate(data, zdict=b"", strategy=zlib.Z_DEFAULT_STRATEGY):
    kw = {"zdict": zdict} if zdict else {}
    c = zlib.compressobj(9, zlib.DEFLATED, -15, 9, strategy, **kw)
    return len(c.compress(data) + c.flush())


# ---------------------------------------------------------------- figure 1: bytes out vs dictionary size, and header cost
def fig_dict_size():
    rng = np.random.default_rng(2016)
    msgs = [event(rng) for _ in range(64)]
    m = msgs[0].encode()
    sizes, toy, real = [], [], []
    for k in (0, 1, 2, 4, 8, 16, 32):
        d = "".join(msgs[1:1 + k]).encode()
        sizes.append(len(d)); toy.append(cost(lz77(m, d))); real.append(deflate(m, d))
    batch = "\n".join(msgs).encode()
    per_msg_batch = deflate(batch) / len(msgs)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7.2), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.array(sizes, float); x[0] = 60   # place "no dictionary" at the left edge of the log axis
    ax.semilogx(x, toy, "o-", color=C_PRIOR, lw=2.5, ms=9, label="toy LZ77 (3-byte matches, 1-byte literals)")
    ax.semilogx(x, real, "s-", color=C_ENT, lw=2.5, ms=9, label="real raw DEFLATE with the same bytes as zdict")
    ax.axhline(len(m), color=C_QUANT, lw=1.5, ls="--")
    ax.text(70, len(m) + 4, f"the message itself: {len(m)} bytes", color=C_QUANT)
    ax.axhline(per_msg_batch, color=C_PRED, lw=1.5, ls="--")
    ax.text(70, per_msg_batch + 4, f"64 messages compressed as one stream: {per_msg_batch:.0f} bytes each", color=C_PRED)
    ax.set_xticks(x); ax.set_xticklabels(["none"] + [f"{s:,}" for s in sizes[1:]])
    ax.set_xlabel("dictionary installed at the receiver (bytes; the previous k messages)")
    ax.set_ylabel("bytes on the wire for one 220-byte JSON event")
    ax.set_title("What the decoder already knows is what the wire does not carry")
    ax.set_ylim(0, 250); ax.grid(True); ax.legend(loc="center right", fontsize=11)
    for xi, yi in zip(x, toy):
        ax.annotate(f"{yi}", (xi, yi), xytext=(0, 9), textcoords="offset points", ha="center", color=C_PRIOR, fontsize=11)

    # header cost: fixed vs dynamic Huffman tables as the payload grows
    text = "\n".join(msgs).encode()
    lens = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
    fx = [deflate(text[:L], strategy=zlib.Z_FIXED) for L in lens]
    dy = [deflate(text[:L]) for L in lens]
    ax2.plot(lens, fx, "o-", color=C_PRIOR, lw=2.5, label="fixed tables (in the spec: 0 header bytes)")
    ax2.plot(lens, dy, "s-", color=C_ENT, lw=2.5, label="dynamic tables (measured, then shipped in the header)")
    ax2.plot(lens, lens, color=GRID, lw=1, ls="--"); ax2.text(700, 740, "no compression", color=DIM, rotation=36)
    cross = next((L for L, a, b in zip(lens, fx, dy) if b < a), None)
    if cross:
        ax2.axvline(cross, color=DIM, ls=":", lw=1)
        ax2.text(cross + 10, 30, f"dynamic wins\nfrom ~{cross} B", color=FG, fontsize=11)
    ax2.set_xlabel("payload length (bytes of JSON events)")
    ax2.set_ylabel("DEFLATE output (bytes)")
    ax2.set_title("Semi-static tables pay a header: bad for tiny inputs")
    ax2.grid(True); ax2.legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    save(fig, "dict_size.png")
    return sizes, toy, real, per_msg_batch, list(zip(lens, fx, dy))


# ---------------------------------------------------------------- figure 2: the knowledge spectrum
def fig_spectrum():
    fig, ax = plt.subplots(figsize=(16, 7.6))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    stations = [
        ("ADAPTIVE", "learned from the\nstream itself", "PPM, CABAC, PAQ,\nadaptive arithmetic",
         "0 bytes installed\ncold-start tax on\nthe first kilobytes", C_PRED),
        ("HEADER", "measured, shipped\nin the stream", "DEFLATE dynamic trees,\nFLAC LPC coefficients,\nJPEG's own tables",
         "0 bytes installed\n~tens of bytes\nper block", C_ENT),
        ("STANDARD TABLES", "constants in\nthe spec", "DEFLATE fixed block,\nJPEG Annex K,\nMP3's 32 tables, μ-law",
         "a few KB, versioned\nwith the format,\nnever optimal", C_PRIOR),
        ("SHARED DICTIONARY", "trained on a corpus,\ndistributed out of band", "zstd --train (COVER),\nBrotli 120 KB,\nHTTP compression dicts",
         "100 KB per schema\nwith an ID;\n3–5× on small JSON", C_PRIOR),
        ("CODEBOOK / BASIS", "fitted centroids\nor axes", "Faiss PQ codebooks,\nCELP, KVTC PCA (2.4%),\nSpinQuant (0.26%)",
         "MB, per model\nor per index;\nrefit when data moves", C_TRANS),
        ("TRAINED DECODER", "the prior is the\nwhole network", "SoundStream, EnCodec,\nlearned image codecs,\nLLM + arithmetic coder",
         "20 MB+ per codec,\n750 B/s on the wire,\nmatched pair only", C_QUANT),
    ]
    xs = np.linspace(1.3, 14.7, len(stations))
    ax.add_patch(FancyArrowPatch((0.3, 4.6), (15.8, 4.6), arrowstyle="-|>", mutation_scale=30, color=DIM, lw=3))
    for (name, what, ex, cost, col), x in zip(stations, xs):
        ax.add_patch(FancyBboxPatch((x - 1.22, 3.9), 2.44, 1.4, boxstyle="round,pad=0.08", facecolor="#ebdfc8", edgecolor=col, lw=2.5))
        ax.text(x, 4.95, name, ha="center", va="center", fontsize=11.5, fontweight="bold", color=col)
        ax.text(x, 4.3, what, ha="center", va="center", fontsize=10.5, color=FG)
        ax.text(x, 6.3, ex, ha="center", va="center", fontsize=10.5, color=FG)
        ax.text(x, 2.6, cost, ha="center", va="center", fontsize=10.5, color=DIM)
    ax.text(0.3, 7.6, "examples", fontsize=12, color=DIM, fontweight="bold")
    ax.text(0.3, 1.6, "what is installed at the receiver", fontsize=12, color=DIM, fontweight="bold")
    ax.annotate("", xy=(15.6, 0.8), xytext=(0.5, 0.8), arrowprops=dict(arrowstyle="->", color=C_PRIOR, lw=2.5))
    ax.text(8, 0.35, "more bytes installed at the receiver, more versioning, more specificity  →  fewer bytes on the wire, and small inputs become compressible",
            ha="center", fontsize=12.5, color=C_PRIOR)
    ax.set_title("The knowledge spectrum: where the decoder's half of the story lives", pad=14)
    save(fig, "spectrum.png")


# ---------------------------------------------------------------- figure 3: cross-entropy vs mismatch
def fig_cross_entropy():
    P = np.array([0.5, 0.25, 0.125, 0.125])
    Q = np.array([0.9, 0.05, 0.03, 0.02])
    H = -(P * np.log2(P)).sum()
    a = np.linspace(0, 1, 201)
    HPQ = np.array([-(P * np.log2((1 - t) * Q + t * P)).sum() for t in a])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.15]})
    labels = ["A", "B", "C", "D"]; xi = np.arange(4)
    ax.bar(xi - 0.2, -np.log2(P), width=0.4, color=C_PRED, label="price under the right prior, $-\\log_2 P$")
    ax.bar(xi + 0.2, -np.log2(Q), width=0.4, color=C_QUANT, label="price under the shipped prior, $-\\log_2 Q$")
    for i in range(4):
        ax.text(i - 0.2, -np.log2(P[i]) + 0.08, f"{-np.log2(P[i]):.2f}", ha="center", color=C_PRED, fontsize=11)
        ax.text(i + 0.2, -np.log2(Q[i]) + 0.08, f"{-np.log2(Q[i]):.2f}", ha="center", color=C_QUANT, fontsize=11)
    ax.set_xticks(xi); ax.set_xticklabels([f"{l}\nP={p}\nQ={q}" for l, p, q in zip(labels, P, Q)])
    ax.set_ylabel("bits charged for the symbol")
    ax.set_title(f"Average: H(P) = {H:.2f}  vs  H(P,Q) = {-(P*np.log2(Q)).sum():.2f} bits/symbol")
    ax.grid(True, axis="y"); ax.legend(fontsize=11, loc="upper left")

    ax2.plot(a, HPQ, color=C_PRIOR, lw=3)
    ax2.axhline(H, color=C_PRED, lw=1.5, ls="--"); ax2.text(0.02, H - 0.09, f"floor H(P) = {H:.3f}", color=C_PRED)
    ax2.axhline(2.0, color=DIM, lw=1.5, ls=":"); ax2.text(0.02, 2.03, "fixed 2-bit code, no prior at all", color=DIM)
    ax2.fill_between(a, H, HPQ, color=C_QUANT, alpha=0.25)
    ax2.text(0.45, 2.25, "$D_{KL}(P\\,\\|\\,Q_t)$: the mismatch tax", color=C_QUANT, fontsize=13)
    ax2.set_xlabel("$t$ in  $Q_t = (1-t)\\,Q + t\\,P$   (0 = trained on the wrong data, 1 = trained on this data)")
    ax2.set_ylabel("bits per symbol actually paid, $H(P, Q_t)$")
    ax2.set_title("A prior saves exactly the information it shares with the data")
    ax2.set_ylim(1.6, 2.6); ax2.grid(True)
    fig.tight_layout()
    save(fig, "cross_entropy.png")


if __name__ == "__main__":
    out = fig_dict_size()
    print("dict sizes", out[0]); print("toy LZ77", out[1]); print("deflate", out[2]); print("batch per msg", out[3])
    print("fixed vs dynamic", out[4])
    fig_spectrum()
    fig_cross_entropy()
