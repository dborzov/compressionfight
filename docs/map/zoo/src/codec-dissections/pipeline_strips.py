"""Pipeline strip diagrams for the twelve dissections, generated from the game's squads
(CYC.CODECS in js/data.js, exported to cyc_data.json by a one-line node script, see README note below).
Each fighter is a box coloured by its chair (CYC.ROLES[role].color), in encoder pipeline order:
transform -> predict -> quantize -> entropy. The Con Artist (perceptual) and the Inside Man (prior)
sit on a second row because data does not flow through them: one steers the quantizer, the other is
held by the decoder before the first byte arrives.
Run: uv run --with matplotlib --with numpy python pipeline_strips.py

cyc_data.json was produced with:
  node -e "global.window=global; require('./js/data.js'); ...JSON.stringify({roles:CYC.ROLES, codecs:CYC.CODECS, ...})"
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG, FG, DIM = "#0b0714", "#ece6f7", "#8a81a3"
here = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(here, "cyc_data.json")))
ROLES, FIGHTERS = data["roles"], data["fighters"]
CODECS = {c["id"]: c for c in data["codecs"]}
FLOW = ["transform", "predict", "quantize", "entropy"]
SIDE = ["perceptual", "prior"]
# short labels for the boxes (the game's `real` strings are too long for a strip)
SHORT = {
    "lz77": "LZ77 match", "lzw": "LZW", "huffman": "Huffman", "ans": "FSE / tANS", "arith": "arithmetic /\nrange coder",
    "rice": "Golomb-Rice", "rle": "zigzag + RLE", "spatial": "spatial / intra\nprediction", "lpc": "LPC", "delta": "delta",
    "motion": "motion comp.", "cm": "context mixing", "neural_prior": "learned prob.\nmodel", "anchor_delta": "anchor deltas",
    "dct": "block DCT", "mdct": "MDCT", "dwt": "wavelet", "prism": "YCbCr /\nmid-side", "pca": "PCA / OPQ\nrotation", "bwt": "BWT + MTF",
    "neural_enc": "conv encoder", "unrope": "undo RoPE", "svd_lowrank": "low-rank SVD", "mla": "MLA latent",
    "uniform": "uniform Δ\n(+ dead zone)", "qmatrix": "quant table", "companding": "power-law\nquantizer", "vq": "VQ / PVQ",
    "rvq": "residual VQ", "pq": "product\nquantizer", "blockint": "int4 blocks\n+ scales", "waterline": "bit allocation",
    "fp4": "NVFP4", "escape": "outlier list", "evict": "eviction", "coupled": "coupled VQ",
    "psychoacoustic": "psychoacoustic\nmasking", "chroma": "chroma 4:2:0\n/ CSF", "gan": "adversarial\nloss", "sinkguard": "sinks + window",
    "spotlight": "top-k selector", "tables": "standard\ntables", "dictionary": "trained\ndictionary", "codebook": "codebooks",
    "decoder_net": "decoder\nweights", "basis": "PCA basis", "tiered": "CPU copy",
}
# pipeline-order overrides where the squad's role order is not the codec's data order
ORDER = {"jpeg": ["prism", "dct", "qmatrix", "rle", "huffman"],
         "bzip2": ["bwt", "rle", "huffman"],
         "flac": ["prism", "lpc", "rice"],
         "png": ["spatial", "lz77", "huffman"],
         "h264": ["prism", "motion", "spatial", "dct", "uniform", "arith"],
         "webp": ["prism", "spatial", "dct", "uniform", "arith"],
         "opus": ["lpc", "mdct", "vq", "arith"],
         "zstd": ["lz77", "ans", "huffman"],
         "mp3": ["mdct", "companding", "huffman"]}
EMPTY = {"soundstream": ("entropy", "no Packer:\nfixed-rate tokens"), "faiss": ("entropy", "no Packer:\nfixed-length codes"),
         "gguf": ("entropy", "no Packer:\nkernel decodes")}

def rbox(ax, x, y, w, h, color, text, sub=None, dashed=False, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=color if not dashed else BG,
                       ec=color, lw=2.2, ls="--" if dashed else "-", alpha=alpha)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), text, ha="center", va="center", fontsize=13,
            color=("#120a1e" if not dashed else color), fontweight="bold", linespacing=1.1)
    if sub:
        ax.text(x + w / 2, y + h * 0.2, sub, ha="center", va="center", fontsize=10.5, color="#120a1e" if not dashed else DIM)

def strip(ax, cid, title_prefix=""):
    c = CODECS[cid]; squad = c["squad"]
    flow_ids = ORDER.get(cid) or [f for r in FLOW for f in squad.get(r, [])]
    side = [(r, f) for r in SIDE for f in squad.get(r, [])]
    n = len(flow_ids) + (1 if cid in EMPTY else 0)
    ax.set_xlim(0, 16); ax.set_ylim(0, 4.3); ax.axis("off")
    ax.text(0.1, 4.05, f"{title_prefix}{c['name']}  ·  {c['year']}", fontsize=17, color=FG, fontweight="bold", va="center")
    ax.text(0.1, 3.55, c["notes"], fontsize=11.5, color=DIM, va="center")
    # data flow row
    w, gap, x0, y = 2.2, 0.55, 0.1, 1.6
    total = n * w + (n - 1) * gap + 2 * 0.9
    scale = min(1.0, 15.8 / total)
    w, gap = w * scale, gap * scale
    x = x0
    rbox(ax, x, y, 0.9 * scale, 1.2, "#2a2140", "data", dashed=False); x += 0.9 * scale + gap
    for fid in flow_ids:
        role = FIGHTERS[fid]["role"]; col = ROLES[role]["color"]
        rbox(ax, x, y, w, 1.2, col, SHORT.get(fid, fid), sub=FIGHTERS[fid]["name"].title())
        ax.add_patch(FancyArrowPatch((x - gap + 0.06, y + 0.6), (x - 0.04, y + 0.6), arrowstyle="-|>", mutation_scale=16, color=FG, lw=1.5))
        x += w + gap
    if cid in EMPTY:
        role, label = EMPTY[cid]
        rbox(ax, x, y, w, 1.2, ROLES[role]["color"], label, dashed=True)
        ax.add_patch(FancyArrowPatch((x - gap + 0.06, y + 0.6), (x - 0.04, y + 0.6), arrowstyle="-|>", mutation_scale=16, color=FG, lw=1.5))
        x += w + gap
    ax.add_patch(FancyArrowPatch((x - gap + 0.06, y + 0.6), (x - 0.04, y + 0.6), arrowstyle="-|>", mutation_scale=16, color=FG, lw=1.5))
    rbox(ax, x, y, 0.9 * scale, 1.2, "#2a2140", "bits")
    # side row: perceptual steers, prior is held
    if side:
        sx = 0.1; sw = 2.0
        for role, fid in side:
            col = ROLES[role]["color"]
            rbox(ax, sx, 0.15, sw, 0.95, col, SHORT.get(fid, fid), alpha=0.9)
            tag = "steers the quantizer" if role == "perceptual" else "held by the decoder"
            ax.text(sx + sw + 0.12, 0.62, tag, fontsize=10.5, color=col, va="center")
            sx += sw + 2.0
    else:
        ax.text(0.1, 0.62, "no Con Artist, no Inside Man: the stream describes itself", fontsize=11, color=DIM, va="center")

groups = [("deflate", ["deflate"]), ("png", ["png"]), ("bzip2", ["bzip2"]), ("jpeg", ["jpeg"]),
          ("jpeg2000_webp", ["jpeg2000", "webp"]), ("flac", ["flac"]), ("mp3", ["mp3"]), ("opus", ["opus"]),
          ("soundstream", ["soundstream"]), ("h264", ["h264"]), ("zstd", ["zstd"]), ("gguf_faiss", ["gguf", "faiss"])]
out_dir = os.path.join(here, "..", "..", "img", "codec-dissections"); os.makedirs(out_dir, exist_ok=True)
for name, ids in groups:
    fig, axes = plt.subplots(len(ids), 1, figsize=(16, 3.9 * len(ids)), dpi=100, facecolor=BG)
    axes = [axes] if len(ids) == 1 else list(axes)
    for ax, cid in zip(axes, ids):
        ax.set_facecolor(BG); strip(ax, cid)
    # chair legend
    lx = 8.4
    for r in FLOW + SIDE:
        axes[0].add_patch(FancyBboxPatch((lx, 3.85), 0.28, 0.32, boxstyle="round,pad=0.01", fc=ROLES[r]["color"], ec=ROLES[r]["color"]))
        axes[0].text(lx + 0.36, 4.01, ROLES[r]["codename"].title(), fontsize=9.5, color=DIM, va="center")
        lx += 1.35
    fig.tight_layout(pad=0.6)
    out = os.path.join(out_dir, f"{name}.png"); fig.savefig(out, facecolor=BG); plt.close(fig); print("wrote", os.path.abspath(out))
