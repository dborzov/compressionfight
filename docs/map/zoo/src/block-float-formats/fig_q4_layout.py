"""Figure: byte layouts of GGUF Q4_0, Q8_0 and NVFP4 blocks.
Output: img/block-float-formats/q4-layout.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from _style import apply as _apply; _apply()
from matplotlib.patches import FancyBboxPatch

BG, INK, INK2, MUTED, GRID = "#f3eadb", "#2a211b", "#5a4a3d", "#5a4a3d", "#b89f78"
C = ["#246b66", "#b8892e", "#b05a4c", "#6b4a7a", "#6f8f4e"]
DIM = "#e0d2b6"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "font.size": 13})

fig, ax = plt.subplots(figsize=(16, 7.4), dpi=100)
ax.set_xlim(0, 16); ax.set_ylim(0, 7.4); ax.axis("off")

def cell(x0, y0, w, h, text, fc, ec, fs=10, tc=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                                facecolor=fc, edgecolor=ec, linewidth=1.2))
    if text:
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight=weight)

def row(y, title, subtitle, scale_bytes, data_bytes, scale_label, data_label, unit, col, total_w=11.2, x0=2.4):
    ax.text(0.2, y + 0.42, title, color=INK, fontsize=14, fontweight="bold", va="center")
    ax.text(0.2, y + 0.08, subtitle, color=INK2, fontsize=11, va="center")
    total = scale_bytes + data_bytes
    bw = total_w / total
    for b in range(scale_bytes):
        cell(x0 + b * bw, y, bw * 0.96, 0.78, scale_label if b == 0 else "", C[3], C[3], fs=11, tc="#f3eadb", weight="bold")
    for b in range(data_bytes):
        cell(x0 + (scale_bytes + b) * bw, y, bw * 0.96, 0.78, data_label(b), DIM, col, fs=9)
    ax.text(x0 + total_w + 0.15, y + 0.39, unit, color=col, fontsize=12, va="center")

# Q4_0
row(5.9, "Q4_0", "block of 32 weights", 2, 16, "d", lambda b: f"{b}|{b+16}", "18 bytes / 32 = 4.5 bpw", C[2])
ax.text(2.4, 5.55, "byte i holds weight i in its low nibble and weight i+16 in its high nibble; value = d × (nibble − 8)",
        color=MUTED, fontsize=11)
# Q8_0
row(4.3, "Q8_0", "block of 32 weights", 2, 32, "d", lambda b: f"{b}", "34 bytes / 32 = 8.5 bpw", C[0])
ax.text(2.4, 3.95, "one signed byte per weight; value = d × int8", color=MUTED, fontsize=11)
# NVFP4
row(2.7, "NVFP4", "block of 16 values", 1, 8, "s", lambda b: f"{2*b}|{2*b+1}", "9 bytes / 16 = 4.5 bpw", C[1])
ax.text(2.4, 2.35, "E4M3 scale per 16, two E2M1 floats per byte, plus one fp32 scale per tensor; value = s × E2M1 × tensor_scale",
        color=MUTED, fontsize=11)
# Q4_K super-block
ax.text(0.2, 1.55, "Q4_K", color=INK, fontsize=14, fontweight="bold", va="center")
ax.text(0.2, 1.2, "super-block of 256", color=INK2, fontsize=11, va="center")
x = 2.4; bw = 11.2 / 144
cell(x, 1.15, bw * 2 * 0.96, 0.78, "d", C[3], C[3], fs=10, tc="#f3eadb", weight="bold"); x += bw * 2
cell(x, 1.15, bw * 2 * 0.96, 0.78, "m", C[3], C[3], fs=10, tc="#f3eadb", weight="bold"); x += bw * 2
cell(x, 1.15, bw * 12 * 0.96, 0.78, "8×(6b,6b)", C[3], C[3], fs=9, tc="#f3eadb"); x += bw * 12
for sub in range(8):
    cell(x, 1.15, bw * 16 * 0.96, 0.78, f"sb{sub}: 16 B", DIM, C[4], fs=10); x += bw * 16
ax.text(2.4 + 11.2 + 0.15, 1.54, "144 bytes / 256 = 4.5 bpw", color=C[4], fontsize=12, va="center")
ax.text(2.4, 0.8, "fp16 d and fp16 m, then eight 6-bit sub-scales and 6-bit sub-minimums (12 bytes), then 8 sub-blocks of 32 nibbles; value = d·scale_j × nibble − m·min_j",
        color=MUTED, fontsize=11)
ax.text(8.0, 7.05, "The decoder is a kernel: each thread reads its block's scale and its own nibble. Nothing is variable-length.",
        ha="center", color=INK, fontsize=14)
fig.savefig("/home/dima/Work/chooseyourcodec/docs/map/zoo/img/block-float-formats/q4-layout.png")
print("ok")
