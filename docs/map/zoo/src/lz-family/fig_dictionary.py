"""Figure: compressed size of a ~300-byte JSON event vs the size of a preset DEFLATE dictionary (real zlib)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, zlib, json, os, random

BG, INK, MUTED, GRID = "#0b0714", "#e8e4f0", "#9b93aa", "#2a2238"
C = ["#3b8fd6", "#c4812a", "#2fa06e", "#b85fb0"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
                     "font.family": "DejaVu Sans", "font.size": 13})

random.seed(1)
def event(i):
    return json.dumps({"timestamp": 1700000000 + 60 * i, "level": "info", "service": "checkout-api",
        "region": random.choice(["eu-west-1", "us-east-1", "ap-south-1"]),
        "user_id": "u_%06x" % random.randrange(16**6),
        "event_type": random.choice(["cart_add", "cart_remove", "checkout_start", "page_view"]),
        "sku": "SKU-%05d" % random.randrange(100000), "quantity": random.randrange(1, 5),
        "price_cents": random.randrange(100, 20000), "currency": "EUR",
        "session_id": "s_%08x" % random.randrange(16**8), "latency_ms": random.randrange(5, 500),
        "client": random.choice(["ios", "android", "web"])}).encode()

train = b"".join(event(i) for i in range(1000, 1400))      # dictionary material
test = [event(i) for i in range(50)]                       # messages to compress
print("mean message bytes", np.mean([len(m) for m in test]))

def size_with_dict(msg, dic):
    co = zlib.compressobj(9, zlib.DEFLATED, -15, 9, 0, zdict=dic) if dic else zlib.compressobj(9, zlib.DEFLATED, -15)
    return len(co.compress(msg) + co.flush())

dsizes = [0, 100, 200, 300, 600, 1200, 2400, 4800, 9600, 19200, 32768]
means = []
for ds in dsizes:
    dic = train[-ds:] if ds else b""      # the last ds bytes of the training sample (recent = closest in the window)
    m = np.mean([size_with_dict(x, dic) for x in test]); means.append(m)
    print(f"dict {ds:>6} B -> mean {m:6.1f} B")

fig, ax = plt.subplots(figsize=(16, 7), dpi=100)
x = np.arange(len(dsizes))
ax.bar(x, means, color=C[0], width=0.72)
ax.axhline(np.mean([len(m) for m in test]), color=C[1], lw=1.4, ls="--")
ax.text(len(dsizes) - 0.6, np.mean([len(m) for m in test]) + 4, "raw message size", color=C[1], ha="right", fontsize=12)
for xi, m in zip(x, means):
    ax.text(xi, m + 3, f"{m:.0f}", ha="center", color=INK, fontsize=12)
ax.set_xticks(x); ax.set_xticklabels(["none"] + [f"{d} B" if d < 1000 else f"{d // 1024 if d >= 1024 else d} {'KB' if d >= 1024 else 'B'}" for d in dsizes[1:]])
ax.set_xlabel("preset dictionary handed to DEFLATE (bytes of similar events)")
ax.set_ylabel("compressed bytes per message (mean of 50)")
ax.grid(True, axis="y", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("A ~300-byte JSON event: what a shared window buys", loc="left", fontsize=15, pad=14)
out = os.path.join(os.path.dirname(__file__), "..", "..", "img", "lz-family", "dictionary.png")
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
from PIL import Image
w = Image.open(out).width                      # re-save so the trimmed image is exactly 1600 px wide
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.3, dpi=100 * 1600 / w)
print("wrote", out)
