---
title: "Block-scaled formats: blocky"
slug: block-float-formats
chair: quantize
family: "Block-scaled integer and float formats"
summary: "Why a GPU wants 4.5 bits per weight rather than 4 and refuses an entropy coder: the decoder is a kernel, the scale is amortised over a block, and every format on Hugging Face is a point on one curve."
fighters: [blockint, fp4, escape, evict]
related: [scalar-quantization, rotations-and-rank, kv-cache-vaults, attention-as-audience]
reading_time: 14
---

# Block-scaled formats: blocky

Imagine an engineer in the spring of 2023, call him Teo, with a laptop that has 64 GB of memory and a 70-billion-parameter language model he wants to run on it. The arithmetic is short and discouraging. Seventy billion weights at two bytes each (the bf16 the model was trained in) is 140 GB. To fit beside the operating system and the working memory of inference he needs the weights under about 40 GB, which is 4.6 bits per weight. He knows the [scalar quantization](../scalar-quantization/) article: round to a grid, and every halving of the step costs a bit. Four bits it is, then. Two questions will take him the rest of the week. *Why does every working format end up at 4.5 bits rather than 4?* And, from a man who has read the [entropy-coding](../prefix-codes/) articles, the more irritating one: *the rounded values are heavily skewed, so why does nobody put a Huffman or ANS coder behind them?*

The answers are the same answer, and it is not about the data. It is about who the decoder is.

## First attempt: one scale for the tensor

The obvious scheme is the one every signal-processing textbook gives: pick a step size for the whole tensor, divide, round, store the integer. Teo picks the step so that the largest weight lands on the top of the 4-bit range. Here is what it does to eight weights, seven of them ordinary and one of them, as real weight matrices always have, a stray giant.

```python
import numpy as np
w = np.array([0.31, -0.52, 0.18, 0.09, -0.24, 0.43, -0.12, 6.0])   # eight weights, one outlier
scale = np.abs(w).max() / 7                  # int4 symmetric: levels -7..7
q = np.round(w / scale)
print("scale", scale.round(3), " codes", q.astype(int))
print("decoded", (q * scale).round(2), " MSE", ((q * scale - w) ** 2).mean().round(4))
small = w[:7]; s2 = np.abs(small).max() / 7   # the same seven without the outlier
print("without the outlier: scale", s2.round(3), " codes", np.round(small / s2).astype(int),
      " MSE", ((np.round(small / s2) * s2 - small) ** 2).mean().round(5))
```
```
scale 0.857  codes [ 0 -1  0  0  0  1  0  7]
decoded [ 0.   -0.86  0.    0.   -0.    0.86 -0.    6.  ]  MSE 0.0631
without the outlier: scale 0.074  codes [ 4 -7  2  1 -3  6 -2]  MSE 0.00039
```

One value of 6.0 forces a step of 0.857 on everyone, and five of the seven ordinary weights round to zero: the outlier is stored perfectly and its neighbours are erased. Take it out of the picture and the same seven weights get a step of 0.074, use the whole code range, and the squared error drops 160-fold. A single weight matrix in a transformer has tens of millions of entries and a tail of outliers with it, and the per-tensor grid is sized by the worst of them. Teo's first 4-bit model does not talk; it produces word salad.

The scalar article already gave the diagnosis, *outliers set the scale*, and listed the cures: escape them to a side list, rotate them away, or shrink the group that shares a scale. Teo reaches for the last one, because it is the cheapest, and it leads straight to the first real question.

## Second attempt: the Packer, refused

Before shrinking anything, notice how skewed those codes are. In a 4-bit tensor of Gaussian-ish weights, the codes near zero are far more common than the codes near the edge; a Huffman tree over the sixteen symbols might average three bits, and an [arithmetic coder](../arithmetic-and-ans/) would do a little better. A 25% saving for free. Teo writes the coder, measures the saving, and then writes the matrix-multiply kernel that has to consume the result.

That is where it dies. A GPU matmul is tens of thousands of threads, each responsible for a handful of outputs, each needing *its own* weights, now: thread 3,187 wants row 42, columns 96 to 127, and it wants them without reading anything else. With a fixed-width code that address is arithmetic: byte offset $42 \cdot \text{row\_stride} + 96/2$. With a variable-length code there is no such arithmetic. To find where column 96 begins, a thread would have to decode columns 0 through 95 first, serially, because a prefix code only tells you where a symbol ends once you have read it. The thread would spend its time walking a bit stream while thirty thousand neighbours wait. One could add checkpoints every $n$ columns, each a stored offset, and then the checkpoints are side information and the decoder still walks from the nearest one; the game's Random Access Race quest is exactly this experiment, and there is no setting where the variable-length layout wins on both size and speed.

So the first aha is a constraint, not a trick: **the decoder is a kernel**. Its ideal is one multiply per element, every thread addressing its own data, nothing serial. Variable-length codes are impossible in that setting, so the format gives up the Packer's chair on purpose and pays the nominal width, four bits, for every weight, however predictable. The compression it buys is random access. Research codecs that do entropy-code weights exist, and they all do the same thing at load time: decompress to the int4 layout *before* the kernel runs. The Packer works on the disk and the network; it never works in HBM.

## Third attempt: blocks

Now shrink the group. Instead of one scale per tensor, store one scale per block of $n$ consecutive weights, in a higher-precision format, and let each block set its own step. The outlier at index 7 still erases its neighbours, but now only the $n - 1$ of them in its block. Everything else in the tensor is untouched. The price is the scale itself: with a 16-bit scale per block, the storage per weight is

$$
\text{bits/weight} = \frac{16 + n\,b}{n} = b + \frac{16}{n},
$$

where $b$ is the nominal width. The 16 bits are amortised over the block; the smaller the block, the better the outliers are contained and the more the scale costs. This is the second aha and it is the whole design space in one line.

```python
import numpy as np
def bpw(scale_bits, n, b): return (scale_bits + n * b) / n
print("Q4_0  (fp16 scale, 32 × int4):        ", bpw(16, 32, 4))
print("Q4_1  (fp16 scale + fp16 min, 32 × 4): ", bpw(32, 32, 4))
print("Q8_0  (fp16 scale, 32 × int8):        ", bpw(16, 32, 8))
print("Q4_K  (256 weights: 2 fp16 + 8×(6+6)-bit sub-scales/mins + 256 × 4):", (32 + 96 + 1024) / 256)
print("NVFP4 (fp8 scale per 16 × E2M1):      ", bpw(8, 16, 4))
print("MXFP4 (8-bit exponent per 32 × E2M1): ", bpw(8, 32, 4))
for n in (8, 16, 32, 64, 128, 256):
    print(f"int4 with an fp16 scale per {n:3d}: {bpw(16, n, 4):.3f} bits/weight")
```
```
Q4_0  (fp16 scale, 32 × int4):         4.5
Q4_1  (fp16 scale + fp16 min, 32 × 4):  5.0
Q8_0  (fp16 scale, 32 × int8):         8.5
Q4_K  (256 weights: 2 fp16 + 8×(6+6)-bit sub-scales/mins + 256 × 4): 4.5
NVFP4 (fp8 scale per 16 × E2M1):       4.5
MXFP4 (8-bit exponent per 32 × E2M1):  4.25
int4 with an fp16 scale per   8: 6.000 bits/weight
int4 with an fp16 scale per  16: 5.000 bits/weight
int4 with an fp16 scale per  32: 4.500 bits/weight
int4 with an fp16 scale per  64: 4.250 bits/weight
int4 with an fp16 scale per 128: 4.125 bits/weight
int4 with an fp16 scale per 256: 4.062 bits/weight
```

There is Teo's 4.5. The GGUF format that llama.cpp made ubiquitous chose a block of 32 with an fp16 scale and called it `Q4_0`: 18 bytes per 32 weights. Its sibling `Q4_1` adds an fp16 minimum so the grid need not be symmetric about zero (5.0 bits). `Q8_0` keeps the same block and widens the codes to a byte (8.5 bits). The K-quants nest the idea: `Q4_K` groups 256 weights into a super-block with two fp16 super-scales, then gives each of eight sub-blocks of 32 a 6-bit scale and a 6-bit minimum *relative to the super-scales*, a scale of scales, so that the per-32 granularity of `Q4_1` costs 4.5 bits instead of 5.0. (`Q4_K_M`, the file most people download, mixes `Q4_K` with 6-bit blocks for the tensors that hurt most, and lands a little above 4.5.) Every one of these is a point on the curves in the figure below, and so is every quantized checkpoint on Hugging Face.

{{< fig src="img/block-float-formats/bits-vs-block.png" alt="Bits per weight against block size on a log axis, one curve per nominal width from 2 to 8 bits, each falling toward its width as the block grows; diamonds mark Q4_0, Q4_1, Q8_0, NVFP4, MXFP4 and KIVI's int2." caption="bits/weight = b + scale_bits/n. The nominal width is a floor the format can never go below (no Packer), and the scale is a tax that shrinks with the block. Every named format is a point on one of these curves." >}}

Now the definition, properly. A **block-scaled quantizer** with block size $n$ and width $b$ represents a block $\mathbf{w} \in \mathbb{R}^n$ by a scale $d$ and integer codes $q_i \in \{-2^{b-1}, \dots, 2^{b-1} - 1\}$, with

$$
q_i = \operatorname{round}\!\left(\frac{w_i}{d}\right), \qquad \hat{w}_i = d\, q_i, \qquad d = \frac{\max_i |w_i|}{2^{b-1} - 1}
$$

for the **absmax** (symmetric) variant, or, for the **min-max** (asymmetric) variant, $\hat{w}_i = d\, q_i + m$ with $d = (\max \mathbf{w} - \min \mathbf{w})/(2^b - 1)$, $m = \min \mathbf{w}$, and unsigned codes. The asymmetric form spends a second number per block to use all $2^b$ levels on a lopsided block; the symmetric form wastes a level or two and saves the storage. The decoder in either case is one fused multiply-add per element, and that is the point.

{{< fig src="img/block-float-formats/error-vs-block.png" alt="Mean squared error of int4 absmax quantization against block size, on 4,096 Gaussian weights with 1% outliers at 20 times the typical magnitude, with the bits-per-weight at each block size labelled; a second curve with no outliers stays an order of magnitude lower and nearly flat." caption="On data with 1% outliers, the error rises steadily with block size as each outlier ruins more neighbours; at 4,096 it is the per-tensor disaster from the first attempt. Without outliers the block size barely matters. Blocks are an outlier-containment device, and their cost is the scale bits." >}}

Pay attention to what the figure is *not* saying. The error without outliers hardly moves between $n = 8$ and $n = 4096$. Blocks are not a general improvement in quantization; they are an outlier-containment device, and their value is entirely a function of how heavy the tails are. That is why the same curve is read differently for weights (mild tails: 32 is plenty) and for activations and keys (which we will get to).

{{< fig src="img/block-float-formats/q4-layout.png" alt="Byte layout diagrams of a Q4_0 block (2-byte fp16 scale then 16 bytes of paired nibbles), a Q8_0 block (2-byte scale then 32 signed bytes), an NVFP4 block (1-byte fp8 scale then 8 bytes of paired E2M1 values) and a Q4_K super-block (two fp16 super-scales, 12 bytes of 6-bit sub-scales and minimums, then 8 sub-blocks of 32 nibbles)." caption="The layouts, byte by byte. In Q4_0, byte i holds weight i in its low nibble and weight i+16 in its high nibble, so a thread that unpacks one byte gets two weights that sit 16 apart; the kernel is written around the layout, and the layout around the kernel." >}}

{{< lab id="blockint" >}}

## A floating grid: NVFP4

Everything so far rounds to a uniform grid inside the block. Teo's weights are bell-shaped, dense near zero and sparse at the edges, and the scalar article's companding lesson applies: a grid whose levels crowd together near zero and spread apart near the edges wastes fewer levels on empty territory. A 4-bit floating-point number is exactly such a grid. **E2M1** (one sign bit, two exponent bits, one mantissa bit) has the fifteen levels $\pm\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$: steps of 0.5 up to 2, then 1, then 2. NVIDIA's **NVFP4** stores a block of 16 of these with one **E4M3** 8-bit floating scale, plus one fp32 scale for the whole tensor, and the Blackwell tensor core reads the format natively: the multiply-by-scale happens inside the matmul unit. Bits per element: $(8 + 16 \times 4)/16 = 4.5$, the same as `Q4_0`, with the scale budget spent on a block half the size.

```python
import numpy as np
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6]); E2M1 = np.concatenate([-E2M1[::-1], E2M1[1:]])  # 15 levels
blk = np.array([0.12, -0.35, 0.80, -0.05, 0.22, -0.61, 0.09, 1.9, -0.14, 0.41, -0.27, 0.03, 0.55, -0.73, 0.18, -0.44])
def q_fp4(x):                      # NVFP4-style: absmax lands on 6, snap to the nearest E2M1 level
    s = np.abs(x).max() / 6
    return E2M1[np.abs(x[:, None] / s - E2M1[None]).argmin(1)] * s
def q_int4(x):                     # int4 absmax: absmax lands on 7, uniform steps
    s = np.abs(x).max() / 7
    return np.round(x / s) * s
for name, x in [("plain block ", blk), ("with outlier", np.where(np.arange(16) == 7, 8.0, blk))]:
    print(name, " MSE fp4", ((q_fp4(x) - x) ** 2).mean().round(4), " MSE int4", ((q_int4(x) - x) ** 2).mean().round(4))
x = np.where(np.arange(16) == 7, 8.0, blk)
print("outlier block, fp4 :", q_fp4(x).round(2))
print("outlier block, int4:", q_int4(x).round(2))
```
```
plain block   MSE fp4 0.0038  MSE int4 0.0063
with outlier  MSE fp4 0.0285  MSE int4 0.0974
outlier block, fp4 : [-0.   -0.67  0.67 -0.   -0.   -0.67 -0.    8.   -0.    0.67 -0.   -0.
  0.67 -0.67 -0.   -0.67]
outlier block, int4: [ 0.   -0.    1.14 -0.    0.   -1.14  0.    8.   -0.    0.   -0.    0.
  0.   -1.14  0.   -0.  ]
```

On the plain block the float grid is modestly better. With an 8.0 dropped in, the int4 step becomes $8/7 = 1.14$ and every weight under 0.57 in magnitude vanishes; the float grid's smallest non-zero level is $0.5 \times 8/6 = 0.67$, so anything above 0.33 survives, and the error is three and a half times lower. (On exactly Gaussian blocks with no outliers int4 wins by a hair, which is the honest version: the float grid is a bet on heavy tails, and tensors pay off that bet.) The **MX** formats standardised by the Open Compute Project make the other trade: the same E2M1 elements, but a block of 32 sharing a pure power-of-two exponent (E8M0) instead of a fp8 scale, 4.25 bits per element and a cheaper scaling circuit. In production, SGLang measured FP8-to-NVFP4 on a 27B model's KV cache at $-0.3$ points on GSM8K and $-1.0$ on GPQA-Diamond, for 1.78× the cache capacity; TensorRT-LLM uses the same format to encode KV pages as they migrate to host memory and decode them on the way back, with attention never reading the cold form.

{{< lab id="fp4" >}}

## The encoder may be as clever as it likes

Nothing in a block format says how the codes must be chosen. Rounding to nearest is the cheapest encoder, not the best one, and because a model is quantized once and run a billion times, the encoder can afford to be expensive. **GPTQ** chooses each column's codes to minimise the error of the layer's *output* on calibration data, using a Hessian of the inputs to decide how to push rounding error onto columns that matter less. **AWQ** notices that a weight's importance is set by the activations that multiply it, and scales channels before quantization so that the important ones get finer effective steps. Both produce exactly the `Q4`-shaped layout above; the kernel cannot tell them apart from naive rounding. That asymmetry, lavish encoder and trivial decoder, is the [job's](../anatomy-of-a-job/) signature: encode once, decode on every token.

## Keys, values, and which way to slice

Teo's model now fits. The next thing that does not fit is the KV cache, which grows with every token of context and, unlike the weights, is written *during* inference by the model itself; there is no calibration pass, no offline GPTQ. It is the same block format with the same kernel constraint, and it introduces the third aha, because the key tensor has a property the weight tensors did not. In every decoder-only model studied, a few fixed channels of the keys carry magnitudes far larger than the rest, the same channels for every token. The values show no such pattern. So the question is not just how big a block, but *which direction it runs in*.

```python
import numpy as np
rng = np.random.default_rng(1)
K = rng.normal(0, 1, (8, 8)).round(2)       # 8 tokens (rows) × 8 channels (columns) of a key
K[:, 2] += 20; K[:, 5] -= 15                # two outlier channels, the same ones for every token
def q_int4(M, axis):                        # one absmax scale per row (axis=1) or per column (axis=0)
    s = np.abs(M).max(axis=axis, keepdims=True) / 7
    return np.round(M / s) * s
print("per-token   (one scale per row):   MSE", ((q_int4(K, 1) - K) ** 2).mean().round(3))
print("per-channel (one scale per column): MSE", ((q_int4(K, 0) - K) ** 2).mean().round(3))
print("token 0 per-token decoded:", q_int4(K, 1)[0].round(2))
print("token 0 original:         ", K[0])
def q_int4_asym(M, axis):                   # KIVI-style: min/max per group, 16 levels, zero point
    lo, hi = M.min(axis=axis, keepdims=True), M.max(axis=axis, keepdims=True)
    s = (hi - lo) / 15
    return np.round((M - lo) / s) * s + lo
print("per-channel with a zero point (asymmetric): MSE", ((q_int4_asym(K, 0) - K) ** 2).mean().round(4))
```
```
per-token   (one scale per row):   MSE 0.474
per-channel (one scale per column): MSE 0.144
token 0 per-token decoded: [  0.     0.    20.33  -0.     0.   -14.52  -0.     0.  ]
token 0 original:          [  0.35   0.82  20.33  -1.3    0.91 -14.55  -0.54   0.58]
per-channel with a zero point (asymmetric): MSE 0.0021
```

A scale shared across the channels of one token is set by the outlier channel, and the token's six ordinary channels are erased, which is the first attempt's disaster repeated on every row. A scale shared across the *tokens* of one channel sees a stationary magnitude: channel 2 is always about 20, channel 0 is always about 1, and each gets its own grid. Add a zero point, because a channel that lives at 18.7 to 20.7 does not need levels at zero, and the error falls by two more orders of magnitude. KIVI (ICML 2024) is this observation as a product: keys quantized per channel in groups of 32 tokens, values per token in groups of 32 channels, 2-bit codes with a scale and zero point per group, no calibration at all, and the newest 128 tokens kept in fp16 until a full group has arrived. Its own ablation on Llama-2-13B at 2 bits tells the story in GSM8K accuracy: fp16 22.67; keys per channel with values per token, 12.21; keys per token, 4.55; any configuration with values per channel, essentially zero; the full method with its fp16 window, 20.77. One decision about which way to slice is worth three orders of magnitude on the task.

Two more fighters belong to this family as its limiting cases. When an outlier channel is too wild even for per-channel scales, or the outliers are scattered rather than channel-aligned, the **escape hatch** pulls them out: every vector gets a threshold, the top 1% of values skip the grid and ride in a sparse fp16 side list with their indices, and the remaining 99% get a grid with a tenth of the step. The cost is $3 + 0.01 \times (16 + \text{index bits}) \approx 3.3$ bits per element for a 3-bit base; KVQuant's 1% outliers take LLaMA-7B from 5.94 to 5.75 perplexity at 3 bits (fp16: 5.68). The price is a sparse gather on the GPU, which is the one thing a kernel hates, and it is partly why [coupled quantization](../vector-quantization/) exists. And at the far end, **eviction** is the degenerate block format: a two-level quantizer whose levels are "full precision" and "nothing", chosen per token. Everything about rate and distortion still applies; what changes is that the distortion is a benchmark score and the choice of what to drop is the [consumer model's](../attention-as-audience/) job. The systems trap is memory: paged allocators free a block only when every token in it is gone, so evicting 14,400 of 16,000 tokens can leave 1,600 survivors spread across a thousand blocks and free almost nothing without a compaction pass. Quantization keeps the page layout intact, which is the unglamorous reason it is what production stacks actually ship.

{{< lab id="escape" >}}

{{< lab id="evict" >}}

The other cure for outlier channels is to rotate them away before quantizing, so that every channel has the same statistics and per-token blocks become safe again; that is QuaRot and SpinQuant, and it belongs to the [rotations](../rotations-and-rank/) article. RoPE is the complication: it rotates channel pairs by a position-dependent angle, smearing an outlier channel's energy between two channels in a way that varies with the token, so per-channel statistics stop being stationary. Methods that quantize keys *before* RoPE and re-apply it after decoding (KVQuant, CQ, KVTC, TensorRT-LLM's `skip_rope_quantization`) are undoing a known transform so that the block quantizer sees a stationary signal.

## Connections

**Block floating point** is a 1970s digital-signal-processing idea: a fixed-point FFT keeps one shared exponent for a whole block of samples and renormalises after each stage, so a cheap integer multiplier gets most of the dynamic range of a floating unit. `Q4_0` is block floating point with a 16-bit exponent and 4-bit mantissas, and NVFP4 is block floating point with a second, tiny exponent inside each element.

**Floating point itself** is the $n = 1$ case. An fp32 number is a 23-bit mantissa with its own 8-bit scale; by the formula above that is $23 + 8/1 = 31$ bits of storage per value for the mantissa and scale, the most outlier-robust and least amortised point on the curve. Every block format is an argument that neighbouring values can share an exponent.

**JPEG's quantization table** is a block format read the other way: one step size per DCT coefficient, shared across *all* the blocks of the image, so the block size along which the scale is amortised is the whole picture. The [scalar quantization](../scalar-quantization/) article's water-filling allocation and KVTC's per-component steps are the same shape.

The **Random Access Race** quest in the game is the second section as an experiment: fixed 8-bit codes against 5.1-bit Huffman codes with checkpoints, and the discovery that no checkpoint spacing wins on both bytes and decode steps. That is the trade this entire family accepts in advance.

**In the game:** BLOCKY (blockint) and QUARTER (fp4) hold the Butcher's chair in THE WEIGHTS VAULT (GGUF Q4_K_M) and THE CONTEXT WINDOW (KIVI); QUARTER returns for THE COLD STORAGE (TensorRT-LLM cold-page NVFP4); ESCAPE HATCH (escape) and the REAPER (evict) are the family's two limits, in THE ONE-BIT VAULT and THE PURGE; and the empty Packer's chair in all of them is the reveal of the Random Access Race.
