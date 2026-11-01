---
title: "Scalar quantization: placing the grid"
slug: scalar-quantization
chair: quantize
family: "Scalar quantizers"
summary: "Rounding is easy; where should the grid be fine and where coarse? Lloyd-Max's answer for a fixed number of levels, why an entropy coder makes the uniform grid almost unbeatable, μ-law's logarithmic ear, JPEG's table of sixty-four step sizes, and the outlier that ruins everyone's grid."
fighters: [uniform, companding, qmatrix, waterline, escape]
related: [rate-and-distortion, block-transforms, block-float-formats, perceptual-models]
reading_time: 16
---

## Shoe sizes

Feet grow continuously. Shoes come in sizes. Somewhere, somebody decided that the sizes would be 9, 9.5, 10 and so on, and that decision, repeated across the whole population, determines how much every person's feet hurt. Make the steps coarser and the factory stocks fewer sizes but the average foot is squeezed harder. Make them finer and the warehouse fills with half-sizes nobody buys. That is the rate-distortion trade from the [previous article](../rate-and-distortion/), and it has a known price: every halving of the step costs a bit and buys 6 dB.

But there is a second question hiding inside the first, and it is the one this article is about. Feet are not spread evenly across sizes. Most adults cluster around a few sizes and almost nobody is a 15. Should the sizes be evenly spaced at all? Or should they crowd together where the feet are and spread out where they are not? If you could place eight sizes anywhere on the number line, where would you put them?

> *Rounding is easy; where should the grid be fine and where coarse?*

Three generations of engineers answered this for three different reasons, and their answers, the Lloyd-Max levels, the μ-law curve, the JPEG table, and the outlier escape, are what fill the scalar quantizer's stage.

## Lloyd's lunch problem

Stuart Lloyd worked at Bell Laboratories in the 1950s, when the company was turning telephone speech into pulse-code modulation: sample the voltage 8,000 times a second, round each sample to one of a fixed number of levels, send the level's index. The number of levels was fixed by the hardware, 128 or 256, and there was no entropy coder in the picture; each sample cost exactly seven or eight bits regardless of which level it hit. Under that constraint the only freedom left is *where the levels sit*, and Lloyd's 1957 internal memo (it was not formally published until 1982; Joel Max found the same answer independently in 1960) asked the exact question above. Imagine him with a plotted histogram of speech samples on the desk: a tall spike at zero, long thin tails. A uniform grid wastes half its levels out in the tails where almost nothing lands.

Here is the problem for a bell-shaped source with eight levels, three bits. First, the best *uniform* grid, found by sweeping the step; then Lloyd's iteration started from that grid:

```python
import numpy as np
g = np.linspace(-6, 6, 240_001); pdf = np.exp(-g**2 / 2) / np.sqrt(2 * np.pi); pdf /= pdf.sum()   # unit Gaussian
def mse(levels):
    idx = np.abs(g[:, None] - levels[None, :]).argmin(1); return np.sum(pdf * (g - levels[idx]) ** 2)
# 3 bits = 8 levels. Best *uniform* grid: sweep the step.
steps = np.linspace(0.3, 1.2, 901)
unif = [mse((np.arange(8) - 3.5) * s) for s in steps]
s_best = steps[np.argmin(unif)]
print(f"uniform 8 levels: best step={s_best:.3f}  MSE={min(unif):.5f}")
# Lloyd: start from that uniform grid, alternate (nearest level) / (conditional mean)
lv = (np.arange(8) - 3.5) * s_best
for it in range(60):
    idx = np.abs(g[:, None] - lv[None, :]).argmin(1)
    lv = np.array([np.sum(pdf[idx == k] * g[idx == k]) / np.sum(pdf[idx == k]) for k in range(8)])
    if it in (0, 4, 59): print(f"iter {it+1:2d}: levels={lv[4:].round(4).tolist()} (±)  MSE={mse(lv):.5f}")
print(f"Lloyd-Max gain over the best uniform grid: {10*np.log10(min(unif)/mse(lv)):.2f} dB")
```
```
uniform 8 levels: best step=0.586  MSE=0.03744
iter  1: levels=[0.2847, 0.8542, 1.424, 2.1607] (±)  MSE=0.03549
iter  5: levels=[0.2614, 0.798, 1.3957, 2.1986] (±)  MSE=0.03473
iter 60: levels=[0.2452, 0.7561, 1.3441, 2.1521] (±)  MSE=0.03455
Lloyd-Max gain over the best uniform grid: 0.35 dB
```

The iteration has two steps, and each is the obvious thing. Given the levels, every sample should go to its *nearest* level (so the decision boundaries sit halfway between neighbours). Given the boundaries, every level should sit at the *centroid*, the conditional mean, of the samples in its bin (because the mean minimises squared error within the bin). Alternate until nothing moves. The levels crowd inward: the innermost pair ends at ±0.245, the outermost at ±2.152, so the gaps grow from 0.49 near zero to 0.81 in the tails. That is the **Lloyd-Max quantizer**: the set of levels $\{y_k\}$ and boundaries $\{b_k\}$ satisfying the two **necessary conditions** $b_k = (y_k + y_{k+1})/2$ and $y_k = \mathbb{E}[X \mid b_{k-1} < X \le b_k]$. The algorithm is a coordinate descent on the MSE and converges to a local minimum, which for a single bump like the Gaussian is the global one. The same two conditions, in $d$ dimensions, are k-means, and the [vector quantization article](../vector-quantization/) is Lloyd's algorithm with the dimension turned up.

{{< fig src="img/scalar-quantization/lloyd_max.png" alt="Two panels over a Gaussian density: the best uniform 8-level grid with step 0.586 and MSE 0.03744, and the Lloyd-Max levels ±0.245, ±0.756, ±1.344, ±2.152 with MSE 0.03455" caption="Eight levels for a unit Gaussian. The uniform grid (top) spends two of its eight levels beyond ±2, where 4.6% of the probability lives. Lloyd-Max (bottom) pulls them in; the gain is 0.35 dB, modest, and that modesty is the first clue." >}}

Now notice the size of the prize. All that iteration bought 0.35 dB, a twentieth of a bit's worth. The reason is instructive, and it sets up the surprise that follows: the uniform grid was already spending its levels roughly where the density was, because a uniform grid on a density that is not too wild is not far from optimal. Lloyd-Max matters most at *very* low rates, one or two bits, and for *very* peaked densities. At eight levels on a bell curve the grid was most of the way there.

## The surprise: with an entropy coder, uniform is nearly the best scalar scheme

Lloyd's constraint was that every level costs the same number of bits. Drop it. Put an entropy coder behind the quantizer, so that a level hit often costs few bits and a level hit rarely costs many. Now ask Lloyd's question again: where should the levels go?

The answer, due to Gish and Pierce in 1968, is the single most useful fact in this stage and the first thing to take from the article. At high rate, the best scalar quantizer followed by an ideal entropy coder is the *uniform* one, and it sits 0.254 bits per sample above Shannon's $R(D)$ floor for any smooth source. The [rate and distortion article](../rate-and-distortion/) measured that gap on Gaussian samples and found 0.256. Lloyd-Max's crowded levels help a fixed-rate coder because they lower the error without changing the bit cost; with an entropy coder, crowding the levels where the density is high just makes the common levels *more* common and the rare ones rarer, and the entropy coder was already charging for that. The two effects cancel to within a rounding error, and what is left is the quarter bit that no scalar scheme can recover.

So the design freedom of a scalar quantizer behind an entropy coder is not where the levels go. It is two other things: the **step size per coordinate**, which is the bit allocation of the previous article, and the **dead zone**, the width of the bin around zero. Both are worth money and both are cheap to implement. Zeros are the most common symbol after any transform, an entropy coder charges a fraction of a bit for them, and widening the zero bin trades a little distortion for a lot of rate; on a Laplacian coefficient the previous article found a 10% distortion gain at equal rate from H.264's rounding offset of one third. Everything in the rest of this article is either a different $\Delta$ for a different coordinate, or a way of fixing what happens when a single coordinate's $\Delta$ is set by a value that does not belong. (The six toy values with a Δ slider are the Uniform quantizer lab, embedded in the [rate and distortion article](../rate-and-distortion/); drag it there and watch the levels halve and the error quadruple.)

## 1960s: the logarithmic ear

Telephony in the 1960s could not afford an entropy coder either, and it had a worse problem than Lloyd's. The criterion was not mean squared error. A listener hears *relative* error: a rounding error of 0.004 on a sample of amplitude 1 is inaudible; the same 0.004 on a whisper of amplitude 0.01 is a 40% distortion and sounds like gravel. A uniform grid treats both samples the same because its step is the same everywhere. What you want is a step that grows with the signal, so that the error is always a fixed *fraction* of the sample. That is a grid whose spacing is logarithmic.

The implementation is one of the field's oldest tricks and still runs on every landline: **companding**, compress-then-expand. Push the sample through a curve that stretches small values and squashes large ones, quantize the result *uniformly*, and apply the inverse curve on decode. The North American curve, with $\mu = 255$, is

$$
F(x) = \operatorname{sgn}(x)\,\frac{\ln(1 + \mu|x|)}{\ln(1+\mu)}, \qquad F^{-1}(y) = \operatorname{sgn}(y)\,\frac{(1+\mu)^{|y|} - 1}{\mu}.
$$

Take a quiet sample, $x = 0.01$, and compare 8-bit uniform against 8-bit μ-law coded the way G.711 does it, a sign bit plus seven magnitude bits:

```python
import numpy as np
mu = 255.0
F    = lambda x: np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)       # compress
Finv = lambda y: np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu))) / mu       # expand
x = 0.01
# 8-bit uniform: 256 levels on [-1, 1), Δ = 1/128
d = 1 / 128; xu = np.round(x / d) * d
# 8-bit μ-law: sign + 7 magnitude bits, 127 steps in the compressed domain
code = np.round(F(x) * 127); xm = Finv(code / 127)
print(f"uniform: code={int(np.round(x/d))}  x̂={xu:.6f}  rel err={abs(xu-x)/x:.1%}  worst at this x={(d/2)/x:.1%}")
print(f"μ-law:   code={int(code)}  x̂={xm:.6f}  rel err={abs(xm-x)/x:.2%}  worst (any x≫1/μ)={256**(1/254)-1:.1%}")
print(f"compressed-domain value F(0.01)={F(x):.4f}: a 1% input uses {F(x):.0%} of the code range")
```
```
uniform: code=1  x̂=0.007812  rel err=21.9%  worst at this x=39.1%
μ-law:   code=29  x̂=0.009990  rel err=0.10%  worst (any x≫1/μ)=2.2%
compressed-domain value F(0.01)=0.2285: a 1% input uses 23% of the code range
```

The uniform coder has exactly one code, 1, for anything between 0.0039 and 0.0117; the whisper at 0.01 comes back as 0.0078, 22% off, and in the worst case 39%. μ-law has twenty-nine codes below 0.01, lands within 0.10%, and its worst case for any sample well above $1/\mu$ is a flat 2.2%, because consecutive reconstruction levels differ by a constant *ratio* of $256^{1/127}$. The price is paid on loud samples, where μ-law's step is coarser than uniform's, and the ear does not notice. Eight bits of μ-law are commonly said to sound like thirteen bits of linear PCM, and that is what the phone company shipped: 64 kbit/s per call, with a quantizer shaped like an ear.

{{< fig src="img/scalar-quantization/mulaw.png" alt="Left: the μ-law curve with the uniform grid in the compressed domain projected back onto the input axis, showing fine steps near zero; right: relative error against amplitude on log axes for 8-bit uniform (rising toward 100% for quiet samples) and 8-bit μ-law (flat under 2.2%)" caption="Bend the ruler. Equal steps along the compressed axis become fine steps near zero on the input axis; the 1% sample at x = 0.01 lands at F = 0.228, with a fifth of the code range beneath it. Right: relative error over three decades of amplitude. Uniform's error grows without bound as the signal gets quiet; μ-law's is bounded by 2.2% everywhere above 1/μ." >}}

{{< lab id="companding" >}}

State it generally. A **companded quantizer** is a uniform quantizer composed with a monotone **compressor** $F$ and its inverse; equivalently, a non-uniform quantizer whose local step size at $x$ is $\Delta / F'(x)$. Choose $F$ to be logarithmic and the step is proportional to $|x|$, which makes the *relative* error constant: the second thing to take from this article is that non-uniform levels are not about squeezing out a last 0.35 dB of MSE but about matching the grid either to the data's density (Lloyd-Max) or to the consumer's error criterion (μ-law). MP3 does the same with a power law, raising coefficient magnitudes to the 0.75 before rounding so that the quantization noise follows the signal's loudness. And the trick outlived telephony: the NF4 format used to fine-tune large language models on one GPU chooses its sixteen levels as quantiles of a Gaussian, so that each 4-bit code is equally likely under a bell-shaped weight distribution. It is Lloyd's insight for a 2023 consumer, a loss curve instead of an ear.

## 1990: sixty-four step sizes

Now jump to a committee room in 1990. The JPEG group has an 8×8 DCT in front of its quantizer, so instead of one coordinate there are sixty-four, each a different spatial frequency. The entropy coder exists this time (Huffman, with a run-length stage for zeros), so Gish and Pierce say: uniform levels, and spend your effort on the step per coordinate. Which step? The eye's contrast sensitivity falls steeply with spatial frequency, so an error in a high-frequency coefficient is far less visible than the same error in a low one. The answer is a table: a $\Delta(u,v)$ for every coefficient position, fine at the top left where the DC and the gentle gradients live, coarse at the bottom right.

Here is the standard luma table at quality 50 applied to the block that the Wikipedia JPEG article uses as its worked example (an 8×8 patch of a photograph):

```python
import numpy as np
block = np.array([[52,55,61,66,70,61,64,73],[63,59,55,90,109,85,69,72],[62,59,68,113,144,104,66,73],
                  [63,58,71,122,154,106,70,69],[67,61,68,104,126,88,68,70],[79,65,60,70,77,68,58,75],
                  [85,71,64,59,55,61,65,83],[87,79,69,68,65,76,78,94]], float)
Q = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56],
              [14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
              [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])   # JPEG luma table, quality 50
n = np.arange(8); C = np.cos((2 * n[None, :] + 1) * n[:, None] * np.pi / 16) * np.sqrt(2 / 8)
C[0] /= np.sqrt(2)                                  # orthonormal DCT-II matrix
coef = C @ (block - 128) @ C.T
q = np.round(coef / Q).astype(int)
print("DC coefficient", round(coef[0, 0], 2), "->", q[0, 0]); print(q)
print("zeros:", np.sum(q == 0), "of 64;  nonzero in the first 3 rows:", np.sum(q[:3] != 0))
rec = C.T @ (q * Q) @ C + 128
print("pixel RMS error after dequantizing:", round(np.sqrt(np.mean((rec - block) ** 2)), 2))
```
```
DC coefficient -415.37 -> -26
[[-26  -3  -6   2   2  -1   0   0]
 [  0  -2  -4   1   1   0   0   0]
 [ -3   1   5  -1  -1   0   0   0]
 [ -3   1   2  -1   0   0   0   0]
 [  1   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0]]
zeros: 44 of 64;  nonzero in the first 3 rows: 15
pixel RMS error after dequantizing: 5.98
```

Forty-four of the sixty-four coefficients are exactly zero, and they sit in one contiguous triangle at the high-frequency corner, which is why JPEG scans the block in a zigzag from top left to bottom right: the zeros become one long run ended by a single end-of-block symbol. The twenty survivors carry the picture; the pixel error after reconstruction is about six grey levels out of 255, invisible at normal viewing distance. The zero triangle is the dead zone and the bit allocation working together: a step of 100 on a coefficient whose typical magnitude is 3 rounds it to zero essentially always, which is reverse water-filling's "drowned" coordinate, implemented as a table entry.

{{< fig src="img/scalar-quantization/jpeg_table.png" alt="Three 8×8 grids: the JPEG luma quantization table as a heat map growing from 16 at top left to over 100 at bottom right, the toy block's DCT coefficients with energy concentrated top left, and the quantized block with 44 zeros in the lower-right triangle" caption="Left: the table, one Δ per frequency, coarse where the eye is blind. Middle: the toy block's coefficients, nearly all of the energy in the top-left corner. Right: coefficient divided by step and rounded, 44 zeros of 64." >}}

{{< lab id="qmatrix" >}}

The **quality slider** is the whole table times one number: for quality $q < 50$ the scale factor is $5000/q$ percent, for $q \ge 50$ it is $200 - 2q$ percent, and every entry is multiplied by it (clamped to 1–255). One knob, sixty-four steps, all walking their 6 dB per bit lines together. The relative shape of the table is the perceptual-model stage's contribution, a contrast-sensitivity curve compiled into integers, and the [perceptual models article](../perceptual-models/) explains where the numbers came from.

The general form of what JPEG does by hand is **bit allocation across coordinates**: given variances $\sigma_i^2$ after a transform, choose the step per coordinate so that total distortion is minimised at the bit budget. The previous article derived the answer, reverse water-filling, $R_i = \max(0, \tfrac12\log_2(\sigma_i^2/\theta))$. Here it is on the eight coordinate variances the game's Water-filling allocation lab uses:

```python
import numpy as np
var = np.array([64, 32, 16, 8, 4, 2, 1, 0.5])            # variances of 8 transform coordinates
theta = 2.0                                               # the water level
R = np.maximum(0, 0.5 * np.log2(var / theta))             # bits per coordinate
D = np.minimum(var, theta)                                # distortion per coordinate
print("bits   :", R.tolist(), " total", R.sum())
print("distort:", D.tolist(), " total", D.sum())
R_eq = R.sum() / 8                                        # same total, spread evenly
D_eq = var * 2.0 ** (-2 * R_eq)
print(f"equal allocation: {R_eq} bits each, total distortion {D_eq.sum():.2f}  "
      f"({10*np.log10(D_eq.sum()/D.sum()):.2f} dB worse)")
```
```
bits   : [2.5, 2.0, 1.5, 1.0, 0.5, 0.0, 0.0, 0.0]  total 7.5
distort: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 0.5]  total 13.5
equal allocation: 0.9375 bits each, total distortion 34.76  (4.11 dB worse)
```

JPEG's table was tuned by eye in 1990. In 2026, KVTC does the identical allocation for a transformer's key-value cache by running a dynamic program over PCA coordinates, with a 16-bit scale per group, and the result reads like the output above: a few coordinates at many bits, a long tail at zero. Same stage, same mathematics, thirty-six years and a different consumer apart.

## 2023: the outlier that sets the scale

The last answer to "where should the grid be fine" comes from a problem that neither Lloyd nor the JPEG committee had, and that anyone who has quantized a transformer's weights or activations has met on the first afternoon. The grid is *symmetric and bounded*: a 3-bit integer grid has levels $-3 \ldots 3$, and to use it you divide every value by a scale $s$ chosen so that the largest value fits. The scale is therefore set by the largest value. One outlier decides the step for everybody.

Here are eight values from the game's Outlier escape lab, seven of them small and one of them 0.55, quantized on a 3-bit grid two ways: with the scale set over all eight, and with the outlier pulled out and kept at full precision in a side list:

```python
import numpy as np
x = np.array([0.012, -0.031, 0.004, 0.550, 0.020, -0.008, 0.015, -0.027])
def int3(v, s):                       # symmetric 3-bit grid: integers -3..3, step s
    return np.clip(np.round(v / s), -3, 3) * s
s_all = np.abs(x).max() / 3           # absmax over all 8
xh_all = int3(x, s_all)
mask = np.abs(x) < 0.5                # escape: the outlier leaves the grid
s_esc = np.abs(x[mask]).max() / 3
xh_esc = x.copy(); xh_esc[mask] = int3(x[mask], s_esc)      # outlier kept at full precision
print(f"no escape : step={s_all:.4f}  x̂={xh_all.round(3).tolist()}\n            MSE={np.mean((x-xh_all)**2):.2e}  bits/elem=3")
print(f"escape    : step={s_esc:.4f}  x̂={xh_esc.round(3).tolist()}\n            MSE={np.mean((x-xh_esc)**2):.2e}  "
      f"bits/elem={3 + (16 + 3) / 8:.2f}  (7 on the grid + one fp16 value + a 3-bit index)")
print(f"step ratio {s_all/s_esc:.1f}×, MSE ratio {np.mean((x-xh_all)**2)/np.mean((x-xh_esc)**2):.0f}×")
```
```
no escape : step=0.1833  x̂=[0.0, -0.0, 0.0, 0.55, 0.0, -0.0, 0.0, -0.0]
            MSE=3.17e-04  bits/elem=3
escape    : step=0.0103  x̂=[0.01, -0.031, 0.0, 0.55, 0.021, -0.01, 0.01, -0.031]
            MSE=7.81e-06  bits/elem=5.38  (7 on the grid + one fp16 value + a 3-bit index)
step ratio 17.7×, MSE ratio 41×
```

With the outlier on the grid, the step is 0.18 and the other seven values, none bigger than 0.031, all round to zero: the quantizer reconstructs one number correctly and erases the rest. Pull the 0.55 out and the step drops eighteen-fold; the seven survivors come back within a few thousandths, and the error falls forty-one times. The side list is not free: a 16-bit value and a 3-bit index for one of eight elements adds 2.4 bits per element here. In a real tensor where the outliers are 1% rather than 12%, the same arithmetic gives $3 + 0.01 \times (16 + \text{index})$, about 3.3 bits per element, which is the number KVQuant reports for its dense-and-sparse decomposition.

{{< lab id="escape" >}}

This is the third thing to take away, and it is as much about systems as about mathematics. **Outliers set the scale.** A value of 30 among values of 1 forces a step of 4 on everything, and no amount of cleverness in level placement will recover what a step of 4 destroys. The cures are a family, and the map has an article for each: make the block smaller so that an outlier only ruins its own neighbours ([block float formats](../block-float-formats/), where the block of 32 with its own scale is exactly this); let the outliers escape to a side list, which is this algorithm, and which JPEG and H.264 did decades earlier with escape codes for oversized coefficients; rotate the data first so that the outlier's energy is smeared across all coordinates and no single one is huge ([rotations and rank](../rotations-and-rank/)); or, in video and in GPTQ, choose the rounding of each value with its downstream effect in mind rather than by nearest level. That last idea, **rate-distortion-optimised rounding**, deserves its one paragraph: a video encoder will round a coefficient *down* to zero when the bits saved are worth more than the distortion added at the current operating point, and GPTQ rounds each weight while compensating the error in the weights not yet rounded, using the Hessian of the layer's output to decide what "compensate" means. Both are the dead zone's logic generalised: the quantizer is allowed to be wrong on purpose where being wrong is cheap.

## Connections

An **analogue-to-digital converter** is this article in silicon, and its data sheet states the 6 dB per bit law as the formula for a full-scale sine wave, $\text{SNR} = 6.02N + 1.76$ dB; the 1.76 is the ratio between a sine's power and that of the uniform rounding error, nothing more.

**Dithering**, adding a small random signal before quantizing, looks like vandalism and is a quantizer-design decision: it decorrelates the error from the signal, turning the harmonic distortion of coarse steps into a flat hiss, which the ear prefers and which also lets the average of many quantized samples recover sub-step precision.

**Histogram equalisation** in image processing maps pixel values through their own cumulative distribution so that the result is uniform. That is a compressor $F$ chosen to be the data's CDF, and quantizing the equalised image uniformly is Lloyd's idea in the limit of many levels: every bin gets the same probability mass. NF4's quantile levels are the same construction for a Gaussian.

## In the game

The quantization stage holds all five of this article's algorithms: [Uniform quantizer](/compressionfight/#fighter=uniform) is the uniform step, [μ-law companding](/compressionfight/#fighter=companding) is μ-law and MP3's power law, [Quantization matrix](/compressionfight/#fighter=qmatrix) is JPEG's table, [Water-filling allocation](/compressionfight/#fighter=waterline) is bit allocation by water level, and [Outlier escape](/compressionfight/#fighter=escape) is the outlier side list. They appear in the Photographs task, the Music distribution task, the Streaming video task, the KV cache cold storage task and the KV cache at 1–2 bits task, and the synergy 'Quantization matrix + zigzag RLE + Huffman' is the forty-four zeros above turned into a pipeline.
